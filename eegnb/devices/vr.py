import json
import logging
import csv
import numpy as np
from time import time
from psychopy import core, monitors
from psychopy.visual.rift import Rift


# Frame-rate deviation above this percentage triggers a user-facing warning.
# Set high (50%) because VR jitter is recoverable per-trial via the timing
# sidecar (app_motion_to_photon_latency_s). The warning catches fundamentally
# broken setups (wrong GPU, no acceleration), not borderline cases.
DISPLAY_DEVIATION_FLAG_PCT = 50.0


def _build_placeholder_monitor():
    """Programmatic Monitor passed to the Rift constructor.
    Display geometry comes from the HMD runtime, not this object.
    Logger is silenced during construction because Monitor.__init__ emits an
    unconditional 'Monitor specification not found' warning for any name not
    present in the user's saved calibration database."""
    from psychopy import logging as psy_logging
    prev_level = psy_logging.console.level
    psy_logging.console.setLevel(psy_logging.ERROR)
    try:
        mon = monitors.Monitor('eegnb_vr_placeholder', autoLog=False)
        mon.setDistance(60)
        mon.setSizePix([1920, 1080])
    finally:
        psy_logging.console.setLevel(prev_level)
    return mon

class VR(Rift):
    """
    Extended VR class for HMDs, providing built-in methods for
    stereoscopic rendering math, precise hardware clock synchronization,
    and per-trial compositor telemetry buffering.
    """
    def __init__(self, *args, **kwargs):
        # Provide a placeholder monitor so PsychoPy doesn't emit
        # 'Monitor specification not found' on first flip.
        kwargs.setdefault('monitor', _build_placeholder_monitor())
        # Disable vsync on the on-screen mirror window.
        kwargs.setdefault('waitBlanking', False)
        super().__init__(*args, **kwargs)
        # Belt-and-braces: also call set_vsync(False) directly on the pyglet
        # backend, in case waitBlanking didn't propagate to the GL context.
        try:
            self.backend.set_vsync(False)
        except Exception:
            try:
                # Older pyglet: vsync attribute on the underlying window.
                self.backend.winHandle.set_vsync(False)
            except Exception:
                pass
        self.libovr_to_wallclock_offset = None
        self.libovr_to_wallclock_bracket = None
        self.timing_data = []

        # Per-flip phase timing — populated by the overridden sub-methods on
        # each flip() and read by the experiment's per-frame logger. Splits
        # the 25 ms "flip block" into:
        #   end_frame_ms:    libovr.endFrame submit
        #   swap_buffers_ms: pyglet mirror-window swapBuffers (may vsync-lock
        #                    to the DESKTOP monitor — primary suspect)
        #   wait_begin_ms:   libovr.waitToBeginFrame for next frame
        self.last_flip_phases = {
            'end_frame_ms':    0.0,
            'swap_buffers_ms': 0.0,
            'wait_begin_ms':   0.0,
        }
        self._swap_buffers_wrapped = False

        # Operator-mirror policy. The mirror window's swapBuffers is locked
        # to the desktop monitor refresh by Windows DWM regardless of vsync
        # settings (~5 ms per call). At 120 Hz panel that's a budget
        # killer; at 72/90 Hz it fits. Default 1 = every flip (classic
        # behavior). Set to a higher N for occasional preview (e.g. 6 ≈
        # 12 Hz mirror at 72 Hz submit). Disabling entirely (0) is NOT
        # recommended — it makes the Oculus runtime flag the app as
        # "behind schedule" and show a persistent corner perf indicator.
        self.mirror_swap_every = 1     # 0 = never; >=1 = every Nth flip
        self._mirror_swap_counter = 0

        # Measured cycle period (seconds), set by validate_frame_rate
        # during setup from blank-flip timing. Diagnostic only — written
        # into the telemetry sidecar so analysis sees the true libovr
        # cycle, independent of what displayRefreshRate advertised.
        self.measured_period_s = None

        # ASW (Asynchronous Spacewarp) tracking. For a pattern-reversal
        # stimulus, ASW frames are SYNTHESIZED by motion-vector extrapolation
        # between the last two submissions — i.e. an invented midway image
        # between phase-A and phase-B checkerboards. Any ASW activation
        # during a trial corrupts the diode-anchored timing and smears the
        # VEP. We can't fully prevent it (Oculus runtime + Debug Tool own
        # the decision), but we can:
        #   (1) hint to the runtime we don't want it via setBool below,
        #   (2) count activations per-frame so analysis can mark affected
        #       trials post-hoc, and
        #   (3) surface session totals in the telemetry sidecar.
        # `_asw_prev_active` tracks edge transitions so we increment the
        # activation counter once per off→on edge, not once per active
        # frame; `asw_active_frame_count` records cumulative time spent
        # with ASW engaged. We surface these to the telemetry sidecar so
        # analysis can flag and exclude sessions where ASW corrupted
        # the stimulus.
        self._asw_prev_active        = False
        self.asw_activation_count    = 0    # off→on edges this session
        self.asw_active_frame_count  = 0    # frames where aswIsActive=1
        self.asw_max_toggle_count    = 0    # max libovr aswActivatedToggleCount
        self.asw_max_presented_count = 0    # max libovr aswPresentedFrameCount
        self.asw_max_failed_count    = 0    # max libovr aswFailedFrameCount
        # Compositor drop count is cumulative in libovr — track the max
        # we've seen each time per-frame stats are sampled. End-of-session
        # value = total compositor-side photon-delivery failures.
        self.comp_dropped_max        = 0
        self.app_dropped_max         = 0

    def _startOfFlip(self):
        # libovr.endFrame() lives inside the base implementation. Time it.
        import time as _t
        t0 = _t.perf_counter()
        try:
            return super()._startOfFlip()
        finally:
            self.last_flip_phases['end_frame_ms'] = (_t.perf_counter() - t0) * 1000.0
            # Wrap backend.swapBuffers on first flip — backend isn't fully
            # initialized until after super().__init__, and the original
            # function reference must be captured before we replace it.
            #
            # The wrapper also implements the mirror-swap policy: skip the
            # DWM-vsync-locked swap on most frames, only do a real swap
            # every mirror_swap_every frames (0 = never). On skipped frames
            # we still pump the window event queue so the OS keeps the
            # mirror window responsive.
            if not self._swap_buffers_wrapped:
                try:
                    backend = self.backend
                    original_swap = backend.swapBuffers
                    winHandle = getattr(backend, 'winHandle', None)
                    def _timed_swap(*a, **kw):
                        ts = _t.perf_counter()
                        try:
                            n = self.mirror_swap_every
                            self._mirror_swap_counter += 1
                            do_swap = n and (self._mirror_swap_counter % n == 0)
                            if do_swap:
                                return original_swap(*a, **kw)
                            # Skip the swap, but pump window events so the
                            # window stays responsive to the OS.
                            if winHandle is not None:
                                try:
                                    winHandle.dispatch_events()
                                except Exception:
                                    pass
                            return None
                        finally:
                            self.last_flip_phases['swap_buffers_ms'] = (
                                _t.perf_counter() - ts) * 1000.0
                    backend.swapBuffers = _timed_swap
                    self._swap_buffers_wrapped = True
                except Exception:
                    pass

    def _waitToBeginHmdFrame(self):
        # Time the libovr wait so flip-phase telemetry records how much of
        # the frame budget the compositor's native gating consumed.
        import time as _t
        t0 = _t.perf_counter()
        try:
            return super()._waitToBeginHmdFrame()
        finally:
            self.last_flip_phases['wait_begin_ms'] = (_t.perf_counter() - t0) * 1000.0

    def validate_frame_rate(self, draw_blank, n_frames=240, warmup=10):
        """Measure actual frame delivery rate and compare to the runtime target.

        Specific to VR because Quest Link's encoded transport pipeline can
        silently degrade (wrong GPU, encode bottleneck) even though the runtime
        still advertises its nominal refresh rate.

        Args:
            draw_blank: callable that draws a frame and flips the window.
                Caller decides eye-buffer logic.
            n_frames: measurement window. 60 frames ≈ 0.5s at 120 Hz.
            warmup: discarded frames so the compositor reaches steady state.

        Returns:
            Dict with ``target_hz``, ``actual_hz``, ``deviation_pct``, ``ok``
            (True iff deviation ≤ ``DISPLAY_DEVIATION_FLAG_PCT``).
        """
        target_hz = float(self.displayRefreshRate)

        for _ in range(warmup):
            draw_blank()

        t0 = core.getTime()
        for _ in range(n_frames):
            draw_blank()
        elapsed = core.getTime() - t0

        actual_hz = n_frames / elapsed
        deviation = abs(actual_hz - target_hz) / target_hz * 100.0

        # Store the *unrounded* period for the absolute pacer to use as
        # its schedule period. Validate runs with pure blank flips, no
        # stimulus draws and no instruction text — that's the cleanest
        # measurement of libovr's natural cycle we can get. Doing the
        # measurement here (in setup, before any heavy rendering) avoids
        # the calibration pollution we saw when measuring inside the
        # experiment loop across instruction screens.
        #
        # We always use the measurement (never abort or fall back),
        # because the Quest 2 can silently downgrade refresh rate
        # (120 → 90 / 72) due to thermals, encode bandwidth, or its own
        # power management. Recording still proceeds at whatever rate
        # the HMD delivers; downstream analysis decides whether the
        # rate is adequate for the experiment. We warn loudly when the
        # measurement deviates significantly so the operator can
        # investigate before the session is wasted.
        measured = elapsed / n_frames
        nominal  = 1.0 / float(self.displayRefreshRate)
        deviation_pct = 100.0 * abs(measured - nominal) / nominal
        self.measured_period_s = measured
        # Capture for sidecar — analysis reads this to know the true
        # refresh rate during recording.
        self.measured_actual_hz = actual_hz
        self.measured_nominal_hz = round(target_hz, 3)
        self.measured_deviation_pct = deviation_pct
        if deviation_pct > 5.0:
            logging.warning(
                "[vr] *** REFRESH RATE MISMATCH *** measured %.2f Hz "
                "(period %.4f ms) vs nominal %.0f Hz (period %.4f ms) "
                "— %.1f%% deviation. Common cause: Quest 2 silently "
                "downgraded refresh rate. Check Oculus app → Devices "
                "→ Graphics → Display refresh rate; restart Link if "
                "needed. Recording will proceed at the measured rate; "
                "analysis can decide if it's adequate.",
                actual_hz, measured * 1000.0,
                target_hz, nominal * 1000.0,
                deviation_pct,
            )
        else:
            # Use print() not logging.info — root logger defaults to
            # WARNING so info() would be silent. Operators want to see
            # this number at startup as the "I trust the measurement"
            # confirmation.
            print(
                "[vr] measured cycle period = {:.4f} ms "
                "(used by absolute pacer; nominal={:.4f} ms, {:+.2f}%)"
                .format(
                    measured * 1000.0,
                    nominal * 1000.0,
                    100.0 * (measured - nominal) / nominal,
                )
            )

        result = {
            'target_hz':     round(target_hz, 1),
            'actual_hz':     round(actual_hz, 1),
            'deviation_pct': round(deviation, 1),
            'ok':            deviation <= DISPLAY_DEVIATION_FLAG_PCT,
        }

        status = 'OK' if result['ok'] else 'DEGRADED — check GPU assignment'
        print(f"[vr] Target: {target_hz:.0f} Hz  Measured: {actual_hz:.1f} Hz  "
              f"Deviation: {deviation:.1f}%  [{status}]")
        return result

    def compute_optical_axis_offsets(self):
        """
        Computes the Normalized Device Coordinates (NDC) horizontal (x) offset 
        needed to center content directly in front of each eye's physical lens.
        
        Because VR headsets use asymmetric lenses (the screen extends further 
        to the outside of the eye than the inside), the mathematical center of 
        the screen (NDC 0,0) does not align with the user's optical axis.
        """
        try:
            import psychxr.drivers.libovr as libovr
            # fov = [UpTan, DownTan, LeftTan, RightTan]
            left_fov, _, _ = libovr.getEyeRenderFov(libovr.EYE_LEFT)
            right_fov, _, _ = libovr.getEyeRenderFov(libovr.EYE_RIGHT)
            left_L, left_R = float(left_fov[2]), float(left_fov[3])
            right_L, right_R = float(right_fov[2]), float(right_fov[3])
            return ((left_L - left_R) / (left_L + left_R),
                    (right_L - right_R) / (right_L + right_R))
        except Exception as e:
            logging.warning(f"[VR] Failed to compute optical axis offsets: {e}")
            return (0.0, 0.0)

    def sync_vr_clock(self):
        """
        Calculates Wall-clock <-> LibOVR clock offset. LibOVR timestamps are on a QPC-based
        clock with arbitrary zero; time.time() is Unix epoch. Sample paired calls in a 
        tight bracket and keep the tightest, so analysis can convert LibOVR times to wall-clock.
        """
        if self.libovr_to_wallclock_offset is not None:
            return self.libovr_to_wallclock_offset

        try:
            from psychxr.drivers.libovr import timeInSeconds
            best_bracket = None
            best_offset = None
            for _ in range(21):
                t0 = time()
                lovr = timeInSeconds()
                t1 = time()
                bracket = t1 - t0
                offset = 0.5 * (t0 + t1) - lovr
                if best_bracket is None or bracket < best_bracket:
                    best_bracket = bracket
                    best_offset = offset
            
            logging.info(
                f"[VR] clock offset (wall - libovr) = "
                f"{best_offset:.6f}s  (tightest bracket = {best_bracket*1e3:.3f}ms)"
            )
            
            self.libovr_to_wallclock_offset = best_offset
            self.libovr_to_wallclock_bracket = best_bracket
            return best_offset
        except Exception as e:
            logging.warning(f"[VR] LibOVR clock sync failed: {e}")
            return None

    def log_display_info(self):
        """
        Reads IPD, PPD, and display resolution from the LibOVR session and logs
        them to the telemetry sidecar as header comment rows.
        Returns (ppd, ipd_mm) for use in stimulus sizing.
        """
        try:
            ppta = self.pixelsPerTanAngleAtCenter
            ppd_h = np.mean([p[0] for p in ppta]) * (np.pi / 180.0)
            ppd_v = np.mean([p[1] for p in ppta]) * (np.pi / 180.0)
            ppd = int(round(min(ppd_h, ppd_v)))

            eye_to_nose = self.eyeToNoseDistance
            ipd_mm = (eye_to_nose[0] + eye_to_nose[1]) * 1000.0

            logging.info(
                f"[VR] IPD={ipd_mm:.1f}mm  ppd={ppd} (h={ppd_h:.1f} v={ppd_v:.1f})  "
                f"res={self.displayResolution}  eye_buf={self.size}"
            )

            self.timing_data.insert(0, ['# ipd_mm', ipd_mm, 'ppd', ppd, f'ppd_h={ppd_h:.1f} ppd_v={ppd_v:.1f}'])

            return ppd, ipd_mm
        except Exception as e:
            logging.warning(f"[VR] Failed to read display info: {e}")
            return None, None

    # Field list shared by log_telemetry and get_recent_perf_stats.
    # These are the LibOVRPerfStatsPerCompositorFrame fields we surface — they
    # split frame time between app-side render, compositor work, and queue/ASW
    # behavior, which is what diagnoses encode/transport stalls vs render cost.
    _PERF_FIELDS = (
        'appFrameIndex',
        'appCpuElapsedTime',
        'appGpuElapsedTime',
        'appMotionToPhotonLatency',
        'appQueueAheadTime',
        'appDroppedFrameCount',
        'compositorFrameIndex',
        'compositorCpuElapsedTime',
        'compositorGpuElapsedTime',
        'compositorCpuStartToGpuEndElapsedTime',
        'compositorGpuEndToVsyncElapsedTime',
        'compositorLatency',
        'compositorDroppedFrameCount',
        'timeToVsync',
        'aswIsActive',
        'aswActivatedToggleCount',
        'aswPresentedFrameCount',
        'aswFailedFrameCount',
        'hmdVsyncIndex',
    )

    def get_recent_perf_stats(self):
        """Snapshot the most-recent LibOVR frame stat as a plain dict.

        Returns None if perf stats aren't ready yet (first few frames) or
        psychxr didn't populate them. Designed to be called once per frame
        right after ``window.flip()`` so the values correspond to the frame
        just submitted.

        Used by per-frame diagnostic logging to split a slow frame into
        app GPU vs compositor GPU vs queue/ASW — the CPU-side timing in
        draw_frame can't distinguish these.
        """
        try:
            perf = getattr(self, '_perfStats', None)
            if perf is None or perf.frameStatsCount == 0:
                return None
            stat = perf.frameStats[0]
            result = {f: getattr(stat, f, None) for f in self._PERF_FIELDS}
        except Exception:
            return None

        # Update ASW counters as a side effect — this method is called
        # once per frame by the experiment's logger, which is the right
        # cadence for edge detection. Doing it here (vs in log_telemetry)
        # also catches activations between trials.
        try:
            is_active = bool(result.get('aswIsActive'))
            if is_active:
                self.asw_active_frame_count += 1
                if not self._asw_prev_active:
                    self.asw_activation_count += 1
                    logging.warning(
                        "[vr] ASW ACTIVATED at frame %s — synthesized "
                        "frames will corrupt pattern-reversal timing. "
                        "Set ASW=Disabled in Oculus Debug Tool.",
                        result.get('appFrameIndex'),
                    )
            self._asw_prev_active = is_active

            tc = result.get('aswActivatedToggleCount')
            if tc is not None and tc > self.asw_max_toggle_count:
                self.asw_max_toggle_count = tc
            pc = result.get('aswPresentedFrameCount')
            if pc is not None and pc > self.asw_max_presented_count:
                self.asw_max_presented_count = pc
            fc = result.get('aswFailedFrameCount')
            if fc is not None and fc > self.asw_max_failed_count:
                self.asw_max_failed_count = fc

            # Compositor + app drop counters are cumulative in libovr;
            # track the max we've seen so the end-of-session value is
            # the total drops across the recording.
            cd = result.get('compositorDroppedFrameCount')
            if cd is not None and cd > self.comp_dropped_max:
                self.comp_dropped_max = int(cd)
            ad = result.get('appDroppedFrameCount')
            if ad is not None and ad > self.app_dropped_max:
                self.app_dropped_max = int(ad)
        except Exception:
            pass

        return result

    def get_session_summary(self):
        """Return a short dict of VR session-level diagnostics for
        end-of-session reporting. Counts are cumulative across the
        recording. Safe to call even when no frames have been
        submitted yet — returns zeros."""
        return {
            'compositor_dropped':   int(self.comp_dropped_max),
            'app_dropped':          int(self.app_dropped_max),
            'asw_activations':      int(self.asw_activation_count),
            'asw_active_frames':    int(self.asw_active_frame_count),
        }

    def save_frame_stats(self, save_fn):
        """Print end-of-session frame timing report and save the
        ``_frame_stats.json`` sidecar.

        Lives here (not in BaseExperiment) because the headline
        numbers are only meaningful with the VR compositor counters
        alongside — PsychoPy's ``nDroppedFrames`` alone uses a
        > 1.5x refresh threshold that over-reports drops whenever
        the panel refresh exceeds the sustainable submit rate, and
        on a flat monitor the cycle stats by themselves aren't
        diagnostically interesting.

        VR sessions get the full picture: cycle mean/std/max +
        compositor drops + ASW counts. The JSON sidecar carries
        ``vr_summary`` so analysis tools can read the ground-truth
        drop counts in one place.
        """
        intervals = self.frameIntervals
        if not intervals:
            return

        intervals_ms = [i * 1000 for i in intervals]
        total       = len(intervals)
        mean_ms     = float(np.mean(intervals_ms))
        std_ms      = float(np.std(intervals_ms))
        max_ms      = float(max(intervals_ms))
        psychopy_dropped = self.nDroppedFrames
        refresh_hz  = float(self.displayRefreshRate)
        duration_s  = sum(intervals)
        summary     = self.get_session_summary()

        print(f"\nFrame timing: {total} frames over {duration_s:.1f}s")
        print(f"  Refresh rate: {refresh_hz:.0f} Hz "
              f"(target {1000.0/refresh_hz:.2f} ms)")
        print(f"  Cycle: mean {mean_ms:.2f} ms  std {std_ms:.2f} ms  "
              f"max {max_ms:.2f} ms")
        print(f"  Compositor drops: {summary['compositor_dropped']}  "
              f"App drops: {summary['app_dropped']}  "
              f"ASW frames: {summary['asw_active_frames']}  "
              f"ASW activations: {summary['asw_activations']}")

        if save_fn is not None:
            stats_path = save_fn.with_name(save_fn.stem + '_frame_stats.json')
            payload = {
                'display_refresh_rate_hz': refresh_hz,
                'total_frames':            total,
                'psychopy_dropped_frames': psychopy_dropped,
                'mean_ms':                 round(mean_ms, 3),
                'std_ms':                  round(std_ms, 3),
                'max_ms':                  round(max_ms, 3),
                'intervals_ms':            [round(i, 3) for i in intervals_ms],
                'vr_summary':              summary,
            }
            with open(stats_path, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"  Saved to {stats_path}")

    def log_telemetry(self, trial_idx, software_time):
        """Extracts native LibOVR performance stats and buffers them in memory."""
        submitted_frame_index = None
        stat_dict = {f: None for f in self._PERF_FIELDS}

        try:
            submitted_frame_index = self._frameIndex - 1
            recent = self.get_recent_perf_stats()
            if recent is not None:
                stat_dict.update(recent)
        except Exception:
            pass

        self.timing_data.append(
            [trial_idx, software_time, submitted_frame_index]
            + [stat_dict[f] for f in self._PERF_FIELDS]
        )

    def save_telemetry(self, save_fn):
        """Saves memory-buffered VR timing telemetry to a CSV sidecar.
        """
        if save_fn is None:
            return

        timing_path = save_fn.with_name(save_fn.stem + '_timing.csv')
        with open(timing_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['trial_idx', 'software_time', 'submitted_frame_index']
                + list(self._PERF_FIELDS)
            )

            # Refresh-rate header rows. Quest 2 can silently downgrade
            # refresh (120 → 90 / 72) without changing displayRefreshRate.
            # Analysis MUST read measured_actual_hz, not the nominal value,
            # to know the true frame rate during this recording.
            measured_hz = getattr(self, 'measured_actual_hz', None)
            nominal_hz  = getattr(self, 'measured_nominal_hz', None)
            deviation   = getattr(self, 'measured_deviation_pct', None)
            measured_period = getattr(self, 'measured_period_s', None)
            if measured_hz is not None:
                writer.writerow([
                    '# nominal_refresh_hz', nominal_hz,
                    'measured_refresh_hz', round(measured_hz, 3),
                    'deviation_pct', round(deviation, 2) if deviation is not None else None,
                    'measured_period_ms',
                    round(measured_period * 1000.0, 4) if measured_period else None,
                ])

            if self.libovr_to_wallclock_offset is not None:
                writer.writerow(['# libovr_to_wallclock_offset_s', self.libovr_to_wallclock_offset, 'bracket_ms', self.libovr_to_wallclock_bracket * 1000])

            # ASW session totals — analysis uses these to decide whether
            # to flag the session. activation_count is our own off→on edge
            # count; the libovr_max_* fields are the runtime's own
            # cumulative counters at session end.
            writer.writerow([
                '# asw_activation_count', self.asw_activation_count,
                'asw_active_frame_count', self.asw_active_frame_count,
                'asw_libovr_max_toggle_count', self.asw_max_toggle_count,
                'asw_libovr_max_presented_count', self.asw_max_presented_count,
                'asw_libovr_max_failed_count', self.asw_max_failed_count,
            ])

            writer.writerows(self.timing_data)
        print(f"  Saved VR timing telemetry to {timing_path}")
        if self.asw_activation_count > 0:
            print(
                f"  [vr] *** ASW ENGAGED {self.asw_activation_count}x "
                f"across {self.asw_active_frame_count} frames — affected "
                f"trials should be reviewed; ASW frames are synthesized "
                f"and may distort the pattern-reversal waveform."
            )
        else:
            print("  [vr] ASW did not engage during this session.")
