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
        # settings (verified: ~5 ms per call). It happens INSIDE flip()
        # before our pacer wait, so the pacer can't hide it — at 120 Hz
        # (8.33 ms budget) a 5 ms swap pushes the cycle over the period
        # and drops the loop to ~95 Hz. Default off; set to a high N
        # (e.g. 12 ≈ 10 Hz preview at 120 Hz) if you want occasional
        # operator preview during a session.
        self.mirror_swap_every = 0     # 0 = never; >=1 = every Nth flip
        self._mirror_swap_counter = 0

        # Absolute-time pacing. libovr's waitToBeginFrame is a relative wait
        # — it returns when "safe to start", with a render-budget margin.
        # If the app consistently completes a frame faster than vsync, the surplus accumulates and libovr
        # eventually auto-throttles. Absolute-time pacing anchors each
        # frame's wake-up to libovr.getPredictedDisplayTime(frame_index)
        # minus a small render budget — drift can't accumulate because the
        # target is computed from absolute time, not from "now".
        self.use_absolute_pacing = False    # opt-in for A/B comparison
        self.render_budget_s     = 0.002    # 2 ms; sized for current p99
        # Measured cycle period (seconds). Set by validate_frame_rate
        # during setup, before any heavy rendering or instruction frames.
        # The absolute pacer uses this as its schedule period — measuring
        # against pure blank flips is the cleanest snapshot of libovr's
        # natural cycle. Falls back to 1/displayRefreshRate if not set.
        self.measured_period_s = None
        # Anchor for absolute pacing — set on the first paced frame and
        # held constant thereafter. Targets become anchor + Δframes×period
        # so we never re-query the (apparently queue-aware and unstable)
        # libovr.getPredictedDisplayTime for our wait math.
        #
        # The anchor and the wait math are in time.perf_counter() seconds.
        # On Windows perf_counter is QPC, which is also what time.sleep
        # resolves against; using one clock everywhere prevents any rate-
        # mismatch drift between libovr.timeInSeconds and perf_counter.
        self._pace_anchor_time     = None    # perf_counter seconds
        self._pace_anchor_frame    = None    # libovr _frameIndex when anchored
        self._pace_debug_logged    = False
        # Throttle re-anchor warnings — at 120 Hz steady-state drift, the
        # raw rate would be one per ~120 frames. Log at most every 5 s.
        self._pace_last_warn_time  = 0.0
        # Diagnostic counters populated each pace cycle, read by the
        # experiment's per-frame logger.
        self.last_pace = {
            'paced_wait_ms':    0.0,    # how long we actually waited
            'pace_overshoot_ms': 0.0,   # how far past target we woke
            'libovr_wait_ms':   0.0,    # residual super().waitToBegin time
        }

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

    def _absolute_pace_wait(self):
        """Block until the anchor-derived target time for this frame.

        All clock arithmetic uses ``time.perf_counter()`` — on Windows this
        is QPC, the same clock ``time.sleep`` resolves against, so the
        schedule cannot drift due to clock-rate mismatch with the sleep.

        Anchor-based pacing: record perf_counter() at the first paced
        frame, then compute every later target purely as
        ``anchor + (_frameIndex - anchor_frame) × period``. Drift cannot
        accumulate because the schedule is a fixed arithmetic sequence;
        the compositor's vsync ticks at the same rate, so phase relative
        to vsync stays constant once locked.

        The schedule period comes from ``self.measured_period_s``, set
        by ``validate_frame_rate`` during setup. Validate's blank-flip
        loop is the cleanest possible measurement of libovr's natural
        cycle — no stimulus draws, no instruction text, no other
        contamination. If validate hasn't run (e.g. unit tests), we
        fall back to 1/displayRefreshRate.

        We deliberately do NOT use libovr.getPredictedDisplayTime here:
        on Quest Link it returns the *photon* time through the encoder
        pipeline (~3-4 vsyncs ahead), which is the right value for pose
        prediction and the wrong value for "when should I wake up to
        render the next frame."

        Hybrid sleep+busy-wait: ``time.sleep`` is granular and can over-
        or under-shoot by ~0.5-1 ms even at 1 ms timer resolution, so we
        sleep for all but the last ~1.5 ms of the wait, then busy-spin to
        the exact target.

        Returns (paced_wait_ms, overshoot_ms).
        """
        import time as _t

        now = _t.perf_counter()
        period = (self.measured_period_s
                  if self.measured_period_s is not None
                  else 1.0 / float(self.displayRefreshRate))

        # First paced frame — anchor and return without waiting.
        if self._pace_anchor_time is None:
            self._pace_anchor_time  = now
            self._pace_anchor_frame = self._frameIndex
            if not self._pace_debug_logged:
                nominal_period = 1.0 / float(self.displayRefreshRate)
                logging.info(
                    "[pacer] anchored at frame=%d  period=%.4f ms "
                    "(nominal=%.4f ms, %+.2f%% from nominal)",
                    self._frameIndex,
                    period * 1000.0,
                    nominal_period * 1000.0,
                    100.0 * (period - nominal_period) / nominal_period,
                )
                self._pace_debug_logged = True
            return 0.0, 0.0

        frames_ahead = self._frameIndex - self._pace_anchor_frame
        target_wake  = (self._pace_anchor_time
                        + frames_ahead * period
                        - self.render_budget_s)

        # Safety: if we've drifted far past the schedule (instruction
        # screen stall, OS preemption, anything), re-anchor instead of
        # zero-waiting to "catch up" across many missed frames.
        delta = target_wake - now
        if delta < -period:
            # Throttle warnings to at most one per 5 s — at 120 Hz steady
            # drift, raw rate is one per ~100 frames which floods the log.
            if now - self._pace_last_warn_time > 5.0:
                logging.warning(
                    "[pacer] re-anchoring at frame=%d  drift=%.1f ms "
                    "(scheduled target was that far in the past)",
                    self._frameIndex, -delta * 1000.0,
                )
                self._pace_last_warn_time = now
            self._pace_anchor_time  = now
            self._pace_anchor_frame = self._frameIndex
            return 0.0, 0.0

        t_enter = now
        while True:
            t = _t.perf_counter()
            remaining = target_wake - t
            if remaining <= 0:
                break
            if remaining > 0.0015:
                _t.sleep(remaining - 0.001)
            # else: fall through to busy-spin
        overshoot = (_t.perf_counter() - target_wake) * 1000.0
        return (_t.perf_counter() - t_enter) * 1000.0, overshoot

    def _waitToBeginHmdFrame(self):
        # libovr.waitToBeginFrame() lives inside the base implementation.
        # When absolute pacing is on we run our own wait first, *then* call
        # super(). libovr's wait sees us already past its wake target and
        # returns ~immediately, so the absolute pacer effectively replaces
        # it without breaking any of libovr's other bookkeeping
        # (session status / perf stats / input update).
        import time as _t
        t0 = _t.perf_counter()
        paced_ms = overshoot_ms = 0.0
        try:
            if self.use_absolute_pacing:
                paced_ms, overshoot_ms = self._absolute_pace_wait()
            t_super_in = _t.perf_counter()
            try:
                return super()._waitToBeginHmdFrame()
            finally:
                libovr_wait_ms = (_t.perf_counter() - t_super_in) * 1000.0
                self.last_pace = {
                    'paced_wait_ms':     paced_ms,
                    'pace_overshoot_ms': overshoot_ms,
                    'libovr_wait_ms':    libovr_wait_ms,
                }
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
            return {f: getattr(stat, f, None) for f in self._PERF_FIELDS}
        except Exception:
            return None

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

            writer.writerows(self.timing_data)
        print(f"  Saved VR timing telemetry to {timing_path}")
