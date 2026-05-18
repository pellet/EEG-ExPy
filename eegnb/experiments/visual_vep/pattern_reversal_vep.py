import logging
import random
import sys
import time
import numpy as np

from psychopy import visual
from typing import Optional, Dict, Any
from eegnb.devices.eeg import EEG
from eegnb.experiments.BlockExperiment import BlockExperiment
from eegnb.analysis.vep_utils import ISCEV_CHECK_DEG_LARGE, ISCEV_CHECK_DEG_SMALL
from stimupy.stimuli.checkerboards import contrast_contrast


# ----------------------------------------------------------------------------
# Diagnostic helpers
# ----------------------------------------------------------------------------

# Dedicated logger for PRVEP diagnostics. Has its own stdout handler so that
# [GPU] and [frame-diag] messages always appear in the console regardless of
# how PsychoPy / stdlib logging happen to be configured. propagate=False
# stops these messages double-printing via the root logger.
_diag = logging.getLogger("prvep_diag")
if not _diag.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _diag.addHandler(_h)
    _diag.setLevel(logging.INFO)
    _diag.propagate = False


# ----------------------------------------------------------------------------
# Windows timer resolution
# ----------------------------------------------------------------------------
# Windows' default scheduler tick is 15.625 ms. Any sleep inside the libovr
# stack (waitToBeginFrame, etc.) rounds up to that tick, which mathematically
# locks a 72 Hz (13.89 ms) render loop to half-rate. Forcing 1 ms here, at
# module import, means the resolution is high BEFORE psychxr/brainflow ever
# touch it. We also poll resolution periodically so we can see if something
# (typically a sleep in another thread) raises it back to 15.6 ms.
def _force_high_res_timer():
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        _diag.info("[timer] called timeBeginPeriod(1)")
    except Exception as e:
        _diag.warning("[timer] timeBeginPeriod failed: %s", e)


def _query_timer_resolution_ms():
    """Return current system timer resolution in ms via NtQueryTimerResolution.
    Resolution is reported in 100-ns units. None on non-Windows or failure."""
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes
        ntdll = ctypes.windll.ntdll
        _min = wintypes.ULONG(); _max = wintypes.ULONG(); _cur = wintypes.ULONG()
        ntdll.NtQueryTimerResolution(
            ctypes.byref(_min), ctypes.byref(_max), ctypes.byref(_cur)
        )
        return _cur.value / 10000.0  # 100-ns → ms
    except Exception:
        return None


# Apply 1 ms resolution at import time so subsequent code runs with high res.
_force_high_res_timer()


def _log_gpu_info() -> None:
    """Log which GPU OpenGL is actually rendering on.

    On laptops with NVIDIA Optimus or AMD switchable graphics, Python can
    silently default to the integrated GPU. PsychoPy + Quest Link both
    require the discrete GPU — using integrated will cause severe frame
    drops. This log makes the choice visible at the top of every session.

    Pyglet (PsychoPy's windowing backend) exposes the OpenGL context info
    via ``pyglet.gl.gl_info`` once a window is open. We import lazily because
    the import only works after the window has been created.
    """
    try:
        from pyglet.gl import gl_info
        vendor   = gl_info.get_vendor()
        renderer = gl_info.get_renderer()
        version  = gl_info.get_version()
        _diag.info("[GPU] vendor=%s", vendor)
        _diag.info("[GPU] renderer=%s", renderer)
        _diag.info("[GPU] gl_version=%s", version)
        merged = (vendor + " " + renderer).lower()
        if "nvidia" not in merged and "amd" not in merged and "radeon" not in merged:
            _diag.warning(
                "[GPU] *** Discrete GPU NOT detected — rendering on '%s'. ***",
                renderer,
            )
            _diag.warning(
                "[GPU]     If this is a laptop with NVIDIA/AMD discrete graphics,"
            )
            _diag.warning(
                "[GPU]     set python.exe to use the discrete GPU in NVIDIA"
            )
            _diag.warning(
                "[GPU]     Control Panel (3D Settings → Manage 3D Settings →"
            )
            _diag.warning(
                "[GPU]     Program Settings → add python.exe → High-performance)."
            )
        else:
            _diag.info("[GPU] OK — discrete GPU in use")
    except Exception as e:
        _diag.warning("[GPU] Could not query OpenGL info: %s", e)

# ISCEV PR-VEP standard
ISCEV_FIELD_DEG = 16.0
ISCEV_MEAN_LUM = 0.0

VR_DIODE_BRIGHT = +1.0   # full white (PsychoPy color space: -1=black, +1=white)
VR_DIODE_DARK   = -1.0   # full black — maximises reversal contrast on the
                         # photodiode. Was 0.0 (mid-grey), which produced
                         # only ~2 µV reversal envelope swing against ~75 µV
                         # backlight strobing pulses, making per-trial diode
                         # alignment unrecoverable. Full black should give
                         # roughly 10x more swing.

# Block conditions: 4 possible combinations of (eye, size)
CONDITIONS = [
    {'eye': 'left',  'size_name': 'large', 'size_idx': 0, 'check_deg': ISCEV_CHECK_DEG_LARGE},
    {'eye': 'right', 'size_name': 'large', 'size_idx': 0, 'check_deg': ISCEV_CHECK_DEG_LARGE},
    {'eye': 'left',  'size_name': 'small', 'size_idx': 1, 'check_deg': ISCEV_CHECK_DEG_SMALL},
    {'eye': 'right', 'size_name': 'small', 'size_idx': 1, 'check_deg': ISCEV_CHECK_DEG_SMALL},
]

# Hierarchical event tags → integer marker codes. The slash-delimited tags let
# MNE epoch by partial match (e.g. event_id key 'rev/left' selects both sizes).
# Kept stable across recordings so analysis can hard-code this dict.
EVENTS = {
    f"rev/{c['eye']}/{c['size_name']}": 1 + i for i, c in enumerate(CONDITIONS)
}


class VisualPatternReversalVEP(BlockExperiment):

    def __init__(self, eeg: Optional[EEG] = None, save_fn=None,
                 block_duration_seconds: int = 50,
                 block_trial_size: int = 100,
                 reps_per_condition: int = 2,
                 use_vr: bool = False,
                 use_fullscr: bool = True):
        """
        Pattern Reversal VEP with two check sizes, counterbalanced across blocks.

        Block schedule: 4 shuffled conditions (left/right eye, large/small check) ×
        ``reps_per_condition`` blocks. Each reversal is marked with a code 1–4
        identifying the condition; the condition is fully recoverable from the
        reversal marker stream alone.
        """
        n_conditions   = 4
        n_blocks       = n_conditions * reps_per_condition
        soa = 0.5

        super().__init__(
            "Visual Pattern Reversal VEP",
            block_duration_seconds, eeg, save_fn,
            block_trial_size, n_blocks,
            iti=0, soa=soa, jitter=0,
            use_vr=use_vr, use_fullscr=use_fullscr, stereoscopic=True,
        )

        self.instruction_text = (
            f"Welcome to the Pattern Reversal VEP experiment!\n\n"
            f"{n_blocks} blocks of {block_duration_seconds} s each.\n"
            f"left/right eye × large/small checks)\n\n"
            f"Press spacebar or trigger to continue."
        )

        # Build block schedule grouped by eye.
        left_eye_blocks  = [i for i, c in enumerate(CONDITIONS) if c['eye'] == 'left']  * reps_per_condition
        right_eye_blocks = [i for i, c in enumerate(CONDITIONS) if c['eye'] == 'right'] * reps_per_condition
        
        random.shuffle(left_eye_blocks)
        random.shuffle(right_eye_blocks)
        
        # Randomize which eye goes first
        if random.random() < 0.5:
            self.block_labels = left_eye_blocks + right_eye_blocks
        else:
            self.block_labels = right_eye_blocks + left_eye_blocks
            
        logging.info("[PRVEP] block schedule: %s", self.block_labels)

        # Expand into a per-trial parameter array.
        self.parameter = np.array(
            [lbl for lbl in self.block_labels for _ in range(block_trial_size)]
        )


    # ------------------------------------------------------------------
    # Stimulus creation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_checker_image(intensity_checks, check_deg, field_deg=ISCEV_FIELD_DEG, ppd=72):
        cpd = 1.0 / (2.0 * check_deg)
        return contrast_contrast(
            visual_size=(field_deg, field_deg),
            ppd=ppd,
            frequency=(cpd, cpd),
            intensity_checks=intensity_checks,
            target_shape=(0, 0),
            alpha=0,
            tau=0,
        )['img']

    def load_stimulus(self) -> Dict[str, Any]:
        # Log which GPU is rendering — first thing, before anything else, so
        # operators see it before stimuli appear. Critical for diagnosing
        # Optimus / switchable-graphics induced frame drops.
        _log_gpu_info()

        # Log Windows timer resolution at stimulus-load time. If it isn't
        # ~1 ms, something between import and now raised it back to 15.6 ms,
        # and any internal sleep in waitToBeginFrame will round up to half
        # the 72 Hz vsync rate.
        tr = _query_timer_resolution_ms()
        if tr is not None:
            _diag.info("[timer] resolution at load_stimulus = %.3f ms", tr)

        refresh_rate = self.refresh_rate

        reversals_per_sec = 1 / self.soa
        assert refresh_rate % reversals_per_sec == 0, (
            f"Frame rate {refresh_rate} Hz must be an integer multiple of "
            f"{reversals_per_sec} Hz reversal rate"
        )

        # Per-frame timing instrumentation. Each entry is a dict with phase
        # breakdowns of draw_frame; flushed to CSV every _log_every_n_frames.
        self._frame_phase_timings: list = []
        self._last_flip_perf: Optional[float] = None
        self._frame_target_ms: float = 1000.0 / refresh_rate
        self._log_every_n_frames: int = 100
        # Captured by the overridden _draw() to attribute the "invisible"
        # wait (setDefaultView → libovr.waitToBeginFrame) that happens
        # *before* draw_frame's own t0 timer starts.
        self._t_draw_enter: Optional[float] = None
        self._t_after_vr_setup: Optional[float] = None
        _diag.info("[frame-diag] target frame budget = %.2f ms (refresh=%.0f Hz)",
                   self._frame_target_ms, refresh_rate)

        if self.use_vr:
            ppd, _ = self.vr.log_display_info()
            logging.info(
                "[PRVEP-HMD] optical_axis_ndc=L%+.3f/R%+.3f",
                self.left_eye_x_pos, self.right_eye_x_pos,
            )
            tex_px = int(round(ISCEV_FIELD_DEG * ppd))
            stim_size_px = (tex_px, tex_px)
        else:
            ppd = 72
            stim_size_px = (self.window_size[1], self.window_size[1])

        self.grey_background = visual.Rect(
            self.window,
            width=self.window.size[0], height=self.window.size[1],
            fillColor=[ISCEV_MEAN_LUM] * 3,
        )
        self.black_background = visual.Rect(
            self.window,
            width=self.window.size[0], height=self.window.size[1],
            fillColor='black',
        )
        
        self._flash_size_px = 100       # flat-monitor corner patch (px)
        self._vr_patch_size_px = 1000   # VR centred patch (px square)

        if self.use_vr:
            patch_size_px = self._vr_patch_size_px
            bright_color = (VR_DIODE_BRIGHT,) * 3
            dark_color   = (VR_DIODE_DARK,)   * 3
            patch_pos = (0, 0)
        else:
            patch_size_px = self._flash_size_px
            bright_color = (1, 1, 1)
            dark_color   = (-1, -1, -1)
            bw, bh = self.window.size[0], self.window.size[1]
            patch_pos = (
                (bw - patch_size_px) / 2,
                -(bh - patch_size_px) / 2,
            )

        self.photodiode_patch_bright = visual.Rect(
            self.window,
            width=patch_size_px, height=patch_size_px,
            units='pix',
            fillColor=bright_color,
            pos=patch_pos,
        )
        self.photodiode_patch_dark = visual.Rect(
            self.window,
            width=patch_size_px, height=patch_size_px,
            units='pix',
            fillColor=dark_color,
            pos=patch_pos,
        )

        def make_checker_stim(intensity_checks, check_deg, pos):
            return visual.ImageStim(
                self.window,
                image=self.make_checker_image(intensity_checks, check_deg, ppd=ppd),
                units='pix', size=stim_size_px, color='white', pos=pos,
            )

        # Fixation cross: explicit '+' polygon so arm length and arm width are
        # independent.  At low VR ppd (~20) the old single-size ShapeStim
        # rendered as a ~5 px blob that looked like a diamond and connected
        # visually with nearby checkerboard corners (scintillating-grid effect).
        # Arm half-length 0.3° keeps the cross inside one check cell at the
        # small-check size (0.5°); arm width 0.12° gives a clearly legible
        # stroke at VR ppd without occluding foveal stimulation.
        FIX_ARM_DEG = 0.30   # half-length from centre to each arm tip
        FIX_W_DEG   = 0.12   # arm width (stroke thickness)
        arm_px = max(6, int(round(FIX_ARM_DEG * ppd)))
        w_px   = max(2, int(round(FIX_W_DEG   * ppd)))

        def _cross_verts(a, w):
            h = w / 2
            return [(-h, a), (h, a), (h, h), (a, h), (a, -h),
                    (h, -h), (h, -a), (-h, -a), (-h, -h), (-a, -h),
                    (-a, h), (-h, h)]

        def make_fixation(pos_px):
            return visual.ShapeStim(
                win=self.window, pos=pos_px,
                vertices=_cross_verts(arm_px, w_px),
                units='pix', closeShape=True,
                fillColor=[1, -1, -1], lineColor=[-1, -1, -1], lineWidth=1,
            )

        self.instruction_stim = visual.TextStim(
            win=self.window,
            text="",
            color=[-1, -1, -1],
            pos=(0, 0),
            height=0.08,
            units='norm',
            wrapWidth=1.8,
        )

        def make_eye_stimuli(eye_x_pix):
            # Two check-size variants per eye: index 0 = large, 1 = small.
            return {
                'checkerboards': [
                    [
                        make_checker_stim((1, -1),  ISCEV_CHECK_DEG_LARGE, (eye_x_pix, 0)),
                        make_checker_stim((-1, 1), ISCEV_CHECK_DEG_LARGE, (eye_x_pix, 0)),
                    ],
                    [
                        make_checker_stim((1, -1),  ISCEV_CHECK_DEG_SMALL, (eye_x_pix, 0)),
                        make_checker_stim((-1, 1), ISCEV_CHECK_DEG_SMALL, (eye_x_pix, 0)),
                    ],
                ],
                'fixation': make_fixation([eye_x_pix, 0]),
            }

        if self.use_vr:
            w = self.window.size[0]
            return {
                'left':  make_eye_stimuli(self.left_eye_x_pos  * (w / 2)),
                'right': make_eye_stimuli(self.right_eye_x_pos * (w / 2)),
            }
        else:
            return {'monoscopic': make_eye_stimuli(0)}

    # ------------------------------------------------------------------
    # Block instructions
    # ------------------------------------------------------------------

    def block_eye_and_size(self, block_index: int):
        c = CONDITIONS[self.block_labels[block_index]]
        return c['eye'], c['size_name']

    def present_block_instructions(self, current_block: int) -> None:
        open_eye, size_name = self.block_eye_and_size(current_block)
        closed_eye = 'right' if open_eye == 'left' else 'left'
        
        # Check if the eye just switched so we can prompt them to move the patch
        is_first_block_for_eye = (current_block == 0) or (self.block_eye_and_size(current_block - 1)[0] != open_eye)
        
        patch_prompt = f"*** MOVE PHOTODIODE TO {closed_eye.upper()} LENS NOW ***\n" if is_first_block_for_eye else ""

        if self.use_vr:
            # Re-assert height each call — VR state changes (calcEyePoses /
            # setBuffer projection) can corrupt the cached norm-unit size on
            # the shared instruction_stim, causing oversized text from block 2
            # onwards.  Setting .height forces PsychoPy to recompute the glyph
            # layout before draw, keeping it consistent across all blocks.
            self.instruction_stim.height = 0.08
            self.instruction_stim.wrapWidth = 1.8
            for eye in ['left', 'right']:
                self.window.setBuffer(eye)

                if eye == closed_eye:
                    self.black_background.draw()
                    self.instruction_stim.color = [1, 1, 1]
                else:
                    self.grey_background.draw()
                    self.instruction_stim.color = [-1, -1, -1]

                self.instruction_stim.text = (
                    f"Block {current_block + 1}/{self.n_blocks} — "
                    f"{open_eye} eye, {size_name} checks\n\n"
                    f"{patch_prompt}"
                    f"Please ensure your {closed_eye} eye is physically covered (e.g. tissue/patch),\n"
                    f"but keep BOTH eyes open underneath to prevent muscle artifacts.\n\n"
                    "Focus on the red dot.\n"
                    "Try not to blink while the squares are animating.\n"
                    "Press spacebar or trigger when ready."
                )

                self.instruction_stim.draw()

                if eye == open_eye:
                    self.stim[eye]['fixation'].draw()
        else:
            text = (
                f"Block {current_block + 1}/{self.n_blocks}\n\n"
                f"{patch_prompt}"
                f"Cover your {closed_eye} eye with a patch (keep both eyes open).\n"
                f"Focus on the red dot with your {open_eye} eye.\n"
                f"Check size: {size_name} ({ISCEV_CHECK_DEG_LARGE if size_name == 'large' else ISCEV_CHECK_DEG_SMALL}°)\n\n"
                "Press spacebar when ready."
            )
            visual.TextStim(win=self.window, text=text, color=[-1, -1, -1]).draw()
            self.stim['monoscopic']['fixation'].draw()
        self.window.flip()

    # ------------------------------------------------------------------
    # Stimulus presentation
    # ------------------------------------------------------------------

    def present_stimulus(self, idx: int):
        self.draw_frame(idx)
        trial_idx = self.current_block_index * self.block_trial_size + idx
        c = CONDITIONS[int(self.parameter[trial_idx])]
        self.push_marker(EVENTS[f"rev/{c['eye']}/{c['size_name']}"], trial_idx)

    def _draw(self, present_stimulus):
        # Override BaseExperiment._draw to time the VR setup phase.
        # setDefaultView() calls libovr.beginFrame, which internally calls
        # waitToBeginFrame — the actual vsync-paced block. Until now that
        # wait was invisible because draw_frame's t0 started *after* it.
        self._t_draw_enter = time.perf_counter()
        if self.use_vr:
            tracking_state = self.window.getTrackingState()
            self.window.calcEyePoses(tracking_state.headPose.thePose)
            self.window.setDefaultView()
        self._t_after_vr_setup = time.perf_counter()
        present_stimulus()

    def draw_frame(self, idx: int):
        # ---- DIAGNOSTIC TIMING: per-phase timestamps ----
        t0 = time.perf_counter()

        trial_idx = self.current_block_index * self.block_trial_size + idx
        c = CONDITIONS[int(self.parameter[trial_idx])]
        eye, size_idx = c['eye'], c['size_idx']
        phase = idx % 2               # alternates 0 / 1 for each reversal

        diode_patch = (self.photodiode_patch_bright if phase == 0
                       else self.photodiode_patch_dark)

        t_setup = time.perf_counter()

        if self.use_vr:
            closed_eye = 'right' if eye == 'left' else 'left'
            self.window.setBuffer(eye)
            t_setbuf1 = time.perf_counter()
            self.grey_background.draw()
            self.stim[eye]['checkerboards'][size_idx][phase].draw()
            self.stim[eye]['fixation'].draw()
            t_draw1 = time.perf_counter()
            self.window.setBuffer(closed_eye)
            t_setbuf2 = time.perf_counter()
            self.black_background.draw()
            diode_patch.draw()   # centred on closed-eye buffer
            t_draw2 = time.perf_counter()
        else:
            t_setbuf1 = t_setup
            self.grey_background.draw()
            self.stim['monoscopic']['checkerboards'][size_idx][phase].draw()
            self.stim['monoscopic']['fixation'].draw()
            diode_patch.draw()
            t_draw1 = t_setbuf2 = t_draw2 = time.perf_counter()

        self.window.flip()
        t_flip = time.perf_counter()

        # ---- Pull LibOVR compositor stats for the frame we just submitted.
        # Lets us split a slow frame into app GPU vs compositor GPU vs
        # queue/ASW — CPU timing above can't see encode/transport cost. ----
        ovr_stats = None
        flip_phases = None
        if self.use_vr:
            try:
                ovr_stats = self.vr.get_recent_perf_stats()
            except Exception:
                ovr_stats = None
            # Per-flip phase split, populated by the overridden _startOfFlip /
            # _waitToBeginHmdFrame and the wrapped backend.swapBuffers.
            flip_phases = dict(self.vr.last_flip_phases)
            pace_info   = dict(getattr(self.vr, 'last_pace', {}) or {})

        # ---- Record per-frame phase breakdown ----
        interval_ms = ((t_flip - self._last_flip_perf) * 1000.0
                       if self._last_flip_perf is not None else 0.0)
        self._last_flip_perf = t_flip

        # vr_setup_ms attributes the previously-invisible block where
        # setDefaultView → waitToBeginFrame waits for the next safe vsync.
        # iter_total_ms covers the whole _draw() iteration (vr setup + draws
        # + flip) so total_ms isn't undercounting any longer.
        t_draw_enter      = self._t_draw_enter
        t_after_vr_setup  = self._t_after_vr_setup
        vr_setup_ms = (
            (t_after_vr_setup - t_draw_enter) * 1000.0
            if t_draw_enter is not None and t_after_vr_setup is not None
            else 0.0
        )
        pre_t0_ms = (
            (t0 - t_after_vr_setup) * 1000.0
            if t_after_vr_setup is not None else 0.0
        )
        iter_total_ms = (
            (t_flip - t_draw_enter) * 1000.0
            if t_draw_enter is not None else (t_flip - t0) * 1000.0
        )
        # gap_ms = wall-clock time BETWEEN this frame's _draw start and the
        # previous frame's flip return. It's the work in the experiment
        # trial loop that happens outside our draw_frame instrumentation:
        # _user_input(), the while-loop bookkeeping, lambda dispatch, and
        # any GIL contention from BrainFlow's serial reader. Should be
        # sub-millisecond on a clean system; a creeping value here is the
        # symptom of a non-rendering bottleneck.
        gap_ms = (
            (t_draw_enter - self._last_flip_perf_for_gap) * 1000.0
            if (t_draw_enter is not None
                and getattr(self, '_last_flip_perf_for_gap', None) is not None)
            else 0.0
        )
        self._last_flip_perf_for_gap = t_flip

        row = {
            'frame'         : len(self._frame_phase_timings),
            'block'         : self.current_block_index,
            'idx_in_block'  : idx,
            'eye'           : eye,
            'phase'         : phase,
            'vr_setup_ms'   : vr_setup_ms,    # NEW: waitToBeginFrame wait
            'pre_t0_ms'     : pre_t0_ms,      # gap between VR setup and draw start
            'setup_ms'      : (t_setup    - t0)        * 1000.0,
            'setbuf1_ms'    : (t_setbuf1  - t_setup)   * 1000.0,
            'draws_eye1_ms' : (t_draw1    - t_setbuf1) * 1000.0,
            'setbuf2_ms'    : (t_setbuf2  - t_draw1)   * 1000.0,
            'draws_eye2_ms' : (t_draw2    - t_setbuf2) * 1000.0,
            'flip_ms'       : (t_flip     - t_draw2)   * 1000.0,
            'total_ms'      : (t_flip     - t0)        * 1000.0,
            'iter_total_ms' : iter_total_ms,   # full _draw iteration
            'gap_ms'        : gap_ms,           # time BETWEEN draws (trial loop overhead)
            'interval_ms'   : interval_ms,
            # Sub-phase breakdown of flip(). NaN-safe defaults when not VR.
            'end_frame_ms'    : flip_phases['end_frame_ms']    if flip_phases else 0.0,
            'swap_buffers_ms' : flip_phases['swap_buffers_ms'] if flip_phases else 0.0,
            'wait_begin_ms'   : flip_phases['wait_begin_ms']   if flip_phases else 0.0,
            # Absolute-pacing diagnostics. Zero when pacing is disabled.
            'paced_wait_ms'    : pace_info.get('paced_wait_ms', 0.0)    if self.use_vr else 0.0,
            'pace_overshoot_ms': pace_info.get('pace_overshoot_ms', 0.0) if self.use_vr else 0.0,
            'libovr_wait_ms'   : pace_info.get('libovr_wait_ms', 0.0)   if self.use_vr else 0.0,
        }
        # Always include the OVR-stat columns (None when unavailable) so the
        # CSV schema stays stable across frames.
        for k in (
            'appCpuElapsedTime', 'appGpuElapsedTime',
            'appMotionToPhotonLatency', 'appQueueAheadTime',
            'appDroppedFrameCount',
            'compositorCpuElapsedTime', 'compositorGpuElapsedTime',
            'compositorGpuEndToVsyncElapsedTime',
            'compositorLatency', 'compositorDroppedFrameCount',
            'timeToVsync',
            'aswIsActive', 'aswPresentedFrameCount', 'aswFailedFrameCount',
            'appFrameIndex', 'compositorFrameIndex', 'hmdVsyncIndex',
        ):
            row[k] = ovr_stats[k] if ovr_stats else None
        self._frame_phase_timings.append(row)

        # Periodic rolling log to the console only. The full per-frame
        # CSV is written once at session end (see _report_frame_stats
        # override below) — no file I/O during the trial loop.
        if len(self._frame_phase_timings) % self._log_every_n_frames == 0:
            self._log_recent_frame_stats()

    def present_soa(self, idx: int):
        self.draw_frame(idx)

    # ------------------------------------------------------------------
    # Frame-timing diagnostic helpers
    # ------------------------------------------------------------------

    def _log_recent_frame_stats(self) -> None:
        """Log mean/p99 timings for the most recent ``_log_every_n_frames``
        frames, broken down by phase. Lets you see WHICH phase is eating
        time — setBuffer (VR projection switches), draws (GPU work), or
        flip (vsync wait + Link encode/transmit).
        """
        if not self._frame_phase_timings:
            return
        recent = self._frame_phase_timings[-self._log_every_n_frames:]
        if not recent:
            return

        def pct(key, p):
            vals = [r[key] for r in recent]
            return float(np.percentile(vals, p))
        def mean(key):
            vals = [r[key] for r in recent]
            return float(np.mean(vals))

        intervals = [r['interval_ms'] for r in recent if r['interval_ms'] > 0]
        # Effective target accounts for VR submit-rate divisor: at
        # divisor=2 the app submits every other vsync, so the right
        # interval target is 2 × (1/refresh_rate). Using the bare
        # nominal target here would mark every successful half-rate
        # frame as "late" (the 99.9%-dropped headline we saw at
        # divisor=2). The diode and OVR comp_dropped are still the
        # ground truth for whether photons landed; this number is just
        # "did the app's submit-loop hit its own deadline."
        divisor = (int(getattr(self.vr, 'submit_rate_divisor', 1))
                   if self.use_vr else 1)
        target = self._frame_target_ms * max(1, divisor)
        n_late = sum(1 for x in intervals if x > 1.5 * target)
        late_pct = 100.0 * n_late / len(intervals) if intervals else 0.0

        _diag.info(
            "[frame-diag] frame=%d  target=%.1fms  "
            "interval mean=%.1f p99=%.1f late=%.0f%%",
            len(self._frame_phase_timings),
            target,
            float(np.mean(intervals)) if intervals else 0.0,
            float(np.percentile(intervals, 99)) if intervals else 0.0,
            late_pct,
        )
        _diag.info(
            "[frame-diag]   iter_total: mean=%.2f p99=%.2f  |  "
            "flip(total): mean=%.2f p99=%.2f  |  "
            "gap(trial-loop): mean=%.2f p99=%.2f",
            mean("iter_total_ms"), pct("iter_total_ms", 99),
            mean("flip_ms"),       pct("flip_ms", 99),
            mean("gap_ms"),        pct("gap_ms", 99),
        )
        if self.use_vr:
            # Split flip into its three libovr/pyglet phases. Whichever
            # mean is closest to flip(total) is the actual bottleneck.
            _diag.info(
                "[flip-split]   endFrame=%.2f  swapBuffers=%.2f  "
                "waitToBeginFrame=%.2f  (means, ms)",
                mean("end_frame_ms"),
                mean("swap_buffers_ms"),
                mean("wait_begin_ms"),
            )
            _diag.info(
                "[flip-split]   endFrame.p99=%.2f  swapBuffers.p99=%.2f  "
                "waitToBeginFrame.p99=%.2f",
                pct("end_frame_ms", 99),
                pct("swap_buffers_ms", 99),
                pct("wait_begin_ms", 99),
            )
            # Absolute-pacing split — only meaningful when enabled.
            if getattr(self.vr, 'use_absolute_pacing', False):
                _diag.info(
                    "[pacer] paced_wait=%.2f  overshoot=%.3f  "
                    "residual_libovr_wait=%.3f  (means, ms)",
                    mean("paced_wait_ms"),
                    mean("pace_overshoot_ms"),
                    mean("libovr_wait_ms"),
                )
                _diag.info(
                    "[pacer] paced_wait p99=%.2f  overshoot p99=%.3f  "
                    "residual p99=%.3f",
                    pct("paced_wait_ms", 99),
                    pct("pace_overshoot_ms", 99),
                    pct("libovr_wait_ms", 99),
                )

        # Re-check timer resolution — if it drifts up (e.g. brainflow's
        # streaming thread or a backend resetting timeEndPeriod) any sleep
        # in waitToBeginFrame rounds up to that tick.
        tr = _query_timer_resolution_ms()
        if tr is not None:
            _diag.info("[timer] current resolution = %.3f ms", tr)
        if self.use_vr:
            _diag.info(
                "[frame-diag]   setBuf1=%.2f draws1=%.2f setBuf2=%.2f draws2=%.2f (means)",
                mean("setbuf1_ms"),
                mean("draws_eye1_ms"),
                mean("setbuf2_ms"),
                mean("draws_eye2_ms"),
            )

            # ---- LibOVR compositor split (the actually-useful diagnostic).
            # Distinguishes three failure modes:
            #   (a) app GPU high  -> render path is too heavy
            #   (b) comp GPU high -> compositor/encode bottleneck (Link bandwidth)
            #   (c) queue-ahead high or m2p latency growing -> pipeline backed up,
            #       app submitting faster than compositor presents
            recent_ovr = [r for r in recent if r.get('appGpuElapsedTime') is not None]
            if recent_ovr:
                def ms(key):
                    return float(np.mean([r[key] for r in recent_ovr])) * 1000.0
                def ms_p99(key):
                    return float(np.percentile([r[key] for r in recent_ovr], 99)) * 1000.0
                # Dropped counters are cumulative — diff first/last in the window.
                first = recent_ovr[0]
                last  = recent_ovr[-1]
                app_drop_delta  = (last.get('appDroppedFrameCount') or 0) - (first.get('appDroppedFrameCount') or 0)
                comp_drop_delta = (last.get('compositorDroppedFrameCount') or 0) - (first.get('compositorDroppedFrameCount') or 0)
                asw_active_pct  = 100.0 * np.mean([bool(r.get('aswIsActive')) for r in recent_ovr])

                _diag.info(
                    "[ovr] app_gpu mean=%.2f p99=%.2f  comp_gpu mean=%.2f p99=%.2f  "
                    "comp_end_to_vsync=%.2f",
                    ms("appGpuElapsedTime"),  ms_p99("appGpuElapsedTime"),
                    ms("compositorGpuElapsedTime"), ms_p99("compositorGpuElapsedTime"),
                    ms("compositorGpuEndToVsyncElapsedTime"),
                )
                _diag.info(
                    "[ovr] m2p_latency mean=%.1fms  queue_ahead=%.2fms  "
                    "time_to_vsync=%.2fms",
                    ms("appMotionToPhotonLatency"),
                    ms("appQueueAheadTime"),
                    ms("timeToVsync"),
                )
                _diag.info(
                    "[ovr] app_dropped(+%d) comp_dropped(+%d)  asw_active=%.0f%%",
                    app_drop_delta, comp_drop_delta, asw_active_pct,
                )

    def present_iti(self):
        if self.use_vr:
            for eye in ('left', 'right'):
                self.window.setBuffer(eye)
                self.grey_background.draw()
        else:
            self.grey_background.draw()
        self.window.flip()
