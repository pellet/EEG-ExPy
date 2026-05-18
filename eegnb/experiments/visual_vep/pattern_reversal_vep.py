import logging
import random
import sys
import numpy as np

from psychopy import visual
from typing import Optional, Dict, Any
from eegnb.devices.eeg import EEG
from eegnb.experiments.BlockExperiment import BlockExperiment
from eegnb.analysis.vep_utils import ISCEV_CHECK_DEG_LARGE, ISCEV_CHECK_DEG_SMALL
from eegnb.utils.realtime import query_timer_resolution_ms
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
        tr = query_timer_resolution_ms()
        if tr is not None:
            _diag.info("[timer] resolution at load_stimulus = %.3f ms", tr)

        refresh_rate = self.refresh_rate

        reversals_per_sec = 1 / self.soa
        assert refresh_rate % reversals_per_sec == 0, (
            f"Frame rate {refresh_rate} Hz must be an integer multiple of "
            f"{reversals_per_sec} Hz reversal rate"
        )

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

    def draw_frame(self, idx: int):
        trial_idx = self.current_block_index * self.block_trial_size + idx
        c = CONDITIONS[int(self.parameter[trial_idx])]
        eye, size_idx = c['eye'], c['size_idx']
        phase = idx % 2               # alternates 0 / 1 for each reversal

        diode_patch = (self.photodiode_patch_bright if phase == 0
                       else self.photodiode_patch_dark)

        if self.use_vr:
            closed_eye = 'right' if eye == 'left' else 'left'
            self.window.setBuffer(eye)
            self.grey_background.draw()
            self.stim[eye]['checkerboards'][size_idx][phase].draw()
            self.stim[eye]['fixation'].draw()
            self.window.setBuffer(closed_eye)
            self.black_background.draw()
            diode_patch.draw()   # centred on closed-eye buffer
        else:
            self.grey_background.draw()
            self.stim['monoscopic']['checkerboards'][size_idx][phase].draw()
            self.stim['monoscopic']['fixation'].draw()
            diode_patch.draw()

        self.window.flip()

    def present_soa(self, idx: int):
        self.draw_frame(idx)

    def present_iti(self):
        if self.use_vr:
            for eye in ('left', 'right'):
                self.window.setBuffer(eye)
                self.grey_background.draw()
        else:
            self.grey_background.draw()
        self.window.flip()
