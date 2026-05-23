import logging
import random
import numpy as np

from psychopy import visual
from typing import Dict, Any
from eegnb.devices.eeg import EEG
from eegnb.experiments.BlockExperiment import BlockExperiment
from eegnb.analysis.vep_utils import ISCEV_CHECK_DEG_LARGE, ISCEV_CHECK_DEG_SMALL
from stimupy.stimuli.checkerboards import contrast_contrast


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

    def __init__(self, eeg: EEG | None = None, save_fn=None,
                 block_duration_seconds: int = 50,
                 block_trial_size: int = 100,
                 reps_per_condition: int = 2,
                 use_vr: bool = False,
                 use_fullscr: bool = True,
                 check_deg_large: float = ISCEV_CHECK_DEG_LARGE,
                 check_deg_small: float = ISCEV_CHECK_DEG_SMALL,
                 expected_refresh_rate: int | None = None):
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
            expected_refresh_rate=expected_refresh_rate,
        )

        advance_prompt = "Press spacebar or trigger to continue." if use_vr else "Press spacebar to continue."
        self.instruction_text = (
            f"Welcome to the Pattern Reversal VEP experiment!\n\n"
            f"{n_blocks} blocks of {block_duration_seconds} s each\n"
            f"(left/right eye × large/small checks).\n\n"
            f"{advance_prompt}"
        )

        self.check_deg_large = check_deg_large
        self.check_deg_small = check_deg_small

        self.conditions = [
            {'eye': 'left',  'size_name': 'large', 'size_idx': 0, 'check_deg': check_deg_large},
            {'eye': 'right', 'size_name': 'large', 'size_idx': 0, 'check_deg': check_deg_large},
            {'eye': 'left',  'size_name': 'small', 'size_idx': 1, 'check_deg': check_deg_small},
            {'eye': 'right', 'size_name': 'small', 'size_idx': 1, 'check_deg': check_deg_small},
        ]
        
        self.events = {
            f"rev/{c['eye']}/{c['size_name']}": 1 + i for i, c in enumerate(self.conditions)
        }

        # Build block schedule grouped by eye.
        left_eye_blocks  = [i for i, c in enumerate(self.conditions) if c['eye'] == 'left']  * reps_per_condition
        right_eye_blocks = [i for i, c in enumerate(self.conditions) if c['eye'] == 'right'] * reps_per_condition
        
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

        # Total width/height of the cross in visual degrees. Monitor uses
        # 0.5° and VR uses 1° because of lower effective pixels-per-degree
        # at the lens makes a thinner cross hard to anchor the gaze on.
        FIX_SIZE_DEG = 1.0 if self.use_vr else 0.5
        cross_size_px = int(round(FIX_SIZE_DEG * ppd))

        def make_fixation(pos_px):
            return visual.ShapeStim(
                win=self.window, pos=pos_px,
                vertices='cross',
                size=(cross_size_px, cross_size_px),
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
                        make_checker_stim((1, -1),  self.check_deg_large, (eye_x_pix, 0)),
                        make_checker_stim((-1, 1), self.check_deg_large, (eye_x_pix, 0)),
                    ],
                    [
                        make_checker_stim((1, -1),  self.check_deg_small, (eye_x_pix, 0)),
                        make_checker_stim((-1, 1), self.check_deg_small, (eye_x_pix, 0)),
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
        c = self.conditions[self.block_labels[block_index]]
        return c['eye'], c['size_name']

    def present_block_instructions(self, current_block: int) -> None:
        open_eye, size_name = self.block_eye_and_size(current_block)
        closed_eye = 'right' if open_eye == 'left' else 'left'
        
        is_first_block_for_eye = (current_block == 0) or (self.block_eye_and_size(current_block - 1)[0] != open_eye)

        # The "keep both eyes open" reminder is load-bearing: a closed eye
        # produces involuntary Bell's-phenomenon movement (EOG artifact) and a
        # surge of occipital alpha rhythm (Berger effect) that together swamp
        # the small P100 signal. ISCEV / Halliday's clinical protocols all
        # specify opaque patch + both eyes open. Only shown on the eye switch
        # — once the patch is on, repeating it every block adds noise.
        if is_first_block_for_eye:
            if self.use_vr:
                patch_prompt = (
                    f"*** MOVE PHOTODIODE TO {closed_eye.upper()} LENS NOW ***\n"
                    f"*** COVER {closed_eye} EYE WITH A PATCH ***\n\n"
                )
            else:
                patch_prompt = f"*** COVER {closed_eye.upper()} EYE WITH A PATCH***\n"
        else:
            patch_prompt = ""

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
                    "Keep both eyes open and focus on the red cross.\n"
                    "Try not to blink whilst the squares are animating.\n"
                    "Press spacebar or trigger when ready."
                )

                self.instruction_stim.draw()

                if eye == open_eye:
                    self.stim[eye]['fixation'].draw()
        else:
            text = (
                f"Block {current_block + 1}/{self.n_blocks}\n\n"
                f"{patch_prompt}"
                "Keep both eyes open and focus on the red cross.\n"
                f"Check size: {size_name} ({self.check_deg_large if size_name == 'large' else self.check_deg_small}°)\n\n"
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
        c = self.conditions[int(self.parameter[trial_idx])]
        self.push_marker(self.events[f"rev/{c['eye']}/{c['size_name']}"], trial_idx)

    def draw_frame(self, idx: int):
        trial_idx = self.current_block_index * self.block_trial_size + idx
        c = self.conditions[int(self.parameter[trial_idx])]
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
