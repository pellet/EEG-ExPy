import logging
import random
import numpy as np

from psychopy import visual
from typing import Optional, Dict, Any
from eegnb.devices.eeg import EEG
from eegnb.experiments.BlockExperiment import BlockExperiment
from eegnb.analysis.vep_utils import ISCEV_CHECK_DEG_LARGE, ISCEV_CHECK_DEG_SMALL
from stimupy.stimuli.checkerboards import contrast_contrast

# ISCEV PR-VEP standard
ISCEV_FIELD_DEG = 16.0
ISCEV_MEAN_LUM = 0.0

# Block conditions, indexed by label 0-3. Label encoding: bit 0 = eye, bit 1 = size class.
# CONDITIONS[label] is also (size_idx, eye_idx) flattened: size_idx = label >> 1, eye_idx = label & 1.
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
    **{f"rev/{c['eye']}/{c['size_name']}":   1   + i for i, c in enumerate(CONDITIONS)},
    **{f"block/{c['eye']}/{c['size_name']}": 100 + i for i, c in enumerate(CONDITIONS)},
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
        ``reps_per_condition`` blocks. Block-start markers (100–103) are pushed on 
        the first reversal of each block to record the condition sequence.
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

        # Build and shuffle the block schedule.
        block_labels = list(range(n_conditions)) * reps_per_condition
        random.shuffle(block_labels)
        self.block_labels = block_labels
        logging.info("[PRVEP] block schedule: %s", block_labels)

        # Expand into a per-trial parameter array.
        self.parameter = np.array(
            [lbl for lbl in block_labels for _ in range(block_trial_size)]
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
        refresh_rate = int(np.round(
            self.window.displayRefreshRate if self.use_vr
            else self.window.getActualFrameRate()
        ))
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

        def make_checker_stim(intensity_checks, check_deg, pos):
            return visual.ImageStim(
                self.window,
                image=self.make_checker_image(intensity_checks, check_deg, ppd=ppd),
                units='pix', size=stim_size_px, color='white', pos=pos,
            )

        # Fixation cross sized in degrees so it stays sub-check at every
        # check-size variant. Clinical PR-VEP uses ~10–20 arcmin marks; 0.25°
        # = 15 arcmin keeps it well below the 30 arcmin small check and the
        # 60 arcmin large check, so foveal stimulation isn't masked.
        FIX_SIZE_DEG = 0.25
        fix_size_px = max(2, int(round(FIX_SIZE_DEG * ppd)))

        def make_fixation(pos_px):
            return visual.ShapeStim(
                win=self.window, pos=pos_px,
                vertices='cross', size=fix_size_px,
                units='pix',
                fillColor=[1, -1, -1], lineColor=[1, -1, -1],
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

        if self.use_vr:
            self.instruction_stim.text = (
                f"Block {current_block + 1}/{self.n_blocks} — "
                f"{open_eye} eye, {size_name} checks\n\n"
                "Focus on the red dot.\n"
                "Try not to blink while the squares are animating.\n"
                "Press spacebar or trigger when ready."
            )
            self.window.setBuffer(open_eye)
            self.grey_background.draw()
            self.instruction_stim.draw()
            self.stim[open_eye]['fixation'].draw()
            self.window.setBuffer(closed_eye)
            self.black_background.draw()
        else:
            text = (
                f"Block {current_block + 1}/{self.n_blocks}\n\n"
                f"Close your {closed_eye} eye.\n"
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
        # Push block-start marker on the first reversal of each block.
        # This lands in the EEG file before the first reversal marker and
        # encodes the full condition (eye × check-size) for the analysis.
        if idx == 0:
            c = CONDITIONS[self.block_labels[self.current_block_index]]
            self.push_vr_marker(
                EVENTS[f"block/{c['eye']}/{c['size_name']}"],
                self.current_block_index * self.block_trial_size,
            )
        self.draw_frame(idx)
        self.push_marker(idx)

    def draw_frame(self, idx: int):
        trial_idx = self.current_block_index * self.block_trial_size + idx
        c = CONDITIONS[int(self.parameter[trial_idx])]
        eye, size_idx = c['eye'], c['size_idx']
        phase = idx % 2               # alternates 0 / 1 for each reversal

        if self.use_vr:
            closed_eye = 'right' if eye == 'left' else 'left'
            self.window.setBuffer(eye)
            self.grey_background.draw()
            self.stim[eye]['checkerboards'][size_idx][phase].draw()
            self.stim[eye]['fixation'].draw()
            self.window.setBuffer(closed_eye)
            self.black_background.draw()
        else:
            self.grey_background.draw()
            self.stim['monoscopic']['checkerboards'][size_idx][phase].draw()
            self.stim['monoscopic']['fixation'].draw()

        self.window.flip()

    def push_marker(self, idx: int):
        trial_idx = self.current_block_index * self.block_trial_size + idx
        c = CONDITIONS[int(self.parameter[trial_idx])]
        self.push_vr_marker(EVENTS[f"rev/{c['eye']}/{c['size_name']}"], trial_idx)

    def present_soa(self, idx: int):
        # Keep the compositor fed at full frame rate; no marker push.
        self.draw_frame(idx)

    def present_iti(self):
        if self.use_vr:
            for eye in ('left', 'right'):
                self.window.setBuffer(eye)
                self.grey_background.draw()
        else:
            self.grey_background.draw()
        self.window.flip()
