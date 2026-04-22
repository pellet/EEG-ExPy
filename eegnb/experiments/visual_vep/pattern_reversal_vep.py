from time import time
import csv
import logging
import numpy as np

from psychopy import visual
from typing import Optional, Dict, Any
from eegnb.devices.eeg import EEG
from eegnb.experiments.BlockExperiment import BlockExperiment
from stimupy.stimuli.checkerboards import contrast_contrast

# ISCEV PR-VEP standard
ISCEV_CHECK_DEG = 1.0
ISCEV_FIELD_DEG = 16.0
ISCEV_MEAN_LUM = 0.0

QUEST2_PPD_NOMINAL = 20

class VisualPatternReversalVEP(BlockExperiment):

    def __init__(self, display_refresh_rate: int, eeg: Optional[EEG] = None, save_fn=None,
                 block_duration_seconds=50, block_trial_size: int=100, n_blocks: int=8, use_vr=False, use_fullscr=True):

        self.display_refresh_rate = display_refresh_rate
        soa=0.5
        iti=0
        jitter=0

        super().__init__("Visual Pattern Reversal VEP", block_duration_seconds, eeg, save_fn, block_trial_size, n_blocks, iti, soa, jitter, use_vr, use_fullscr, stereoscopic=True)

        self.instruction_text = f"""Welcome to the Visual Pattern Reversal VEP experiment!
        
        This experiment will run for {n_blocks} blocks of {block_duration_seconds} seconds each.
        
        Press spacebar or controller to continue.
        """

        # Setting up the trial and parameter list
        left_eye = 0
        right_eye = 1
        # Alternate between left and right eye blocks
        block_eyes = []
        for block_num in range(n_blocks):
            eye = left_eye if block_num % 2 == 0 else right_eye
            block_eyes.extend([eye] * block_trial_size)
        self.parameter = np.array(block_eyes)

    @staticmethod
    def create_checkerboard(intensity_checks, field_deg=ISCEV_FIELD_DEG,
                            check_deg=ISCEV_CHECK_DEG, ppd=72):
        cpd = 1.0 / (2.0 * check_deg)
        return contrast_contrast(
            visual_size=(field_deg, field_deg),
            ppd=ppd,
            frequency=(cpd, cpd),
            intensity_checks=intensity_checks,
            target_shape=(0, 0),
            alpha=0,
            tau=0,
        )

    def load_stimulus(self) -> Dict[str, Any]:
        # Frame rate, in Hz
        # TODO: Fix - Rift.GetActualFrameRate() crashes in psychxr due to 'EndFrame called before BeginFrame'
        actual_frame_rate = np.round(self.window.displayRefreshRate if self.use_vr else self.window.getActualFrameRate())

        # Ensure the expected frame rate matches and is divisable by the stimulus rate(soa)
        assert actual_frame_rate % self.soa == 0, f"Expected frame rate divisable by stimulus rate: {self.soa}, but got {actual_frame_rate} Hz"
        assert abs(self.display_refresh_rate - actual_frame_rate) <= self.display_refresh_rate * 0.05, f"Expected frame rate {self.display_refresh_rate} Hz, but got {actual_frame_rate} Hz"

        if self.use_vr:
            ppd, ipd_mm = self.rift.log_display_info()
            logging.info(f"[PRVEP-HMD] optical_axis_ndc=L{self.left_eye_x_pos:+.3f}/R{self.right_eye_x_pos:+.3f}")

            # 1 texel = 1 buffer pixel
            tex_px = int(round(ISCEV_FIELD_DEG * ppd))
            stim_size_px = (tex_px, tex_px)
        else:
            ppd = 72
            stim_size_px = (self.window_size[1], self.window_size[1])

        if not self.use_vr:
            patch_size = 50
            x = -self.window.size[0] / 2 + patch_size / 2
            y = -self.window.size[1] / 2 + patch_size / 2
            self.optode_patch = visual.Rect(
                self.window, width=patch_size, height=patch_size,
                pos=(x, y), units='pix', fillColor='white'
            )
        else:
            self.optode_patch = None

        self.black_background = visual.Rect(self.window,
                                            width=self.window.size[0],
                                            height=self.window.size[1],
                                            fillColor='black')

        # Match checkerboard mean luminance to avoid adaptation shift.
        self.grey_background = visual.Rect(self.window,
                                            width=self.window.size[0],
                                            height=self.window.size[1],
                                            fillColor=[ISCEV_MEAN_LUM] * 3)

        def create_checkerboard_stim(intensity_checks, pos):
            return visual.ImageStim(
                self.window,
                image=self.create_checkerboard(
                    intensity_checks,
                    field_deg=ISCEV_FIELD_DEG,
                    check_deg=ISCEV_CHECK_DEG,
                    ppd=ppd,
                )['img'],
                units='pix', size=stim_size_px, color='white', pos=pos,
            )

        # Create fixation stimuli
        def create_fixation_stim(pos):
            size = 0.02 if self.use_vr else 0.4
            return visual.Rect(
                win=self.window,
                pos=pos,
                width=size,
                height=size,
                units='norm' if self.use_vr else None,
                fillColor=[1, -1, -1],
                lineColor=[1, -1, -1],
            )

        def create_vr_block_instruction(pos):
            return visual.TextStim(
                win=self.window,
                text="Focus on the red dot, and try not to blink whilst the "
                     "squares are flashing, press the spacebar or pull the "
                     "controller trigger when ready to commence.",
                color=[-1, -1, -1],
                pos=pos, height=0.1,
            )

        # All stimuli placed on each lens's optical axis for symmetric FOV
        # and maximum lens resolution.
        def create_eye_stimuli(eye_x_pos, pix_x_pos):
            return {
                'checkerboards': [
                    create_checkerboard_stim((1, -1), pos=(pix_x_pos, 0)),
                    create_checkerboard_stim((-1, 1), pos=(pix_x_pos, 0))
                ],
                'fixation': create_fixation_stim([eye_x_pos, 0]),
                'vr_block_instructions': create_vr_block_instruction((eye_x_pos, 0))
            }

        if self.use_vr:
            # pix_x = ndc_x * (per-eye buffer width / 2)
            window_width = self.window.size[0]
            left_pix_x_pos = self.left_eye_x_pos * (window_width / 2)
            right_pix_x_pos = self.right_eye_x_pos * (window_width / 2)
            return {
                'left': create_eye_stimuli(self.left_eye_x_pos, left_pix_x_pos),
                'right': create_eye_stimuli(self.right_eye_x_pos, right_pix_x_pos)
            }
        else:
            return {
                'monoscopic': create_eye_stimuli(0, 0)
            }

    def _present_vr_block_instructions(self, open_eye, closed_eye):
        self.window.setBuffer(open_eye)
        self.grey_background.draw()
        self.stim[open_eye]['vr_block_instructions'].draw()
        self.stim[open_eye]['fixation'].draw()
        self.window.setBuffer(closed_eye)
        self.black_background.draw()

    def present_block_instructions(self, current_block: int) -> None:
        if self.use_vr:
            if current_block % 2 == 0:
                self._present_vr_block_instructions(open_eye="left", closed_eye="right")
            else:
                self._present_vr_block_instructions(open_eye="right", closed_eye="left")
        else:
            if current_block % 2 == 0:
                instruction_text = (
                    "Close your right eye, then focus on the red dot with your left eye. "
                    "Press spacebar or controller when ready."
                )
            else:
                instruction_text = (
                    "Close your left eye, then focus on the red dot with your right eye. "
                    "Press spacebar or controller when ready."
                )
            text = visual.TextStim(win=self.window, text=instruction_text, color=[-1, -1, -1])
            text.draw()
            self.stim['monoscopic']['fixation'].draw()
        self.window.flip()

    def present_stimulus(self, idx: int):
        self._draw_frame(idx)
        self._push_marker(idx)

    def _draw_frame(self, idx: int):
        trial_idx = self.current_block_index * self.block_trial_size + idx
        label = self.parameter[trial_idx]

        open_eye = 'left' if label == 0 else 'right'
        closed_eye = 'left' if label == 1 else 'right'

        # draw checkerboard and fixation
        if self.use_vr:
            self.window.setBuffer(open_eye)
            self.grey_background.draw()
            display = self.stim['left' if label == 0 else 'right']
        else:
            self.black_background.draw()
            display = self.stim['monoscopic']

        checkerboard_frame = idx % 2
        display['checkerboards'][checkerboard_frame].draw()
        display['fixation'].draw()

        if self.use_vr:
            self.window.setBuffer(closed_eye)
            self.black_background.draw()

        # Alternate sync patch polarity with each reversal so the photodiode
        # fires on every checkerboard flip, not just odd or even frames.
        if self.optode_patch is not None:
            self.optode_patch.fillColor = 'white' if checkerboard_frame == 0 else 'black'
            self.optode_patch.draw()

        self.window.flip()

    def _push_marker(self, idx: int):
        trial_idx = self.current_block_index * self.block_trial_size + idx
        label = self.parameter[trial_idx]
        marker = self.markernames[label]
        
        self.push_vr_marker(marker, trial_idx)

    def present_soa(self, idx: int):
        # Redraw the current checkerboard each frame during the SOA wait so the
        # VR compositor stays fed (~120 Hz). No marker push / timing row.
        self._draw_frame(idx)


    def present_iti(self):
        if self.use_vr:
            for eye in ['left', 'right']:
                self.window.setBuffer(eye)
                self.black_background.draw()
        self.window.flip()
