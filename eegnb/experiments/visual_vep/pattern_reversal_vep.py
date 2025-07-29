from time import time
import numpy as np
from pandas import DataFrame

from psychopy import visual
from typing import Optional, Any, List
from eegnb.devices.eeg import EEG
from eegnb.experiments import Experiment
from stimupy.stimuli.checkerboards import contrast_contrast
import logging


class VisualPatternReversalVEP(Experiment.BaseExperiment):

    # 100 reversals per eye(200 in total) should provide good SNR
    def __init__(self, duration=200, eeg: Optional[EEG] = None, save_fn=None,
                 n_trials=400, iti=0, soa=0.5, jitter=0, use_vr=False, use_fullscr=True):

        exp_name = "Visual Pattern Reversal VEP"
        logging.basicConfig(level=logging.INFO)
        super().__init__(exp_name, duration, eeg, save_fn, n_trials, iti, soa, jitter, use_vr, use_fullscr)

    @staticmethod
    def create_monitor_checkerboard(intensity_checks):
        # Standard parameters for monitor-based pattern reversal VEP
        # Using standard 1 degree check size at 30 pixels per degree
        return contrast_contrast(
            visual_size=(16, 16),  # aspect ratio in degrees
            ppd=72,  # pixels per degree
            frequency=(0.5, 0.5),  # spatial frequency of the checkerboard (0.5 cpd = 1 degree check size)
            intensity_checks=intensity_checks,
            target_shape=(1, 1),
            alpha=0,
            tau=0
        )

    @staticmethod
    def create_vr_checkerboard(intensity_checks):
        # Optimized parameters for Oculus/Meta Quest 2 with PC link
        # Quest 2 has approximately 20 pixels per degree and a ~90° FOV
        # Using standard 1 degree check size (0.5 cpd)
        return contrast_contrast(
            visual_size=(20, 20),  # size in degrees - covers a good portion of the FOV
            ppd=20,  # pixels per degree for Quest 2
            frequency=(0.5, 0.5),  # spatial frequency (0.5 cpd = 1 degree check size)
            intensity_checks=intensity_checks,
            target_shape=(1, 1),
            alpha=0,
            tau=0
        )

    def load_stimulus(self):
        # Frame rate, in Hz
        if self.use_vr:
            self.frame_rate = self.window.displayRefreshRate
        elif self.window.getActualFrameRate() is not None:
            self.frame_rate = self.window.getActualFrameRate()
        else:
            logging.error(
                "Unable to obtain display refresh rate. If Pro-motion is enabled on macOS, you should set the display refresh rate to 60hz to reduce jitter.")
            self.frame_rate = 60

        # Setting up the trial and parameter list
        # Show stimulus in left eye for first half of block, right eye for second half
        block_size = 50
        n_repeats = self.n_trials // block_size
        left_eye = 0
        right_eye = 1
        # First half of block (25 trials) = left eye, second half (25 trials) = right eye
        block = [left_eye] * 25 + [right_eye] * 25
        self.parameter = np.array(block * n_repeats)
        self.trials = DataFrame(dict(parameter=self.parameter))

        if self.use_vr:
            # Create VR checkerboard
            create_checkerboard = self.create_vr_checkerboard

        else:
            # Create Monitor checkerboard
            create_checkerboard = self.create_monitor_checkerboard

        if self.use_vr:
            # the window is large over the eye, checkerboard should only cover the central vision
            size = self.window.size / 1.5
        else:
            size = (self.window_size[1], self.window_size[1])

        # the surrounding / periphery needs to be dark
        self.black_background = visual.Rect(self.window,
                                            width=self.window.size[0],
                                            height=self.window.size[1],
                                            fillColor='black')

        def create_checkerboard_stim(intensity_checks):
            return visual.ImageStim(self.window,
                                    image=create_checkerboard(intensity_checks)['img'],
                                    units='pix', size=size, color='white')

        self.stim = [create_checkerboard_stim((1, -1)), create_checkerboard_stim((-1, 1))]
        return self.stim

    def present_stimulus(self, idx: int):
        self.black_background.draw()
        label = self.trials["parameter"].iloc[idx]
        
        # For VR, set which eye should see the stimulus
        if self.use_vr:
            # label 0 = left eye, label 1 = right eye
            eye_buffer = 'left' if label == 0 else 'right'
            self.window.setBuffer(eye_buffer, clear=True)
        
        self.black_background.draw()
        # draw checkerboard
        checkerboard_frame = idx % 2
        image = self.stim[checkerboard_frame]
        image.draw()

        # Push sample with correct eye marker
        if self.eeg:
            timestamp = time()
            if self.eeg.backend == "muselsl":
                marker = [self.markernames[label]]
            else:
                marker = self.markernames[label]
            self.eeg.push_sample(marker=marker, timestamp=timestamp)

        self.window.flip()

    def present_iti(self):
        self.black_background.draw()
        self.window.flip()
