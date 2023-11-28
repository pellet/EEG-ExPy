
"""  eeg-notebooks/eegnb/experiments/visual_p300/p300.py """

import os
from time import time
from glob import glob
from random import choice
from optparse import OptionParser
import random

import numpy as np
from pandas import DataFrame
from psychopy import visual, core, event
from psychopy.tools.rifttools import LibOVRHapticsBuffer
from psychxr.drivers.libovr import CONTROLLER_TYPE_RTOUCH, CONTROLLER_TYPE_LTOUCH, HAPTICS_BUFFER_SAMPLES_MAX

from eegnb.stimuli import CAT_DOG
from eegnb.experiments import Experiment
from eegnb.devices.eeg import EEG
from typing import Optional

class SomatosensoryP300(Experiment.BaseExperiment):
    
    def __init__(self, duration=120, eeg: Optional[EEG]=None, save_fn=None,
            n_trials = 2010, iti = 0.4, soa = 0.3, jitter = 0.2, use_vr = False):
        
        exp_name = "Somatosensory P300"
        super().__init__(exp_name, duration, eeg, save_fn, n_trials, iti, soa, jitter, use_vr)
        
    def load_stimulus(self):
        
        load_image = lambda fn: visual.ImageStim(win=self.window, image=fn)
        
        self.targets = list(map(load_image, glob(os.path.join(CAT_DOG, "target-*.jpg"))))
        self.nontargets = list(map(load_image, glob(os.path.join(CAT_DOG, "nontarget-*.jpg"))))
        
        return [self.nontargets, self.targets]

    def present_stimulus(self, idx:int, trial):

        label = self.trials["parameter"].iloc[idx]
        # image = choice(self.targets if label == 1 else self.nontargets)
        controller = choice(CONTROLLER_TYPE_RTOUCH if label == 1 else CONTROLLER_TYPE_LTOUCH)
        # image.draw()
        samples = np.linspace(
            1.0, 0.0, num=HAPTICS_BUFFER_SAMPLES_MAX-1, dtype=np.float32)
        hbuff = LibOVRHapticsBuffer(samples)
        self.vibrate(controller, buffer=hbuff)

        # Push sample
        if self.eeg:
            timestamp = time()
            if self.eeg.backend == "muselsl":
                marker = [self.markernames[label]]
            else:
                marker = self.markernames[label]
            self.eeg.push_sample(marker=marker, timestamp=timestamp)

        self.window.flip()