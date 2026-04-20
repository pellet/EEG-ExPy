"""
PRVEP Run Experiment
===============================

This example demonstrates the initiation of an EEG stream with eeg-expy,
and how to run the Pattern Reversal VEP (PRVEP) experiment.

The experiment presents a checkerboard that reverses its black and white squares
at 2 reversals per second, while the participant fixates a central dot.
Each reversal elicits a P100 response at occipital electrodes.

The experiment supports both standard monitor presentation and Meta Quest VR
presentation via ``use_vr=True``. VR mode is preferred as it provides monocular
stimulation per eye without manual eye closure, and uses compositor-predicted
photon timestamps for improved timing accuracy.

"""

###################################################################################################
# Setup
# ---------------------
#
# Imports

import platform
from os import getenv
from dotenv import load_dotenv
load_dotenv()

from eegnb import generate_save_fn
from eegnb.devices import CYTON_CONFIG_GAIN_4X
from eegnb.devices.eeg import EEG
from eegnb.experiments.visual_vep import VisualPatternReversalVEP

###################################################################################################
# Configuration
# ---------------------
#
# Set your experiment parameters here before running.
#

# Display: set use_vr=True for Meta Quest, False for monitor
use_vr = True

# Device: "cyton", "unicorn", "muse2", etc.
device = "cyton"

# Serial port: "COM3" for Windows, "/dev/ttyUSB0" for Linux
serial_port = "COM3"

# Config: CYTON_CONFIG_GAIN_4X needed for Thinkpulse active electrodes, otherwise leave as None.
config = None

# Electrode montage type: "cap" or "mark-iv"
montage_type = "cap"
# Ground M1, Ref Fz.
ch_names = ["Fp1", "Fp2", "T5", "T6", "O1", "O2", "Oz", "Pz"]

# Subject and session identifiers
subject_id = 0
session_nb = 7

###################################################################################################
# Initiate EEG device
# ---------------------
#
# Start EEG device based on configuration above.
eeg_device = EEG(device, serial_port=serial_port, ch_names=ch_names, config=config)
#eeg_device = EEG(device="synthetic")

###################################################################################################
# Display and save path setup
# ---------------------

if use_vr:
    refresh_rate = 120
    display = "quest-2_{}Hz".format(refresh_rate)
else:
    refresh_rate = 100
    display = "acer-34-predator_{}Hz".format(refresh_rate)

site="{}_{}".format(display, montage_type)
data_dir = getenv("DATA_DIR")
save_fn = generate_save_fn(eeg_device.device_name,
                           experiment="visual-PRVEP",
                           site=site,
                           subject_id=subject_id,
                           session_nb=session_nb,
                           data_dir=data_dir)
print(save_fn)

###################################################################################################
# Run experiment
# ---------------------
#
# Run the Pattern Reversal VEP. The experiment will present alternating checkerboard
# blocks for each eye (or for both eyes on monitor). Press spacebar/controller trigger
# at each block instruction prompt to begin that block.

pattern_reversal_vep = VisualPatternReversalVEP(
    display_refresh_rate=refresh_rate,
    eeg=eeg_device,
    save_fn=save_fn,
    use_vr=use_vr
)
pattern_reversal_vep.run()
