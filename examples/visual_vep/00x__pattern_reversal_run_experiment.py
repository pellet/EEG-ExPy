"""
PRVEP Run Experiment
==========================================

Pattern Reversal VEP using PsychoPy on a standard monitor or Meta Quest
headset (via psychxr / Meta-Link). The EEG device and save-file are owned
by this script; the stimulus is driven by PsychoPy.

Block schedule: 4 conditions (left/right eye × large/small check) ×
``reps_per_condition`` reps = 8 blocks, shuffled at startup.

Marker codes:
    1   — reversal, left-eye  block
    2   — reversal, right-eye block

    Block-start codes (bit 0 = eye, bit 1 = size class):
    100 — block_start, left  eye, large check (~60 arcmin / 1 deg)
    101 — block_start, right eye, large check
    102 — block_start, left  eye, small check (~30 arcmin / 0.5 deg)
    103 — block_start, right eye, small check
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
# Ground A2, Ref M1.
ch_names = ["Fz", "Pz", "P7", "P8", "O1", "O2", "Oz", "M2"]

# Subject and session identifiers
subject_id = 0
session_nb = 10

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
print(f"Saving data to: {save_fn}")

###################################################################################################
# Run experiment
# ---------------------
#
# Run the Pattern Reversal VEP. The experiment will present alternating checkerboard
# blocks for each eye (or for both eyes on monitor). Press spacebar/controller trigger
# at each block instruction prompt to begin that block.

pattern_reversal_vep = VisualPatternReversalVEP(
    eeg=eeg_device,
    save_fn=save_fn,
    use_vr=use_vr
)
pattern_reversal_vep.run()
