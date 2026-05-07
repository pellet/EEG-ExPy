"""
PRVEP Run Experiment
==========================================

Pattern Reversal VEP using PsychoPy on a standard monitor or Meta Quest
headset (via psychxr / Meta-Link). The EEG device and save-file are owned
by this script; the stimulus is driven by PsychoPy.

Block schedule: 4 conditions (left/right eye × large/small check) ×
``reps_per_condition`` reps = 8 blocks, shuffled at startup.

Marker codes:
    1 — block_start, left  eye, large check (~60 arcmin / 1 deg)
    2 — block_start, right eye, large check
    3 — block_start, left  eye, small check (~30 arcmin / 0.5 deg)
    4 — block_start, right eye, small check
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
config = CYTON_CONFIG_GAIN_4X

# Electrode montage type: "cap" or "mark-iv"
montage_type = "mark-iv"

# Ground A2, Ref Fz.
ch_names = ["A1", "A2", "PO7", "PO8", "Oz", "Oz", "O1", "O2"]

# Subject and session identifiers
subject_id = 0
session_nb = 17

###################################################################################################
# Initiate EEG device
# ---------------------
#
# Start EEG device based on configuration above.
eeg_device = EEG(device, serial_port=serial_port, ch_names=ch_names, config=config)
#eeg_device = EEG(device="synthetic")

###################################################################################################
# Build experiment object and detect display settings
# ---------------------
#
# The experiment is constructed before the save path so the Rift session is
# already open and we can read the actual refresh rate from the runtime rather
# than hardcoding it. The save path is then built from the real Hz and set on
# the experiment before run() is called.

pattern_reversal_vep = VisualPatternReversalVEP(
    eeg=eeg_device,
    use_vr=use_vr
)

if use_vr:
    _QUEST_HZ = [72, 90, 120]  # nominal Meta Quest refresh rates
    _raw_hz = pattern_reversal_vep.vr.displayRefreshRate
    refresh_rate = min(_QUEST_HZ, key=lambda h: abs(h - _raw_hz))
    display = f"quest-2_{refresh_rate}Hz"
else:
    refresh_rate = 100   # flat display fallback — update for your monitor
    display = f"acer-34-predator_{refresh_rate}Hz"

site = f"{display}_{montage_type}"
data_dir = getenv("DATA_DIR")
save_fn = generate_save_fn(eeg_device.device_name,
                           experiment="visual-PRVEP",
                           site=site,
                           subject_id=subject_id,
                           session_nb=session_nb,
                           data_dir=data_dir)
print(f"Saving data to: {save_fn}  (detected {refresh_rate} Hz)")
pattern_reversal_vep.save_fn = save_fn

pattern_reversal_vep.run()
