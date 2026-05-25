"""
PRVEP Run Experiment
==========================================

Pattern Reversal VEP using PsychoPy on a standard monitor or Meta Quest
headset (via psychxr / Meta-Link). The EEG device and save-file are owned
by this script; the stimulus is driven by PsychoPy.

Block schedule: 4 conditions (left/right eye × large/small check) ×
``reps_per_condition`` reps = 8 blocks, shuffled at startup.

Marker codes:
    1 — block_start, left  eye, large check (~60 arcmin / 1 deg — ISCEV standard)
    2 — block_start, right eye, large check
    3 — block_start, left  eye, small check (~15 arcmin / 0.25 deg — ISCEV standard)
    4 — block_start, right eye, small check
"""

###################################################################################################
# Setup
# ---------------------
#
# Imports

import logging
import os
import platform
from os import getenv
from dotenv import load_dotenv
load_dotenv()

# Surface EEG-ExPy's diagnostic INFO messages (GPU detection, timer
# resolution, frame-pacing diagnostics) to the console.
logging.basicConfig(level=logging.INFO, format='%(message)s')

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
use_vr = False

# Device: "cyton", "unicorn", "muse2", "synthetic", None, etc.
device = "cyton"

# Serial port: "COM3" for Windows, "/dev/ttyUSB0" for Linux
serial_port = "COM3"

# Per-cap channel names, in Cyton input order (1..8). Personal hardware,
# not part of the shared library. Add a new entry when you set up a new cap.
MONTAGES = {
    # 3D-printed mark-iv occipital array. Ground A2, Ref Fz.
    "thinkpulse-mark-iv": ["P7", "P8", "PO3", "PO4", "O1", "O2", "POz", "Oz"],
    # Standard 10-20 cap (Tencom 20-ch). Ground A2, Ref Fz.
    "cap":     ["P7", "P8", "P3", "P4", "Pz", "Oz", "O1",  "O2"],
}

# Per-rig expected refresh rates. The hardcoded Hz is the rate we *expect*
# the panel to run at and is used for the save-path label. The experiment
# detects the actual rate from the psychopy/VR runtime in setup(); the two
# are compared and any divergence is logged and persisted into the
# display_check sidecar for verification at analysis time.
MONITORS = {
    "acer-34-predator": {"hz": 100},
}
QUEST2_EXPECTED_HZ = 72

# ---- pick montage and monitor for this session --------------------------
montage_type = "cap"
config = CYTON_CONFIG_GAIN_4X if montage_type == "thinkpulse-mark-iv" else None

monitor_name = "acer-34-predator"
ch_names = MONTAGES[montage_type]

# Subject and session identifiers
subject_id = 0
session_nb = 30

###################################################################################################
# Initiate EEG device
# ---------------------
#
eeg_device = EEG(device, serial_port=serial_port, ch_names=ch_names,
                 config=config,
                 analog_mode=True)  # stream AUX (A5-A7) for photodiode trigger

###################################################################################################
# Build experiment object
# ---------------------
#
#
# =============================================================================
# IMPORTANT: VR REFRESH RATE (Meta Horizon Link App)
# =============================================================================
# If using a photodiode with Cyton at the default 250 Hz sample rate, you SHOULD
# set the Quest 2 to 72 Hz in the Oculus PC App.
#
# Why? A 120 Hz strobe creates a 10 Hz "beat frequency" interference pattern
# with the 250 Hz ADC, causing up to ±30 ms of jitter in the photodiode markers.
# At 72 Hz, the phase aligns almost perfectly every frame, dropping the diode
# measurement noise to ~6 ms and allowing for hyper-accurate per-trial jitter
# correction.
# =============================================================================

if use_vr:
    expected_refresh_rate = QUEST2_EXPECTED_HZ
    display = f"quest-2_{expected_refresh_rate}Hz"
else:
    expected_refresh_rate = MONITORS[monitor_name]["hz"]
    display = f"{monitor_name}_{expected_refresh_rate}Hz"

pattern_reversal_vep = VisualPatternReversalVEP(
    eeg=eeg_device,
    use_vr=use_vr,
    use_fullscr=True,
    expected_refresh_rate=expected_refresh_rate,
)

site = f"{display}_{montage_type}"
data_dir = getenv("DATA_DIR")
save_fn = generate_save_fn(eeg_device.device_name,
                           experiment="visual-PRVEP",
                           site=site,
                           subject_id=subject_id,
                           session_nb=session_nb,
                           data_dir=data_dir)
print(f"Saving data to: {save_fn}  (expected {expected_refresh_rate} Hz)")
pattern_reversal_vep.save_fn = save_fn
pattern_reversal_vep.run()
