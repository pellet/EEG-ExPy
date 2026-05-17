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

import os
import platform
from os import getenv
from dotenv import load_dotenv
load_dotenv()

from eegnb import generate_save_fn
from eegnb.devices import CYTON_CONFIG_GAIN_4X
from eegnb.devices.eeg import EEG
from eegnb.experiments.visual_vep import VisualPatternReversalVEP
from eegnb.utils.display import snap_refresh_rate

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

# Per-cap channel names, in Cyton input order (1..8). Personal hardware,
# not part of the shared library. Add a new entry when you set up a new cap.
MONTAGES = {
    # 3D-printed mark-iv occipital array. Ground A2, Ref Fz.
    "thinkpulse-mark-iv": ["P7", "P8", "PO3", "PO4", "O1", "O2", "POz", "Oz"],
    # Standard 10-20 cap (Tencom 20-ch). Ground A2, Ref Fz.
    "cap":     ["P7", "P8", "P3", "P4", "Pz", "Oz", "O1",  "O2"],
}

# Personal monitor specs — refresh rate is used for the save path and for
# the integer-multiple assertion in load_stimulus().
MONITORS = {
    "acer-34-predator": {"hz": 100},
}

# ---- pick montage and monitor for this session --------------------------
montage_type = "cap"
config = CYTON_CONFIG_GAIN_4X if montage_type == "thinkpulse-mark-iv" else None

monitor_name = "acer-34-predator"
ch_names = MONTAGES[montage_type]

# Subject and session identifiers
subject_id = 0
session_nb = 28

# Diagnostic A/B switch: when True, no EEG device is constructed and the
# experiment runs without eeg.start()/eeg.stop(). Used to isolate whether
# BrainFlow's streaming thread is what drops the frame loop from 72 Hz to
# 36 Hz. Leave False for real recordings.
SKIP_EEG = False

###################################################################################################
# Initiate EEG device
# ---------------------
#
# Start EEG device based on configuration above.
if SKIP_EEG:
    print("[diag] SKIP_EEG=True — running without any EEG device")
    eeg_device = None
else:
    eeg_device = EEG(device, serial_port=serial_port, ch_names=ch_names,
                     config=config,
                     analog_mode=True)  # stream AUX (A5-A7) for photodiode trigger
    # eeg_device = EEG(device="synthetic")

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
    use_vr=use_vr,
    use_fullscr=True
)

if use_vr:
    refresh_rate = snap_refresh_rate(pattern_reversal_vep.vr.displayRefreshRate)
    display = f"quest-2_{refresh_rate}Hz"
else:
    refresh_rate = MONITORS[monitor_name]["hz"]
    display = f"{monitor_name}_{refresh_rate}Hz"

site = f"{display}_{montage_type}"
data_dir = getenv("DATA_DIR")
# When SKIP_EEG is on, eeg_device is None — pick a stable device_name for
# the save path so the same diagnostic session directory is reused.
device_name = eeg_device.device_name if eeg_device is not None else "synthetic"
save_fn = generate_save_fn(device_name,
                           experiment="visual-PRVEP",
                           site=site,
                           subject_id=subject_id,
                           session_nb=session_nb,
                           data_dir=data_dir)
print(f"Saving data to: {save_fn}  (detected {refresh_rate} Hz)")
pattern_reversal_vep.save_fn = save_fn

# Quest 2 set to 72 Hz in the Oculus app. We submit at full rate (1:1)
# so the runtime sees us hitting its expected cadence — that avoids the
# corner "app behind schedule" overlay we hit when submitting at half
# rate on a 120 Hz panel. Our measured natural cycle under real-EEG load
# is ~13.8 ms = one 72 Hz vsync, so it's sustainable. Diode-anchored
# epoching is unaffected by the panel rate.
#
# Pacer is off — libovr's native waitToBeginFrame is doing all the
# gating, and our anchor-based pacer was a no-op (paced_wait=0). Mirror
# swap stays at the class default (every flip, classic behavior); this
# matters — disabling it makes the Oculus runtime flag the app as
# "behind schedule" even when comp_dropped is 0.
if use_vr:
    pattern_reversal_vep.vr.use_absolute_pacing = False

pattern_reversal_vep.run()
