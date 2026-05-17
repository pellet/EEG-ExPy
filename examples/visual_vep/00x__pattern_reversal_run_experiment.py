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

# Absolute-time frame pacing.
# The schedule period comes from VR.measured_period_s, which is set by
# validate_frame_rate during setup (clean blank-flip measurement, no
# stimulus/instruction contamination). Eliminates the small but
# accumulating drift between nominal 1/120 = 8.333 ms and the HMD's
# actual cycle (~8.36 ms over Quest Link) that otherwise feeds libovr's
# queue-throttle and produces bimodal/half-rate patches.
if use_vr:
    # A/B test: pacer disabled. The pacer was a no-op in recent sessions
    # (paced_wait=0.00, libovr's native waitToBeginFrame was doing all the
    # gating). Disabling rules it out as a contributor to the corner
    # loading-indicator that appears on the stim eye during real-EEG runs.
    pattern_reversal_vep.vr.use_absolute_pacing = False
    pattern_reversal_vep.vr.render_budget_s     = 0.002
    # A/B test: re-enable the mirror swap every flip (classic behavior
    # before the "reduce dropped frames" work began). The participant's
    # report — "spinner appeared after switching from eye 1 to eye 2 and
    # stayed on eye 2" — is consistent with libovr eye-buffer swap-chain
    # state going stale when one eye's buffer wasn't being drawn while
    # mirror swaps were disabled. Re-enabling the mirror swap restores
    # the original render pipeline behavior. Costs ~5 ms per flip on DWM
    # but at 72 Hz (13.9 ms budget) there's just enough room.
    pattern_reversal_vep.vr.mirror_swap_every   = 1
    # Quest 2 set to 72 Hz in Oculus app. Submit every vsync (1:1) so the
    # runtime sees us hitting its expected frame rate — that eliminates
    # the corner "app behind schedule" overlay we got at divisor=2 on a
    # 120 Hz panel (the runtime counts every other vsync as app_dropped
    # when we submit at half-rate). Our measured natural cycle is ~13.8 ms
    # under real-EEG load, which is exactly one 72 Hz vsync, so this is
    # sustainable. Diode-anchored epoching is unaffected by the rate.
    pattern_reversal_vep.vr.submit_rate_divisor = 1
    # A/B isolation flag. False = revert ASW-disable hint and in-window
    # mirror message (both added during the "reduce dropped frames"
    # work). Use this to test whether the corner perf-indicator is
    # caused by one of those runtime-level tweaks vs something else.
    # Set True to re-apply the tweaks; the per-frame ASW counters run
    # regardless because they're pure telemetry.
    pattern_reversal_vep.vr.apply_runtime_tweaks = False

pattern_reversal_vep.run()
