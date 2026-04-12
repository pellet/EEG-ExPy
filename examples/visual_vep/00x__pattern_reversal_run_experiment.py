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
from os import path, getenv
from dotenv import load_dotenv
load_dotenv()

from eegnb import generate_save_fn
from eegnb.devices import CYTON_CONFIG_GAIN_12X
from eegnb.devices.eeg import EEG
from eegnb.experiments.visual_vep import VisualPatternReversalVEP

###################################################################################################
# Configuration
# ---------------------
#
# Set your experiment parameters here before running.
#

# Set debug=True to use a synthetic EEG device (no hardware needed)
debug = False

# Display: set use_vr=True for Meta Quest, False for monitor
use_vr = False
use_fullscr = True

# Device: "cyton", "unicorn", "muse2", etc.
device_name = "cyton"
serial_num = "UN-2022.04.23"

# Electrode montage type: "cap", "individual", "mark-iv", or "think-pulse"
montage_type = "think-pulse"

# Subject and session identifiers
subject_id = 1
session_nb = 0

###################################################################################################
# Initiate EEG device
# ---------------------
#
# Start EEG device based on configuration above.

serial_port = None
if device_name == "cyton":
    serial_port = "COM3" if platform.system() == "Windows" else "/dev/cu.usbserial-DM03H289"

if debug:
    eeg_device = EEG(device="synthetic")
elif device_name == "cyton":
    if montage_type in ("cap", "individual"):
        ch_names = ["Fz", "Cz", "Fp1", "Fp2", "O1", "O2", "Pz", "Oz"]
    elif montage_type in ("mark-iv", "think-pulse"):
        ch_names = ["Fp1", "Fp2", "C1", "C2", "O1", "O2", "POz", "Oz"]
    else:
        ch_names = None
    eeg_device = EEG(device=device_name, serial_port=serial_port,
                     serial_num=serial_num, ch_names=ch_names,
                     config=CYTON_CONFIG_GAIN_12X)
else:
    eeg_device = EEG(device=device_name, serial_port=serial_port, serial_num=serial_num)

###################################################################################################
# Display and save path setup
# ---------------------

if use_vr:
    refresh_rate = 120
    display = "quest-2_{}Hz".format(refresh_rate)
else:
    refresh_rate = 100
    display = "acer-34-predator_{}Hz".format(refresh_rate)

metadata = montage_type
site = "{}_{}".format(platform.system(), display)

data_dir = path.join(path.expanduser("~/"), getenv("DATA_DIR"), "vtfi/data")
save_fn = generate_save_fn(eeg_device.device_name,
                           experiment="block_pattern-reversal",
                           site="{}_{}".format(site, metadata),
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
    use_fullscr=use_fullscr,
    use_vr=use_vr
)
pattern_reversal_vep.run()
