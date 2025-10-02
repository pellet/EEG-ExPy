"""
PRVEP run experiment
===============================

This example demonstrates the initiation of an EEG stream with eeg-expy,
and how to run the Pattern Reversal VEP (PRVEP) experiment.

"""

from os import path, getenv

###################################################################################################
# Setup
# ---------------------
#
# Imports
from eegnb import generate_save_fn
from eegnb.devices.eeg import EEG
from eegnb.experiments.visual_vep.pattern_reversal_vep import VisualPatternReversalVEP
import platform

###################################################################################################
# Initiate EEG device
# ---------------------
#
# Start EEG device

if platform.system() == "Windows":
    serial_port = "COM3"
    display_type = ""
else:
    serial_port = "/dev/cu.usbserial-DM03H289"
eeg_device = EEG(device="cyton",
                 ch_names=['CFz', 'CPz', 'C1', 'C2', 'O1', 'O2', 'POz', 'Oz'],
                 serial_port=serial_port)
# eeg_device = EEG(device="synthetic")

# Create save file name
data_dir = getenv('DATA_DIR')
data_dir = path.join(path.expanduser("~/"), data_dir, "data")
save_fn = generate_save_fn(eeg_device.device_name,
                           experiment="block_both_eyes_pattern_reversal-mark_iv_headset",
                           site=f"{platform.system()}_{display_type}",
                           subject_id=0,
                           session_nb=1,
                           data_dir=data_dir)
print(save_fn)

###################################################################################################
# Run experiment
# ---------------------
#

pattern_reversal_vep = VisualPatternReversalVEP(eeg=eeg_device, save_fn=save_fn, use_fullscr=True)
pattern_reversal_vep.run()
