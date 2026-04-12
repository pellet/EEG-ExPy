"""
Pattern Reversal VEP Visualization
==================================

This example demonstrates loading, organizing, and visualizing EP response data from the visual P100 experiment. 

An animation of a checkerboard reversal is shown (the checkerboard squares' colours are toggled once each half a second).

The data used is the first subject and first session of one of the eeg-notebooks pattern reversal example datasets.
It was recorded using an OpenBCI Ultracortex EEG headset (Mark IV) with its last five electrodes placed in the headset's
node locations of (PO1, Oz, PO2, P3 and P4).
These headset node locations were used to fit around a Meta Quest 2 headset, which tilted/angled the headset backwards
so that the real locations of the electrodes are closer to the occipital lobe - O1, Iz, O2, PO1 and PO2.
The session consisted of using the Meta Quest 2 linked with a PC to display the checkerboard reversal animation
for thirty seconds of continuous recording.  

We first load the dataset from the specified data directory. 
After loading the data from the occipital channels, we place it in an MNE `Epochs` object, and then an `Evoked` object to obtain
the trial-averaged delay of the response. 

The final figures show the P100 response EP waveform, comparison between eyes, and difference waves.
"""

###################################################################################################
# Setup
# ---------------------

# Some standard pythonic imports
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from os import path, getenv
from dotenv import load_dotenv
load_dotenv()

# MNE functions
from mne import Epochs, find_events

# EEG-Notebooks functions
from eegnb.analysis.utils import load_data
from eegnb.analysis import vep_utils
from eegnb.analysis.vep_utils import plot_vep

###################################################################################################
# Hardware Lag Definitions
# ---------------------
#
# These are the known hardware offsets for different setups, to be subtracted 
# from the software timestamps.
#

def usb_lag():
    return 0.062

def mac_lag():
    return 0.053

def windows_lag():
    return 0.0142

def legion5slim_unicorn_quest3s_usb_lag():
    return 0.036

def quest_3s_usb_and_unicorn_lag():
    return 0.0368

def legion5slim_quest2usb_cyton_lag():
    # As an approximation since it's not in rules but used in notebooks:
    return 0.036

# sphinx_gallery_thumbnail_number = 3

###################################################################################################
# Load Data
# ---------------------
#
# We will use the vtfi pattern reversal dataset.
# The data is expected to be located in the DATA_DIR defined in the .env file.
#

assert getenv('DATA_DIR'), "DATA_DIR environment variable is not set. Set it in the .env file."

data_dir = path.join(path.expanduser("~/"), getenv('DATA_DIR'))
raw = load_data(subject=1, session=0,
                experiment='block_pattern-reversal', site='Windows_quest-3s_120Hz', device_name='unicorn',
                data_dir=data_dir)

###################################################################################################
# Visualize the power spectrum
# ----------------------------

raw.plot_psd()

###################################################################################################
# Filtering
# ----------------------------
#
# Use FIR rather than IIR to keep linear phase
#

raw.filter(1, 30, method='fir')
raw.plot_psd(fmin=1, fmax=30)

###################################################################################################
# Epoching
# ----------------------------
#
# Create an array containing the timestamps and which eye was presented the stimulus,
# then epoch around those events.
#

events = find_events(raw)
event_id = {'left_eye': 1, 'right_eye': 2}

epochs = Epochs(raw, events=events, event_id=event_id,
                tmin=-0.1, tmax=0.4, baseline=None,
                reject={'eeg': 65e-6}, preload=True,
                verbose=False, picks=[7])

# Shift time by the known hardware lag
epochs.shift_time(-windows_lag())
print('sample drop %: ', (1 - len(epochs.events)/len(events)) * 100)

###################################################################################################
# Epoch average
# ----------------------------

evoked = epochs.average()
evoked.plot(spatial_colors=True, show=False)

evoked_potentials_left = epochs['left_eye'].average(picks=['Oz'])
plot_vep(evoked_potentials_left)

evoked_potentials_right = epochs['right_eye'].average(picks=['Oz'])
plot_vep(evoked_potentials_right)

###################################################################################################
# Compare evoked potentials by event type
# ----------------------------

evoked_left = epochs['left_eye'].average(picks=['Oz'])
evoked_right = epochs['right_eye'].average(picks=['Oz'])

fig, ax = plt.subplots(figsize=(10, 6))

times = evoked_left.times * 1000  # Convert to milliseconds
left_data = evoked_left.data[0] * 1e6  # Convert to microvolts
right_data = evoked_right.data[0] * 1e6  # Convert to microvolts

ax.plot(times, left_data, label='Left Eye', color='blue', linewidth=2)
ax.plot(times, right_data, label='Right Eye', color='red', linewidth=2)

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (μV)')
ax.set_title('Comparison of Evoked Potentials: Left Eye vs Right Eye')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Stimulus Onset')

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"Left eye - Number of epochs: {len(epochs['left_eye'])}")
print(f"Right eye - Number of epochs: {len(epochs['right_eye'])}")

# Find P100 peak for each condition (typically around 100ms)
p100_window = (80, 120)  # milliseconds
time_mask = (times >= p100_window[0]) & (times <= p100_window[1])

left_p100_idx = np.argmax(left_data[time_mask])
right_p100_idx = np.argmax(right_data[time_mask])

left_p100_time = times[time_mask][left_p100_idx]
left_p100_amp = left_data[time_mask][left_p100_idx]

right_p100_time = times[time_mask][right_p100_idx]
right_p100_amp = right_data[time_mask][right_p100_idx]

print(f"\nP100 Peak Analysis:")
print(f"Left eye - Peak at {left_p100_time:.1f}ms, amplitude: {left_p100_amp:.2f}μV")
print(f"Right eye - Peak at {right_p100_time:.1f}ms, amplitude: {right_p100_amp:.2f}μV")

###################################################################################################
# Create difference wave
# ----------------------------

difference_data = left_data - right_data

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, difference_data, label='Left - Right', color='green', linewidth=2)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude Difference (μV)')
ax.set_title('Difference Wave: Left Eye - Right Eye')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Stimulus Onset')
ax.legend()

plt.tight_layout()
plt.show()
