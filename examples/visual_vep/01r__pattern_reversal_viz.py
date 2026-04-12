"""
Pattern Reversal VEP Visualization
==================================

This example demonstrates loading, organizing, and visualizing EP response data
from the visual Pattern Reversal VEP (PR-VEP) experiment.

An animation of a checkerboard reversal is shown (the checkerboard squares'
colours are toggled once each half a second).

The data used is the first subject and first session of the eeg-expy PR-VEP
example dataset, recorded using a g.tec Unicorn EEG headset with electrodes
placed at occipital locations (O1, Iz, O2, PO1, PO2) fitted around a Meta
Quest 3S headset. The session used the Meta Quest 3S linked with a PC to
display the checkerboard reversal animation in VR, alternating monocular
stimulation between left and right eye across blocks.

We first use ``fetch_dataset`` to obtain the data files. If the files are not
already present in the local data directory they will be downloaded from the
cloud.

After loading the data from the occipital channels, we place it in an MNE
``Epochs`` object, and then an ``Evoked`` object to obtain the trial-averaged
response. The final figures show the P100 response ERP waveform, a comparison
between eyes, and the interocular difference wave.

"""

###################################################################################################
# Setup
# -----

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from mne import Epochs, find_events

from eegnb.analysis.utils import load_data
from eegnb.analysis import vep_utils
from eegnb.analysis.vep_utils import plot_vep
from eegnb.datasets import fetch_dataset

# sphinx_gallery_thumbnail_number = 3

###################################################################################################
# Hardware lag definitions
# ------------------------
#
# Known display-pipeline offsets for different setups, subtracted from software
# timestamps so that t=0 corresponds to actual photon delivery.
#

def windows_quest3s_usb_unicorn_lag():
    return 0.0368

###################################################################################################
# Load Data
# ---------
#
# Download the PR-VEP example dataset if it is not already present locally.
#

eegnb_data_path = os.path.join(os.path.expanduser('~/'), '.eegnb', 'data')
prvep_data_path = os.path.join(eegnb_data_path, 'visual-PRVEP', 'eegnb_examples')

if not os.path.isdir(prvep_data_path):
    fetch_dataset(data_dir=eegnb_data_path, experiment='visual-PRVEP', site='eegnb_examples')

raw = load_data(subject=1, session=0,
                experiment='visual-PRVEP', site='eegnb_examples', device_name='unicorn',
                data_dir=eegnb_data_path)

###################################################################################################
# Visualize the power spectrum
# ----------------------------

raw.plot_psd()

###################################################################################################
# Filtering
# ---------
#
# Use FIR rather than IIR to keep linear phase.
#

raw.filter(1, 30, method='fir')
raw.plot_psd(fmin=1, fmax=30)

###################################################################################################
# Epoching
# --------
#
# Epoch around stimulus onsets, separating left- and right-eye trials.
#

events = find_events(raw)
event_id = {'left_eye': 1, 'right_eye': 2}

epochs = Epochs(raw, events=events, event_id=event_id,
                tmin=-0.1, tmax=0.4, baseline=None,
                reject={'eeg': 65e-6}, preload=True,
                verbose=False, picks=[7])

epochs.shift_time(-windows_quest3s_usb_unicorn_lag())
print('sample drop %: ', (1 - len(epochs.events)/len(events)) * 100)

###################################################################################################
# Epoch average
# -------------

evoked = epochs.average()
evoked.plot(spatial_colors=True, show=False)

evoked_potentials_left = epochs['left_eye'].average(picks=['Oz'])
plot_vep(evoked_potentials_left)

evoked_potentials_right = epochs['right_eye'].average(picks=['Oz'])
plot_vep(evoked_potentials_right)

###################################################################################################
# Compare evoked potentials by eye
# ---------------------------------

evoked_left = epochs['left_eye'].average(picks=['Oz'])
evoked_right = epochs['right_eye'].average(picks=['Oz'])

fig, ax = plt.subplots(figsize=(10, 6))
times = evoked_left.times * 1000
left_data = evoked_left.data[0] * 1e6
right_data = evoked_right.data[0] * 1e6

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

print(f"Left eye - Number of epochs: {len(epochs['left_eye'])}")
print(f"Right eye - Number of epochs: {len(epochs['right_eye'])}")

p100_window = (80, 120)
time_mask = (times >= p100_window[0]) & (times <= p100_window[1])

left_p100_idx = np.argmax(left_data[time_mask])
right_p100_idx = np.argmax(right_data[time_mask])

print(f"\nP100 Peak Analysis:")
print(f"Left eye  - Peak at {times[time_mask][left_p100_idx]:.1f}ms, amplitude: {left_data[time_mask][left_p100_idx]:.2f}μV")
print(f"Right eye - Peak at {times[time_mask][right_p100_idx]:.1f}ms, amplitude: {right_data[time_mask][right_p100_idx]:.2f}μV")

###################################################################################################
# Interocular difference wave
# ---------------------------

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
