"""
Longitudinal P100 Tracking
===========================

This example demonstrates how to load multiple PR-VEP recording sessions
for the same subject and track P100 latency over time.  This is useful for
monitoring changes in visual pathway conduction — for instance during nerve
recovery, remyelination, or neuroplasticity studies — where latency shifts
of a few milliseconds between sessions are meaningful.

The workflow is:

1. Discover all sessions for a given subject.
2. For each session, epoch around stimulus onsets, extract the per-eye P100
   latency using parabolic interpolation (sub-sample precision), and store
   the results.
3. Plot per-eye P100 latency and interocular difference over sessions.

Before attributing a latency change to an intervention, record several
baseline sessions (at least 3–5, ideally over 1–2 weeks) to establish your
individual test-retest range.

"""

###################################################################################################
# Setup
# -----

import os
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from mne import Epochs, find_events

from eegnb.analysis.utils import load_csv_as_raw
from eegnb.analysis.vep_utils import get_peak
from eegnb.devices.utils import EEG_INDICES, SAMPLE_FREQS

###################################################################################################
# Configuration
# -------------
#
# Point ``data_root`` at the directory tree that contains your recordings.
# The expected layout follows the eeg-expy convention::
#
#     data_root/subject{XXXX}/session{XXX}/recording_*.csv
#
# Adjust ``device_name``, ``subject_id``, and ``hardware_lag`` for your setup.
#

data_root = os.path.join(os.path.expanduser('~/'), 'eeg-data', 'visual-PRVEP')
device_name = 'cyton'
subject_id = 1
hardware_lag = 0.0  # seconds — set to your measured display-pipeline offset

###################################################################################################
# Discover sessions
# -----------------
#
# Scan the subject directory for session folders and sort them by the
# recording timestamp embedded in the CSV filename.
#

subject_dir = os.path.join(data_root, f'subject{subject_id:04d}')
session_dirs = sorted(glob.glob(os.path.join(subject_dir, 'session*')))

print(f'Found {len(session_dirs)} sessions for subject {subject_id}')

###################################################################################################
# Extract P100 per session
# ------------------------
#
# For each session we:
#
# - Load the CSV into an MNE Raw object
# - Band-pass filter 1–30 Hz (FIR)
# - Epoch around stimulus markers, separating left and right eye
# - Subtract the hardware lag so t=0 is true photon delivery
# - Extract N75, P100, and N145 latencies with parabolic interpolation
#

sfreq = SAMPLE_FREQS[device_name]
ch_ind = EEG_INDICES[device_name]

results = []

for session_dir in session_dirs:
    csv_files = sorted(glob.glob(os.path.join(session_dir, 'recording_*.csv')))
    if not csv_files:
        continue

    # Parse recording date from filename
    fname = os.path.basename(csv_files[0])
    date_str = fname.replace('recording_', '').replace('.csv', '')
    try:
        session_date = datetime.strptime(date_str, '%Y-%m-%d-%H.%M.%S')
    except ValueError:
        session_date = None

    session_label = os.path.basename(session_dir)

    raw = load_csv_as_raw(csv_files, sfreq=sfreq, ch_ind=ch_ind,
                          replace_ch_names=None, verbose=0)
    raw.filter(1, 30, method='fir', verbose=False)

    events = find_events(raw, verbose=False)
    event_id = {'left_eye': 1, 'right_eye': 2}

    epochs = Epochs(raw, events=events, event_id=event_id,
                    tmin=-0.1, tmax=0.4, baseline=None,
                    reject={'eeg': 65e-6}, preload=True,
                    verbose=False)

    if hardware_lag:
        epochs.shift_time(-hardware_lag)

    drop_pct = (1 - len(epochs.events) / len(events)) * 100

    session_result = {
        'session': session_label,
        'date': session_date,
        'n_epochs_left': len(epochs['left_eye']),
        'n_epochs_right': len(epochs['right_eye']),
        'drop_pct': drop_pct,
    }

    for eye in ['left_eye', 'right_eye']:
        if len(epochs[eye]) < 10:
            session_result[f'{eye}_p100'] = np.nan
            continue

        evoked = epochs[eye].average(picks=['Oz'])

        n75_latency = get_peak('N75', evoked, 0.06, 0.125, 'neg')
        p100_latency = get_peak('P100', evoked, n75_latency, n75_latency + 0.1, 'pos')

        session_result[f'{eye}_p100'] = p100_latency * 1e3  # convert to ms

    results.append(session_result)

print(f'\nExtracted P100 from {len(results)} sessions')

###################################################################################################
# Summary table
# -------------

print(f'\n{"Session":<14} {"Date":<12} {"L-eye P100":>11} {"R-eye P100":>11} '
      f'{"IOD":>8} {"Drop%":>6} {"L-epochs":>9} {"R-epochs":>9}')
print('-' * 82)
for r in results:
    date_str = r['date'].strftime('%Y-%m-%d') if r['date'] else '—'
    left = r['left_eye_p100']
    right = r['right_eye_p100']
    iod = left - right if not (np.isnan(left) or np.isnan(right)) else np.nan
    print(f'{r["session"]:<14} {date_str:<12} {left:>9.2f}ms {right:>9.2f}ms '
          f'{iod:>6.2f}ms {r["drop_pct"]:>5.1f}% {r["n_epochs_left"]:>9} {r["n_epochs_right"]:>9}')

###################################################################################################
# Plot P100 latency over sessions
# --------------------------------

dates = [r['date'] for r in results]
left_p100 = [r['left_eye_p100'] for r in results]
right_p100 = [r['right_eye_p100'] for r in results]
iod = [l - r if not (np.isnan(l) or np.isnan(r)) else np.nan
       for l, r in zip(left_p100, right_p100)]

use_dates = all(d is not None for d in dates)
x = dates if use_dates else range(len(results))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Per-eye P100 latency
ax1.plot(x, left_p100, 'o-', label='Left eye', color='blue', linewidth=2)
ax1.plot(x, right_p100, 's-', label='Right eye', color='red', linewidth=2)
ax1.set_ylabel('P100 latency (ms)')
ax1.set_title('P100 Latency Across Sessions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Interocular difference
ax2.plot(x, iod, 'D-', color='green', linewidth=2)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax2.set_ylabel('Interocular difference (ms)')
ax2.set_xlabel('Session date' if use_dates else 'Session index')
ax2.set_title('Interocular P100 Latency Difference (Left − Right)')
ax2.grid(True, alpha=0.3)

if use_dates:
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

plt.tight_layout()
plt.show()
