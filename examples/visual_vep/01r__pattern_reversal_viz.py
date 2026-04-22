"""
Pattern Reversal VEP: Load and Visualize
=========================================

This example demonstrates loading, organizing, and visualizing evoked response
data from the Visual Pattern Reversal VEP (PR-VEP) experiment.

An animation of a checkerboard reversal is shown (the checkerboard squares'
colours are toggled once each half a second). Stimulus is rendered stereoscopically
through a Meta Quest HMD and triggered via OpenBCI Cyton.

The data used is recorded using an OpenBCI Cyton with a Tencom 20 channel cap,
with cup electrodes placed at Fp1, Fp2, T5, T6, O1, O2, Oz, Pz.

Per-trial PC-side latency correction is applied using ``app_motion_to_photon_latency_s``
from the LibOVR compositor frame stats sidecar, and the residual Quest Link +
panel lag is handled by a fixed ``link_panel_lag`` constant.
"""

###################################################################################################
# Setup
# ---------------------

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd

from mne import Epochs, find_events, concatenate_raws

from eegnb import get_recording_dir
from eegnb.analysis.utils import load_csv_as_raw
from eegnb.analysis.vep_utils import get_pr_vep_latencies
from eegnb.devices.utils import EEG_INDICES, SAMPLE_FREQS

# sphinx_gallery_thumbnail_number = 3

###################################################################################################
# Hardware lag definitions
# ---------------------
#
# The total flip-return → photon delay on this rig is split into a measured
# part and an unmeasured residual:
#
# - **Measured per-trial (PC side)**: ``app_motion_to_photon_latency_s`` from
#   LibOVR frame stats, applied trial-by-trial below.
# - **Residual fit (link_panel_lag)**: Quest Link video encode/decode + USB transport
#   + panel scan-out + LCD response. Rough budgets from public benchmarks:
#   Link transport ≈ 20–40 ms, Panel + LCD ≈ 10–20 ms, total range ≈ 30–60 ms.
#   The Cyton RF transmission adds a further unmeasured ~1–5 ms (proprietary stack,
#   not standard BLE).
#

# Center of estimated unmeasured residual range (s)
link_panel_lag = 0.025
# ± half-range (s)
link_panel_lag_err = 0.015

###################################################################################################
# Load Data
# ---------------------
#
# Load all recordings for the session and concatenate into a single raw object.
# The timing sidecar CSV is parsed per-file and concatenated to match events.
#
# Session 8+ timing sidecars include LibOVR perf-stat columns and a ``#``-prefixed
# ``libovr_to_wallclock_offset_s`` metadata row. Earlier sessions used a 5-column
# schema without per-trial compositor latency.
#

SESSION_NB = 9
assert SESSION_NB >= 8, (
    f"This script assumes the session 8+ timing schema "
    f"(LibOVR perf stats + libovr_to_wallclock_offset_s metadata row); got session {SESSION_NB}"
)

SUBJECT_ID = 0
DEVICE_NAME = 'cyton'
EXPERIMENT = 'visual-PRVEP'
DISPLAY = 'quest-2_120Hz'
MONTAGE = 'cap'
SITE = f'{DISPLAY}_{MONTAGE}'

recording_dir = get_recording_dir(DEVICE_NAME, EXPERIMENT, SUBJECT_ID, SESSION_NB, site=SITE)
print(f"[data] recording dir: {recording_dir}")

# Exclude _timing.csv sidecars — they live alongside the EEG CSVs
recording_files = sorted(p for p in recording_dir.glob('*.csv') if not p.stem.endswith('_timing'))
print(f"[data] found {len(recording_files)} EEG recording(s): {[p.name for p in recording_files]}")

per_recording = []
for p in recording_files:
    timing_path = p.with_name(p.stem + '_timing.csv')
    if not timing_path.exists():
        print(f"[skip] No timing sidecar for {p.name}")
        continue

    rec_raw = load_csv_as_raw([str(p)], sfreq=250, ch_ind=EEG_INDICES['cyton'],
                               aux_ind=None, replace_ch_names=None, verbose=False)

    rec_timing = pd.read_csv(timing_path, comment='#').reset_index(drop=True)

    rec_events = find_events(rec_raw, shortest_event=1, verbose=False)

    n = min(len(rec_events), len(rec_timing))
    if len(rec_events) != len(rec_timing):
        print(f"[warn] {p.name}: events={len(rec_events)}, timing={len(rec_timing)} — truncating to {n}")
    per_recording.append({
        'raw': rec_raw,
        'events': rec_events[:n],
        'timing': rec_timing.iloc[:n].reset_index(drop=True),
    })

if not per_recording:
    raise RuntimeError(
        f"No recordings loaded from {recording_dir}. "
        "Check SUBJECT_ID, SESSION_NB, and DEVICE_NAME are correct, "
        "and that each EEG CSV has a matching _timing.csv sidecar."
    )

raw, events = concatenate_raws(
    [rec['raw'] for rec in per_recording],
    events_list=[rec['events'] for rec in per_recording],
)
timing_df = pd.concat([rec['timing'] for rec in per_recording], ignore_index=True)
assert len(events) == len(timing_df), "per-file truncation should keep events and timing aligned"

print(f"\n[raw] sfreq={raw.info['sfreq']} Hz, n_samples={raw.n_times}, duration={raw.times[-1]:.1f}s")
print(f"[raw] channels: {raw.ch_names}")

###################################################################################################
# Visualize the power spectrum
# ----------------------------

raw.plot_psd()

###################################################################################################
# Filtering
# ----------------------------
#
# Use FIR (linear phase) rather than IIR to avoid frequency-dependent group delay,
# which would shift the P100 peak by an amount that depends on its spectral content,
# contaminating latency measurements. MNE's zero-phase FIR cancels even the constant
# delay so the filtered P100 sits at the same sample as the unfiltered one.
# Using ISCEV bandpass standard: 1–100 Hz.
#

hp, lp = 1, 100
raw.filter(hp, lp, method='fir')
raw.plot_psd(fmin=hp, fmax=lp)

###################################################################################################
# Per-trial PC-side photon-latency correction
# ---------------------
#
# Each event sample index is shifted by the per-trial measured
# ``app_motion_to_photon_latency_s`` from LibOVR frame stats — a retrospective
# measurement of how long the frame actually took to reach the compositor/vsync.
#
# Missing trials (typically the first frame or two before perf stats are populated)
# fall back to the session mean.
#

pc_lag_s = timing_df['app_motion_to_photon_latency_s'].values.astype(float)
valid = np.isfinite(pc_lag_s) & (pc_lag_s > 0)
if (~valid).any():
    fallback = pc_lag_s[valid].mean() if valid.any() else 0.0
    print(f"[warn] {int((~valid).sum())}/{len(pc_lag_s)} trials missing "
          f"app_motion_to_photon_latency_s — using mean fallback {fallback*1000:.2f} ms")
    pc_lag_s = np.where(valid, pc_lag_s, fallback)

sample_shifts = np.round(pc_lag_s * raw.info['sfreq']).astype(int)
print(f"\n[pc-lag] app_motion_to_photon_latency_s (ms):  "
      f"min={pc_lag_s.min()*1000:.2f}  "
      f"max={pc_lag_s.max()*1000:.2f}  "
      f"mean={pc_lag_s.mean()*1000:.2f}  "
      f"std={pc_lag_s.std()*1000:.2f}  "
      f"|shift| samples: max={np.abs(sample_shifts).max()}")

# Per-trial PC-side-corrected event array.
events_corrected = events.copy()
events_corrected[:, 0] += sample_shifts

###################################################################################################
# Hardware lag breakdown chart
# ----------------------------

pc_pipeline_lag = pc_lag_s.mean() * 1000
unmeasured_lag = link_panel_lag * 1000

fig_lag, ax_lag = plt.subplots(figsize=(8, 4))
y_pos = 0

ax_lag.barh(y_pos, pc_pipeline_lag, color='#4c72b0', edgecolor='white',
            label=f'PC Pipeline (measured): {pc_pipeline_lag:.1f} ms')
ax_lag.barh(y_pos, unmeasured_lag, left=pc_pipeline_lag, color='#c44e52', edgecolor='white',
            label=f'Quest Link + Panel + Cyton RF (unmeasured): {unmeasured_lag:.1f} ms')

ax_lag.set_yticks([])
ax_lag.set_xlabel('Latency from Trigger (ms)')
ax_lag.set_title('Composition of VEP Hardware Lag')

ax_lag.errorbar(pc_pipeline_lag / 2, y_pos, xerr=pc_lag_s.std() * 1000,
                color='#aec6e8', capsize=5, lw=2, label='Measured Variance (±1 SD)')
ax_lag.errorbar(pc_pipeline_lag + (unmeasured_lag / 2), y_pos,
                xerr=link_panel_lag_err * 1000,
                color='black', capsize=5, lw=2,
                label=f'Unmeasured Uncertainty (±{link_panel_lag_err*1000:.0f}ms)')

handles, labels = ax_lag.get_legend_handles_labels()
fig_lag.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=8, frameon=True)
fig_lag.subplots_adjust(bottom=0.38)

###################################################################################################
# Epoching
# ----------------------------
#
# Epoch around stimulus onsets, separating left- and right-eye trials.
# Epochs are shifted by ``link_panel_lag`` to account for the residual hardware lag.
#

event_id = {'left_eye': 1, 'right_eye': 2}
print(f"\n[events] total={len(events)}, "
      f"left_eye={int((events[:, 2] == 1).sum())}, "
      f"right_eye={int((events[:, 2] == 2).sum())}")

PICK_CH = 'Oz'  # ISCEV-standard electrode
REJECT_UV = 35e-6
BASELINE = (-0.1, 0)

ch_epochs = Epochs(raw, events=events, event_id=event_id,
                   tmin=-0.1, tmax=0.4, baseline=BASELINE,
                   reject={'eeg': REJECT_UV},
                   preload=True, verbose=False, picks=[PICK_CH],
                   metadata=timing_df,
                   event_repeated='drop')
ch_epochs.shift_time(-link_panel_lag)

n_left = len(ch_epochs['left_eye'])
n_right = len(ch_epochs['right_eye'])
n_total = n_left + n_right
drop_pct = (1 - n_total / len(events)) * 100
print(f"\n[{PICK_CH} epochs] reject ptp = {REJECT_UV * 1e6:.0f} uV")
print(f"  kept {n_total}/{len(events)}  "
      f"(left={n_left}, right={n_right})  drop={drop_pct:.1f}%")

# Corrected-events epochs on the same trial set for overlay.
ch_epochs_corr = Epochs(raw, events=events_corrected[ch_epochs.selection],
                        event_id=event_id, tmin=-0.1, tmax=0.4, baseline=BASELINE,
                        reject=None, preload=True, verbose=False, picks=[PICK_CH],
                        metadata=timing_df.iloc[ch_epochs.selection].reset_index(drop=True),
                        event_repeated='drop')
ch_epochs_corr.shift_time(-link_panel_lag)

###################################################################################################
# Oz evoked response: Left Eye vs Right Eye
# -----------------------------------------
#
# Solid lines: per-trial PC lag corrected. Dotted lines: mean-corrected baseline.
# Shaded regions: ±1 SEM across trials.
#

from scipy.ndimage import maximum_filter1d, minimum_filter1d

evoked_left = ch_epochs['left_eye'].average(picks=[PICK_CH])
evoked_right = ch_epochs['right_eye'].average(picks=[PICK_CH])

times = evoked_left.times * 1000
left_data = evoked_left.data[0] * 1e6
right_data = evoked_right.data[0] * 1e6

LANDMARK_MS = [75, 100, 145]  # N75, P100, N145
LANDMARK_COLORS = ['#888888', 'green', '#555555']
LANDMARK_LABELS = ['N75 (75 ms)', 'P100 (100 ms)', 'N145 (145 ms)']

sfreq = evoked_left.info['sfreq']
times_mean_corr = times - (pc_lag_s.mean() * 1000)

evoked_left_corr  = ch_epochs_corr['left_eye'].average(picks=[PICK_CH])
evoked_right_corr = ch_epochs_corr['right_eye'].average(picks=[PICK_CH])
left_corr  = evoked_left_corr.data[0] * 1e6
right_corr = evoked_right_corr.data[0] * 1e6

left_trials  = ch_epochs_corr['left_eye'].get_data(picks=[PICK_CH])[:, 0, :] * 1e6
right_trials = ch_epochs_corr['right_eye'].get_data(picks=[PICK_CH])[:, 0, :] * 1e6
left_sem  = left_trials.std(axis=0) / np.sqrt(len(left_trials))
right_sem = right_trials.std(axis=0) / np.sqrt(len(right_trials))

# Detect and report P100 peak latencies
n75_left, p100_left, n145_left   = get_pr_vep_latencies(evoked_left_corr)
n75_right, p100_right, n145_right = get_pr_vep_latencies(evoked_right_corr)

def print_peak_info(eye_name, peak_info):
    if peak_info is not None:
        latency_ms = round(peak_info['latency'] * 1e3, 2)
        uv = round(peak_info['amplitude'] * 1e6, 2)
        print(f"[{eye_name}] {peak_info['name']} Peak: {uv} µV at {latency_ms} ms  (ch={peak_info['channel']})")

for eye, peaks in [('Left Eye', (n75_left, p100_left, n145_left)),
                   ('Right Eye', (n75_right, p100_right, n145_right))]:
    for p in peaks:
        print_peak_info(eye, p)

def plot_ch(ax, data, color, eye_label, sem=None, data_mean_corr=None, times_mean_corr=None):
    ax.plot(times, data, color=color, linewidth=2, label=f'{eye_label} (Per-trial corrected)')
    if data_mean_corr is not None and times_mean_corr is not None:
        ax.plot(times_mean_corr, data_mean_corr, color=color, linestyle=':', alpha=0.6,
                linewidth=1.6, label=f'{eye_label} (Mean corrected)')
    if sem is not None:
        ax.fill_between(times, data - sem, data + sem, color=color, alpha=0.25,
                        label=f'{eye_label} ±1 SEM')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

fig, ax = plt.subplots(figsize=(10, 6))

plot_ch(ax, left_corr,  'blue', 'Left Eye',  sem=left_sem,
        data_mean_corr=left_data,  times_mean_corr=times_mean_corr)
plot_ch(ax, right_corr, 'red',  'Right Eye', sem=right_sem,
        data_mean_corr=right_data, times_mean_corr=times_mean_corr)

for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
    ax.axvline(x=ms, color=col, linestyle='--', alpha=0.6, label=lbl)

ax.set_title(f'Evoked P100 Response: Left vs Right Eye (PC Lag Corrected) — {PICK_CH}')
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper right')
fig.tight_layout()
plt.show()
