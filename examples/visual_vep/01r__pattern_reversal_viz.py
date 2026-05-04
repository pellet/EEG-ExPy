# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:sphinx
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: eeg-experiments
#     language: python
#     name: python3
# ---

"""

# Pattern Reversal VEP: Load and Visualize

This example demonstrates loading, organizing, and visualizing evoked response
data from the Visual Pattern Reversal VEP (PR-VEP) experiment.

An animation of a checkerboard reversal is shown (the checkerboard squares'
colours are toggled once each half a second). Stimulus is rendered stereoscopically
through a Meta Quest HMD, with synchronization markers (triggers) sent to the OpenBCI Cyton.

The data used is recorded using an OpenBCI Cyton with a Tencom 20 channel cap,
with cup electrodes placed at M1 reference, Fz, Pz, P7, P8, O1, O2, Oz, M2 and an ear-clip on A2 for ground.

Per-trial PC-side latency correction is applied using ``app_motion_to_photon_latency_s``
from the LibOVR compositor frame stats sidecar, and the residual Quest Link +
panel lag is handled by a fixed ``link_panel_lag`` constant.

**12 biomarkers** are computed, grouped into three sections:

- **Pre-chiasmatic / optic nerve** (BM1–BM5): inter-ocular latency difference
  (IOLD), per-size IOLD, spatial-frequency slope, amplitude ratio, bootstrap CIs.
- **Morphological** (BM6): W-peak / bifurcated P100 detection.
- **Post-chiasmatic / cortical** (BM7–BM12): hemispheric O1/O2 asymmetry,
  inter-ocular Δ-asymmetry, lateral P7/P8, lateral Δ-asymmetry, composite
  hemispheres, topology QC.

Results are persisted to ``biomarkers.json`` in the recording directory and
consumed by ``02r__pattern_reversal_longitudinal.py`` for trend analysis.

"""

###############################################################################
# ## Setup
#
#

import os
import json
import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

from scipy.stats import trim_mean
from scipy.signal import find_peaks

from mne import Epochs, EvokedArray, find_events, concatenate_raws

from eegnb import get_recording_dir
from eegnb.analysis.utils import load_csv_as_raw
from eegnb.datasets import fetch_dataset
from eegnb.analysis.vep_utils import (
    get_pr_vep_latencies,
    ISCEV_CHECK_DEG_LARGE, ISCEV_CHECK_DEG_SMALL,
    IOLD_FLAG_MS, LOG2_AMP_FLAG,
    trimmed_average, json_safe_float,
    compute_iold, compute_iold_per_size, compute_amplitude_ratio, compute_check_size_slope,
    bootstrap_p100_latency, compute_hemi_asymmetry, compute_hemi_delta_asymmetry,
)
_f = json_safe_float
from eegnb.devices.utils import EEG_INDICES, SAMPLE_FREQS

# sphinx_gallery_thumbnail_number = 3

###############################################################################
# ## Hardware lag definitions (PsychoPy / Meta-Link path)
#
#
# Meta-Link path total flip-return → photon delay on this rig is split into a measured
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
#
#

###############################################################################
# ## Stimulus calibration caveat -- Quest panel is not ISCEV-calibrated
#
# Clinical PR-VEP norms (P100 ~100 ms, ~10-20 uV at Oz-Fz on adults) are
# derived from photometrically-spec'd CRT or LCD monitors at a fixed
# viewing distance. The Quest setup deviates on the panel-photometry
# axis -- the Quest 2 fast-switching LCD is not calibrated to a specific
# cd/m^2 or contrast ratio. Cortical drive (and therefore P100 amplitude)
# scales with these, so absolute amplitudes will diverge from clinical
# norms by a constant factor.
#
# Fortunately, this is still highly effective for differential analysis!
# Because the display characteristics are stable, within-subject comparisons
# (like inter-ocular differences) remain extremely robust. Furthermore, the
# fast-switching LCD provides excellent temporal precision, yielding highly
# reliable latency measurements despite the absolute amplitude shift.
#
# Field & check size are NOT a calibration concern on this path. The
# stimulus is rendered at a runtime-derived PPD (Pixels Per Degree at
# field centre) read from the OVR runtime each session via
# ``Rift.pixelsPerTanAngleAtCenter`` -- not estimated from a spec sheet.
# IPD (Inter-Pupillary Distance, the distance between the user's pupils
# in mm) is similarly read from the runtime via ``eyeToNoseDistance`` and
# is already baked into the per-eye projection matrices PsychoPy uses,
# so it does not bias the angular extent of stimulus content. Both PPD
# and IPD are written into the ``_timing.csv`` header at session start
# (see ``log_display_info`` in ``eegnb/devices/vr.py``), and
# the stimupy checkerboard is then sized by a prescribed degrees-per-check
# value (1.0 deg = 60 arcmin = ISCEV "large", 0.25 deg = 15 arcmin =
# ISCEV "small"). Compositor barrel-distortion correction is applied
# downstream of the rendered eye-buffer texture and does not enter the
# calculation. The residual uncertainty (much smaller than a spec-sheet
# estimate) comes from eye relief -- how far the user's pupil sits from
# the lens, which slightly biases off-axis angular size and peripheral
# vignetting -- and per-unit lens manufacturing variance. Both affect
# peripheral checks more than the foveal ones that drive P100, so for
# ISCEV-relevant central-field analysis the residual is well under 1%.
#
# Expected morphology differences (to characterise as more recordings come in):
# - Larger early negative deflections (N75-ish) plausible if panel contrast
#   / field size drives stronger extrastriate contribution than clinical
#   norms assume.
# - Latency offset ~10-20 ms on the Meta-Link path.
#
# Absolute P100 latency / amplitude here are NOT interchangeable with
# clinical PR-VEP norms. Differential biomarkers (IOLD, slope, amplitude
# ratio) are robust to these confounds and remain interpretable.
#
#
#

###############################################################################
# ## Bracketing / replacing ``link_panel_lag`` empirically
#
# The 25 ± 15 ms residual below is a budgeted estimate, not a measurement.
# Differential biomarkers (IOLD, slopes, ratios, Δ-asymmetry) are robust to
# this offset because both eyes share the same path, so the residual cancels
# in any L−R contrast. Absolute P100 latency (vs clinical norms) does NOT
# survive the residual and should not be reported as a clinical number until
# the residual is pinned down. Two paths to do that, in increasing order of
# rigour:
#
# 1. **Software baseline (error-prone, free).** Run a session on a control
#    subject with intact optic pathways and compare the measured P100 (after
#    PC-side correction only, ``link_panel_lag = 0``) to the clinical norm
#    (~100 ms at 60 arcmin on a calibrated CRT). The shift between observed
#    and norm is the residual. Caveats: the Quest panel isn't ISCEV-calibrated
#    so contrast/luminance will perturb absolute latency by an unknown amount
#    on top of the chain delay; controls vary ±5–10 ms at baseline; and any
#    one subject's value is noisy. Useful to *bracket* the residual to within
#    ~10 ms but not to certify it.
#
# 2. **Photodiode / optode (gold standard).** Tape a photodiode onto one HMD
#    eye lens facing the panel and route its analogue output to a Cyton aux
#    channel (or a second trigger line). The diode fires when actual photons
#    arrive at the eye — i.e. measures the entire chain (PsychoPy flip →
#    Quest Link → panel scan-out → LCD response) in a single sample. With a
#    photodiode trigger present, ``link_panel_lag`` becomes 0 by construction
#    and the per-trial PC-side correction is no longer needed either:
#    epoching off the diode event aligns trials to actual stimulus onset
#    with sub-frame precision. This makes absolute P100 latency a usable
#    biomarker rather than an estimate. The native-Quest absolute-latency
#    target supersedes this on its release path; for the meta-link path here,
#    a diode would close the gap immediately.
#
# ##############################################################################

# Center of estimated unmeasured residual range (s)
link_panel_lag = 0.025
# ± half-range (s)
link_panel_lag_err = 0.015

###############################################################################
# ## Load Data
#
# Load all recordings for the session and concatenate into a single raw object.
# The timing sidecar CSV is parsed per-file and concatenated to match events.
#


# --- CHANGE THESE PLACEHOLDERS TO POINT AT YOUR OWN RECORDING ---------------
SUBJECT_ID = 0
SESSION_NB = 16
DEVICE_NAME = 'cyton'
EXPERIMENT = 'visual-PRVEP'
DISPLAY = 'quest-2_120Hz'
MONTAGE = 'cap'
SITE = f'eegnb_examples/{DISPLAY}_{MONTAGE}'
# From session 016 the Fz cup moved to the SRB pin (hardware reference) and
# the old M1 cup moved to channel 1.  Applied per-recording at load time so
# recordings with the old CSV header ('Fz') and new ('M1') can be concatenated.
CH_REMAP = {'Fz': 'M1'} if SESSION_NB >= 16 else {}
# Minimum recording duration — skips short setup/restart runs.
# Move unwanted longer recordings to bad_recordings/ in the session directory;
# the glob will not find them there.
MIN_RECORDING_SECS = 120
# ---------------------------------------------------------------------------

eegnb_data_path = os.path.join(os.path.expanduser('~/'), '.eegnb', 'data')    
prvep_data_path = os.path.join(eegnb_data_path, EXPERIMENT, 'eegnb_examples')

if not os.path.isdir(prvep_data_path):
    print("Downloading PR-VEP example dataset from Google Drive...")
    fetch_dataset(data_dir=eegnb_data_path, experiment=EXPERIMENT, site='eegnb_examples')

recording_dir = get_recording_dir(DEVICE_NAME, EXPERIMENT, SUBJECT_ID, SESSION_NB, site=SITE)
print(f"[data] recording dir: {recording_dir}")

all_files = sorted(p for p in recording_dir.glob('*.csv') if not p.stem.endswith('_timing'))
print(f"[data] found {len(all_files)} EEG recording(s): {[p.name for p in all_files]}")

recording_files = []
for p in all_files:
    timing_path = p.with_name(p.stem + '_timing.csv')
    if not timing_path.exists():
        print(f"[skip] No timing sidecar: {p.name}")
        continue
    n_rows = sum(1 for _ in open(p)) - 1
    dur_secs = n_rows / 250
    if dur_secs < MIN_RECORDING_SECS:
        print(f"[skip] Too short ({dur_secs:.0f}s < {MIN_RECORDING_SECS}s): {p.name}")
        continue
    recording_files.append(p)

print(f"[data] using {len(recording_files)} recording(s): {[p.name for p in recording_files]}")

per_recording = []
for p in recording_files:
    timing_path = p.with_name(p.stem + '_timing.csv')

    rec_raw = load_csv_as_raw([str(p)], sfreq=250, ch_ind=EEG_INDICES['cyton'],
                               aux_ind=None, replace_ch_names=None, verbose=False)

    if CH_REMAP:
        remap = {k: v for k, v in CH_REMAP.items() if k in rec_raw.ch_names}
        if remap:
            rec_raw.rename_channels(remap)

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
        "Check SUBJECT_ID, SESSION_NB, DEVICE_NAME, and MIN_RECORDING_SECS."
    )

raw, events = concatenate_raws(
    [rec['raw'] for rec in per_recording],
    events_list=[rec['events'] for rec in per_recording],
)
timing_df = pd.concat([rec['timing'] for rec in per_recording], ignore_index=True)
assert len(events) == len(timing_df), "per-file truncation should keep events and timing aligned"

print(f"\n[raw] sfreq={raw.info['sfreq']} Hz, n_samples={raw.n_times}, duration={raw.times[-1]:.1f}s")
print(f"[raw] channels: {raw.ch_names}")

###############################################################################
# ## Recording quality diagnostic
#
# Two-stage contact quality check:
#
# 1. **Raw CSV (here)** — std / drift / p99 per channel directly from the recorded CSV, before any MNE processing. Mean-subtracts before computing metrics so DC offset does not inflate the flags. Detects whether flagged channels are isolated contacts or shared across all channels (loose M1/SRB reference).
# 2. **Post-epoch baseline (below, after epoching)** — pre-stimulus baseline RMS per channel after filtering + referencing. Absolute values are interpretable here; provides SNR at Oz and a go/no-go recommendation.

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(globals()['_dh'][0]).resolve().parents[3]))
from eegnb.analysis.recording_quality import check_session

_rq = check_session(recording_dir)
print(_rq['report'])

if _rq['shared_ref_suspect']:
    print()
    print("=" * 60)
    print("⚑ SHARED REFERENCE SUSPECT (M1/SRB loose)")
    print("  All-channel noise inflation detected.")
    print("  Every channel recorded through this reference is")
    print("  compromised. Biomarkers that depend on absolute")
    print("  amplitude or inter-channel ratios are unreliable.")
    print("  Re-seat M1 and re-record before trusting results.")
    print("=" * 60)
elif _rq['flagged_channels']:
    print(f"\n⚑ Flagged channels: {', '.join(_rq['flagged_channels'])}")
    print("  Isolated contact issue(s) — other channels are ok.")
else:
    print("\nAll channels within normal range — contact quality ok.")

###############################################################################
# ## Visualize the power spectrum
#
#

raw.plot_psd()

###############################################################################
# ## Filtering
#
# Use FIR (linear phase) rather than IIR to avoid frequency-dependent group delay,
# which would shift the P100 peak by an amount that depends on its spectral content,
# contaminating latency measurements. MNE's zero-phase FIR cancels even the constant
# delay so the filtered P100 sits at the same sample as the unfiltered one.
# Using ISCEV bandpass standard: 1–100 Hz.
#
#
#

hp, lp = 1, 100
raw.filter(hp, lp, method='fir')

###############################################################################
# ## Per-trial PC-side photon-latency correction
#
# Each event sample index is shifted by the per-trial measured
# ``app_motion_to_photon_latency_s`` from LibOVR frame stats — a retrospective
# measurement of how long the frame actually took to reach the compositor/vsync.
#
# Missing trials (typically the first frame or two before perf stats are populated)
# fall back to the session mean.
#
#
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

###############################################################################
# ## Hardware lag breakdown chart
#
#

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

###############################################################################
# ## Condition decoding
#
# Marker scheme:
# Reversal codes carry both eye and size (1–4).
# Block-start markers (100-103) are pushed at the start of each block but are redundant for trial epoching.
#
# Condition codes:

COND_TO_INT = {
    ('left_eye',  'large'): 1,
    ('right_eye', 'large'): 2,
    ('left_eye',  'small'): 3,
    ('right_eye', 'small'): 4,
}

# Drop block-start markers (100-103), keeping only the actual reversal markers (1-4).
mask = np.isin(events[:, 2], list(COND_TO_INT.values()))
events = events[mask]
events_corrected = events_corrected[mask]

###############################################################################
# ## Epoching parameters
#

event_id = {f"{eye}/{size}": code for (eye, size), code in COND_TO_INT.items()}
for cond, cid in event_id.items():
    n = int((events[:, 2] == cid).sum())
    print(f"[events] {cond}: {n}")

PICK_CH = 'Oz'  # ISCEV-standard electrode
HEMI_CHANNELS = ['O1', 'O2']  # Hemispheric channels for post-chiasmatic analysis
LATERAL_CHANNELS = ['P7', 'P8']  # Lateral extrastriate (V2/V3/MT). Generators on the
                                  # lateral cortical surface project mostly ipsilaterally
                                  # to the scalp, so P7/P8 are far less affected by the
                                  # paradoxical lateralization that makes O1/O2 ambiguous
                                  # for hemispheric localization. This makes P7/P8 a
                                  # relatively direct readout of unilateral lateral-
                                  # occipital / parieto-occipital cortex — useful for
                                  # detecting localized retro-chiasmatic / extrastriate
                                  # involvement (cortical lesions, focal hypofunction)
                                  # that Oz / O1 / O2 alone cannot cleanly side-localize.
# Topology QC channels: Pz gradient + Fz Halliday inversion.  Fz is absent
# when it is the hardware reference (session 016+), so filter against what's
# actually recorded to avoid pick errors downstream.
TOPOLOGY_CHANNELS = [ch for ch in ['Pz', 'Fz'] if ch in raw.ch_names]
ALL_PICK_CHANNELS = [PICK_CH] + HEMI_CHANNELS + LATERAL_CHANNELS + TOPOLOGY_CHANNELS
REJECT_UV = 150e-6
BASELINE = (-0.1, 0)

LANDMARK_MS = [75, 100, 145]  # N75, P100, N145
LANDMARK_COLORS = ['#888888', 'green', '#555555']
LANDMARK_LABELS = ['N75 (75 ms)', 'P100 (100 ms)', 'N145 (145 ms)']

P100_WIN_MS = (60, 160)  # P100 search window (positive max)
CHECK_SIZE_ARCMIN = {
    'large': ISCEV_CHECK_DEG_LARGE * 60.0,  # 60 arcmin
    'small': ISCEV_CHECK_DEG_SMALL * 60.0,  # 15 arcmin
}

###############################################################################
# ## Reference scheme selector
#
# Pick which EEG reference to analyse in this notebook run. Re-run the notebook
# with the other value to also persist its biomarkers (the persistence cell at
# the bottom merges into ``biomarkers.json`` rather than overwriting).
#
# - **Fz (ISCEV strict)**: Oz-Fz derivation. The ISCEV 2016 PR-VEP standard
#   recommends a mid-frontal reference (Fz), so this scheme is the choice for
#   direct comparison against published clinical norms. Trade-off: very
#   sensitive to Fz contact quality on dry-electrode rigs -- one bad Fz
#   contact can subtract artifact into Oz and invert P100 polarity.
# - **Linked Mastoid M1+M2**: ISCEV lists ear/mastoid as an acceptable
#   alternative reference; widely used in cognitive-neuroscience ERP work
#   because mastoids sit on bony prominence with less EMG and are far less
#   contact-noise-prone than Fz on dry actives. Absolute P100 latency /
#   amplitude differ slightly from the Fz-referenced waveform, so direct
#   comparison to Fz-referenced clinical norms is approximate; differential
#   biomarkers (IOLD, slope, asymmetry) are reference-invariant and remain
#   directly comparable. Required (not optional) for Biomarker 12's Halliday
#   Fz-inversion check, which needs Fz as a recorded channel rather than the
#   reference.
#
# M1 is the Cyton hardware reference (SRB pin), so stored channel data is
# already relative to M1. A zero-valued M1 channel is synthesised and averaged
# with the recorded M2, giving Oz-(M1+M2)/2 after the algebra.
#

REF_SCHEME = 'Fz (ISCEV)'  # 'Fz (ISCEV)'  or  'Linked Mastoid M1+M2'

if REF_SCHEME == 'Fz (ISCEV)':
    raw_ref = raw.copy()
    if 'Fz' in raw_ref.ch_names:
        # Fz is a recorded channel — subtract it as software reference
        raw_ref.set_eeg_reference(ref_channels=['Fz'])
    else:
        # Fz is the hardware SRB — data already in Fz space, no software step needed
        print("[ref] Fz is hardware reference — no software re-reference applied")
elif REF_SCHEME == 'Linked Mastoid M1+M2':
    raw_ref = raw.copy()
    if 'M1' not in raw_ref.ch_names:
        # M1 is the SRB (hardware reference) — synthesise it as zero so the
        # algebra (channel − M2/2) approximates linked mastoid
        m1_zero = raw_ref.copy().pick(['M2'])
        m1_zero._data[:] = 0
        m1_zero.rename_channels({'M2': 'M1'})
        raw_ref.add_channels([m1_zero])
    # M1 is a real recorded channel (session 016+) or the synthesised zero above
    raw_ref.set_eeg_reference(ref_channels=['M1', 'M2'])
else:
    raise ValueError(f'Unknown REF_SCHEME: {REF_SCHEME!r}')

ref_label = REF_SCHEME
results = {'ref_label': ref_label}
print(f"\n{'='*60}\nReference: {ref_label}\n{'='*60}")

raw_ref.compute_psd(fmin=hp, fmax=lp).plot()

###############################################################################
# ## Epoching
#

ch_epochs = Epochs(raw_ref, events=events, event_id=event_id,
                   tmin=-0.1, tmax=0.4, baseline=BASELINE,
                   reject={'eeg': REJECT_UV},
                   preload=True, verbose=False, picks=ALL_PICK_CHANNELS,
                   event_repeated='drop')
ch_epochs.shift_time(-link_panel_lag)

n_total = len(ch_epochs)
drop_pct = (1 - n_total / len(events)) * 100
print(f"\n[{PICK_CH}] reject ptp={REJECT_UV * 1e6:.0f} uV  "
      f"kept {n_total}/{len(events)} ({drop_pct:.1f}% dropped)")
results['n_trials_total'] = int(len(events))
results['n_trials_kept'] = int(n_total)
results['drop_pct'] = _f(drop_pct)
results['n_per_condition'] = {
    cond: int((events[:, 2] == cid).sum()) for cond, cid in event_id.items()
}

# Corrected-events epochs on the same kept trials.
ch_epochs_corr = Epochs(raw_ref, events=events_corrected[ch_epochs.selection],
                        event_id=event_id, tmin=-0.1, tmax=0.4, baseline=BASELINE,
                        reject=None, preload=True, verbose=False, picks=ALL_PICK_CHANNELS,
                        event_repeated='drop')
ch_epochs_corr.shift_time(-link_panel_lag)

def avg_eyes(ep, eye_prefix):
    keys = [k for k in event_id if k.startswith(eye_prefix)]
    return trimmed_average(ep[keys]) if keys else None

# =========================================================================
# WAVEFORM PLOTS
# =========================================================================

""
###############################################################################
# Stage 2 — Post-epoch baseline quality (filtered + referenced)
# Absolute values are meaningful here. Baseline window: -100 to 0 ms.
###############################################################################

BASELINE_WIN      = (-0.1, 0.0)
NOISE_FACTOR_EP   = 1.5   # flag if channel baseline RMS > this × group median
OZ_SNR_MIN        = 2.0   # flag if Oz P100 SNR falls below this

baseline_mask = (ch_epochs_corr.times >= BASELINE_WIN[0]) & \
                (ch_epochs_corr.times <= BASELINE_WIN[1])

baseline_data = ch_epochs_corr.get_data()[:, :, baseline_mask]  # (epochs, ch, times)
baseline_rms  = np.sqrt(np.mean(baseline_data ** 2, axis=(0, 2))) * 1e6  # µV per channel

ch_names_ep = ch_epochs_corr.ch_names
med_rms = float(np.median(baseline_rms))

print("Stage 2 — Baseline RMS per channel (post-filter, post-reference, -100–0 ms)")
print(f"{'Channel':<8}  {'RMS µV':>8}  {'Factor':>8}  Status")
print("-" * 44)
quality_flags = {}
for ch, rms in zip(ch_names_ep, baseline_rms):
    factor = rms / med_rms
    flag = factor > NOISE_FACTOR_EP
    quality_flags[ch] = {'rms_uv': round(float(rms), 2), 'flag': flag}
    print(f"{ch:<8}  {rms:>8.2f}  {factor:>8.2f}  {'⚑ FLAG' if flag else 'ok'}")

# Oz SNR: best detected P100 / Oz baseline RMS
oz_idx = ch_names_ep.index('Oz')
oz_rms = baseline_rms[oz_idx]
print(f"\nOz baseline RMS = {oz_rms:.2f} µV   group median = {med_rms:.2f} µV")

flagged_chs = [ch for ch, v in quality_flags.items() if v['flag']]
if flagged_chs:
    print(f"\n⚑ Noisy channels (>{NOISE_FACTOR_EP}× median): {flagged_chs}")
    if len(flagged_chs) == len(ch_names_ep):
        print("  All channels elevated → shared reference (M2/SRB) is likely the cause.")
    else:
        print("  Subset of channels → individual electrode contact issue(s).")
else:
    print(f"\nAll channels within {NOISE_FACTOR_EP}× median baseline — contact quality ok.")

###############################################################################
# ## Oz evoked: left vs right eye
#
# Solid lines: per-trial PC lag corrected. Dotted lines: mean-corrected baseline.
# Shaded regions: ±1 SEM across trials.
#

evoked_left_large  = trimmed_average(ch_epochs['left_eye/large'])
evoked_right_large = trimmed_average(ch_epochs['right_eye/large'])
evoked_left_small  = trimmed_average(ch_epochs['left_eye/small'])
evoked_right_small = trimmed_average(ch_epochs['right_eye/small'])

idx_oz = evoked_left_large.ch_names.index(PICK_CH)

times = evoked_left_large.times * 1000
left_data_large  = evoked_left_large.data[idx_oz]  * 1e6
right_data_large = evoked_right_large.data[idx_oz] * 1e6
left_data_small  = evoked_left_small.data[idx_oz]  * 1e6
right_data_small = evoked_right_small.data[idx_oz] * 1e6

times_mean_corr = times - (pc_lag_s.mean() * 1000)

evoked_left_corr_large  = trimmed_average(ch_epochs_corr['left_eye/large'])
evoked_right_corr_large = trimmed_average(ch_epochs_corr['right_eye/large'])
left_corr_large  = evoked_left_corr_large.data[idx_oz]  * 1e6
right_corr_large = evoked_right_corr_large.data[idx_oz] * 1e6

evoked_left_corr_small  = trimmed_average(ch_epochs_corr['left_eye/small'])
evoked_right_corr_small = trimmed_average(ch_epochs_corr['right_eye/small'])
left_corr_small  = evoked_left_corr_small.data[idx_oz]  * 1e6
right_corr_small = evoked_right_corr_small.data[idx_oz] * 1e6

left_trials_large  = ch_epochs_corr['left_eye/large'].get_data(picks=[PICK_CH])[:, 0, :] * 1e6
right_trials_large = ch_epochs_corr['right_eye/large'].get_data(picks=[PICK_CH])[:, 0, :] * 1e6
left_sem_large  = left_trials_large.std(axis=0)  / np.sqrt(len(left_trials_large))
right_sem_large = right_trials_large.std(axis=0) / np.sqrt(len(right_trials_large))

left_trials_small  = ch_epochs_corr['left_eye/small'].get_data(picks=[PICK_CH])[:, 0, :] * 1e6
right_trials_small = ch_epochs_corr['right_eye/small'].get_data(picks=[PICK_CH])[:, 0, :] * 1e6
left_sem_small  = left_trials_small.std(axis=0)  / np.sqrt(len(left_trials_small))
right_sem_small = right_trials_small.std(axis=0) / np.sqrt(len(right_trials_small))

# Size-averaged evokeds used as primary input for most biomarkers.
evoked_left_corr_avg  = avg_eyes(ch_epochs_corr, 'left_eye')
evoked_right_corr_avg = avg_eyes(ch_epochs_corr, 'right_eye')
n75_left,  p100_left,  n145_left  = get_pr_vep_latencies(evoked_left_corr_avg.copy().pick([PICK_CH]))
n75_right, p100_right, n145_right = get_pr_vep_latencies(evoked_right_corr_avg.copy().pick([PICK_CH]))

for eye_name, peaks in [('Left Eye (Avg)',  (n75_left,  p100_left,  n145_left)),
                         ('Right Eye (Avg)', (n75_right, p100_right, n145_right))]:
    for peak in peaks:
        if peak is not None:
            print(f"[{eye_name}] {peak['name']} Peak: "
                  f"{round(peak['amplitude']*1e6, 2)} µV at "
                  f"{round(peak['latency']*1e3, 2)} ms  (ch={peak['channel']})")

    n75, p100, n145 = peaks
    if n75 is not None and p100 is not None:
        ptp_1 = (p100['amplitude'] - n75['amplitude']) * 1e6
        print(f"[{eye_name}] N75-P100 Peak-to-Peak: {ptp_1:.2f} µV")
    if p100 is not None and n145 is not None:
        ptp_2 = (p100['amplitude'] - n145['amplitude']) * 1e6
        print(f"[{eye_name}] P100-N145 Peak-to-Peak: {ptp_2:.2f} µV")
    if n75 is not None and p100 is not None and n145 is not None:
        print(f"[{eye_name}] Total N75-P100-N145 Energy: {ptp_1 + ptp_2:.2f} µV")

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

fig, (ax_large, ax_small) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

plot_ch(ax_large, left_corr_large,  'blue', 'Left Eye',  sem=left_sem_large,
        data_mean_corr=left_data_large,  times_mean_corr=times_mean_corr)
plot_ch(ax_large, right_corr_large, 'red',  'Right Eye', sem=right_sem_large,
        data_mean_corr=right_data_large, times_mean_corr=times_mean_corr)

for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
    ax_large.axvline(x=ms, color=col, linestyle='--', alpha=0.6, label=lbl)

ax_large.set_title(f'[{ref_label}] Large Checks: Left vs Right Eye — {PICK_CH}')
handles, lbls = ax_large.get_legend_handles_labels()
ax_large.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
                fontsize=10, loc='upper right')

plot_ch(ax_small, left_corr_small,  'blue', 'Left Eye',  sem=left_sem_small,
        data_mean_corr=left_data_small,  times_mean_corr=times_mean_corr)
plot_ch(ax_small, right_corr_small, 'red',  'Right Eye', sem=right_sem_small,
        data_mean_corr=right_data_small, times_mean_corr=times_mean_corr)

for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
    ax_small.axvline(x=ms, color=col, linestyle='--', alpha=0.6, label=lbl)

ax_small.set_title(f'[{ref_label}] Small Checks: Left vs Right Eye — {PICK_CH}')
handles, lbls = ax_small.get_legend_handles_labels()
ax_small.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
                fontsize=10, loc='upper right')

fig.tight_layout()
plt.show()

###############################################################################
# ## Occipital channel comparison (Oz, O1, O2) — size-averaged per eye
#
# O1/O2 alongside Oz on the same axes. Subject0000 has a larger right occipital
# lobe that crosses the midline. Due to paradoxical lateralization (right V1
# dipole projects to left scalp), this predicts O1 > O2 amplitude as a
# baseline anatomical effect independent of pathology. Confirming this here
# separates the structural asymmetry from any eye-dependent lesion signal.
#

fig_o1o2, axes_o1o2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
occ_styles = {
    'Oz': dict(color='black',   lw=2.5, ls='-',  alpha=1.0),
    'O1': dict(color='#9467bd', lw=1.8, ls='--', alpha=0.85),
    'O2': dict(color='#e377c2', lw=1.8, ls='--', alpha=0.85),
}

for ax, (eye_name, ev_avg) in zip(axes_o1o2,
                                   [('Left Eye',  evoked_left_corr_avg),
                                    ('Right Eye', evoked_right_corr_avg)]):
    if ev_avg is None:
        continue
    t_occ = ev_avg.times * 1000
    for ch, style in occ_styles.items():
        if ch not in ev_avg.ch_names:
            continue
        idx = ev_avg.ch_names.index(ch)
        ax.plot(t_occ, ev_avg.data[idx] * 1e6, label=ch, **style)

    for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
        ax.axvline(ms, color=col, linestyle='--', alpha=0.5,
                   label=lbl if ax == axes_o1o2[0] else '')
    ax.set_title(f'[{ref_label}] Occipital channels — {eye_name} (size-averaged)')
    ax.set_xlabel('Time (ms)')
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    if ax == axes_o1o2[0]:
        ax.set_ylabel('Amplitude (µV)')
    ax.grid(True, alpha=0.3)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
              fontsize=10, loc='upper right')

fig_o1o2.tight_layout()
plt.show()

###############################################################################
# ## Inter-ocular difference wave at Oz
#
# The full time course of the L−R contrast. Surfaces morphology effects that
# single-point biomarkers miss: delayed second peaks on the affected eye,
# split / W-shaped P100 from partially demyelinated fibres, broadening of
# the P100. Zero baseline = eyes match; sign convention follows IOLD (positive
# = left-eye delay near the P100 peak).
#

fig_diff, (ax_diff_large, ax_diff_small) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, size_label, l_data, r_data in [
    (ax_diff_large, 'Large', left_corr_large, right_corr_large),
    (ax_diff_small, 'Small', left_corr_small, right_corr_small),
]:
    diff = l_data - r_data
    ax.plot(times, l_data, color='blue', alpha=0.25, linewidth=1.2, label='Left Eye')
    ax.plot(times, r_data, color='red',  alpha=0.25, linewidth=1.2, label='Right Eye')
    ax.plot(times, diff,   color='#7d3c98', linewidth=2.5, label='L − R difference')
    ax.fill_between(times, 0, diff, color='#7d3c98', alpha=0.15)

    for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
        ax.axvline(x=ms, color=col, linestyle='--', alpha=0.6, label=lbl)

    ax.set_title(f'[{ref_label}] {size_label} Checks: L − R Difference Wave — {PICK_CH}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
              fontsize=10, loc='upper right')

fig_diff.tight_layout()
plt.show()

###############################################################################
# ## Diagnostic: jitter-correction impact
#
# Compares the P100 sharpness with and without per-trial PC lag correction.
#

evoked_left_uncorr  = avg_eyes(ch_epochs, 'left_eye')
evoked_right_uncorr = avg_eyes(ch_epochs, 'right_eye')

if all(x is not None for x in [evoked_left_corr_avg, evoked_right_corr_avg,
                                evoked_left_uncorr, evoked_right_uncorr]):
    fig_jitter, axes_j = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, eye_name, ev_u, ev_c, color in zip(
        axes_j, ['Left Eye', 'Right Eye'],
        [evoked_left_uncorr, evoked_right_uncorr],
        [evoked_left_corr_avg, evoked_right_corr_avg],
        ['blue', 'red']
    ):
        t_u = ev_u.times * 1000
        t_c = ev_c.times * 1000
        i_oz = ev_u.ch_names.index(PICK_CH)

        ax.plot(t_u, ev_u.data[i_oz] * 1e6, color='gray', linestyle='--', linewidth=2,
                label='Uncorrected')
        ax.plot(t_c, ev_c.data[i_oz] * 1e6, color=color, linewidth=2,
                label='PC-lag corrected')

        for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
            ax.axvline(x=ms, color=col, linestyle=':', alpha=0.5,
                       label=lbl if ax == axes_j[0] else '')

        ax.set_title(f'[{ref_label}] {eye_name}: Jitter Correction Impact — {PICK_CH}')
        ax.set_xlabel('Time (ms)')
        if ax == axes_j[0]:
            ax.set_ylabel('Amplitude (µV)')
        ax.axhline(0, color='black', alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
                  loc='upper right')

    fig_jitter.tight_layout()
    plt.show()

###############################################################################
# ## Diagnostic: estimator robustness
#
# Overlays single trials to reveal outlier contamination (e.g., blinks), then
# compares standard mean, median, and 10%-trimmed mean estimators.
#

fig_est, axes_e = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, eye_prefix, color in zip(axes_e, ['left_eye', 'right_eye'], ['blue', 'red']):
    keys = [k for k in event_id if k.startswith(eye_prefix)]
    if not keys:
        continue

    ep = ch_epochs_corr[keys].copy().pick([PICK_CH])
    data = ep.get_data() * 1e6
    if data.shape[0] == 0:
        continue

    data = data[:, 0, :]
    times_e = ep.times * 1000

    subset = data[:100] if data.shape[0] > 100 else data
    for trial in subset:
        ax.plot(times_e, trial, color='gray', alpha=0.08, linewidth=0.5)

    ax.plot(times_e, np.mean(data, axis=0),            color='orange', linestyle='--',
            linewidth=2, label='Standard mean')
    ax.plot(times_e, np.median(data, axis=0),          color='green',  linestyle='-.',
            linewidth=2, label='Median')
    ax.plot(times_e, trim_mean(data, 0.1, axis=0),     color=color,
            linewidth=3, label='10% Trimmed mean')

    for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
        ax.axvline(x=ms, color=col, linestyle=':', alpha=0.5,
                   label=lbl if ax == axes_e[0] else '')

    ax.set_title(f'[{ref_label}] {eye_prefix.replace("_", " ").title()}: '
                 f'Single Trials & Estimators — {PICK_CH}')
    ax.set_xlabel('Time (ms)')
    if ax == axes_e[0]:
        ax.set_ylabel('Amplitude (µV)')
    ax.set_ylim(-30, 30)
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
              loc='upper right')

fig_est.tight_layout()
plt.show()

###############################################################################
# ## Diagnostic: multi-channel topography
#
# Displays the trimmed-mean waveform across all posterior channels to confirm
# the Oz > O1/O2 > Pz generator gradient expected from a V1 source.
#

fig_mc, axes_mc = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

plot_channels = [ch for ch in ['Oz', 'O1', 'O2', 'P7', 'P8', 'Pz']
                 if ch in ch_epochs_corr.ch_names]
ch_colors = {'Oz': 'black', 'O1': '#9467bd', 'O2': '#e377c2',
             'P7': '#1f77b4', 'P8': '#d62728', 'Pz': '#2ca02c'}

for ax, eye_name, ev_avg in zip(axes_mc, ['Left Eye', 'Right Eye'],
                                 [evoked_left_corr_avg, evoked_right_corr_avg]):
    if ev_avg is None:
        continue

    t_mc = ev_avg.times * 1000
    for ch in plot_channels:
        idx = ev_avg.ch_names.index(ch)
        lw    = 3   if ch == 'Oz' else 1.5
        alpha = 1.0 if ch == 'Oz' else 0.7
        ax.plot(t_mc, ev_avg.data[idx] * 1e6,
                color=ch_colors.get(ch, 'gray'), linewidth=lw, alpha=alpha, label=ch)

    for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
        ax.axvline(x=ms, color=col, linestyle=':', alpha=0.5,
                   label=lbl if ax == axes_mc[0] else '')

    ax.set_title(f'[{ref_label}] {eye_name}: Multi-Channel Topography')
    ax.set_xlabel('Time (ms)')
    if ax == axes_mc[0]:
        ax.set_ylabel('Amplitude (µV)')
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
              loc='upper right')

fig_mc.tight_layout()
plt.show()

# =========================================================================
# SECTION 1: Pre-chiasmatic / Optic Nerve Biomarkers
#
# BM1–BM5 all measure the L−R optic-nerve contrast at Oz. Because both eyes
# share the same Quest → Cyton signal chain the unmeasured residual lags
# cancel in every L−R difference, making these biomarkers robust to the
# absolute timing uncertainty in this setup.
# =========================================================================
print("\n" + "="*70)
print("SECTION 1 — Pre-chiasmatic / Optic Nerve")
print("="*70)

###############################################################################
# ## BM1 — IOLD: inter-ocular P100 latency difference (pooled)
#
# The signed L−R difference in P100 latency at Oz (size-averaged evoked).
# > 6–8 ms is the most-cited clinical threshold for unilateral demyelination.
#

results['iold'] = compute_iold(p100_left, p100_right)

print("\n--- BM1: IOLD (pooled) ---")
if results['iold'] is None:
    print("[BM1] Cannot compute — P100 not detected in one or both eyes")
else:
    d = results['iold']
    direction = "left delayed" if d['iold_ms'] > 0 else "right delayed"
    status = 'FLAG' if d['flag'] else f'within ±{IOLD_FLAG_MS:.0f} ms'
    print(f"[BM1] P100  Left = {d['p100_left_ms']:.2f} ms   Right = {d['p100_right_ms']:.2f} ms")
    print(f"[BM1] IOLD  L−R  = {d['iold_ms']:+.2f} ms ({direction})  [{status}]")

    fig_bm1, ax_bm1 = plt.subplots(figsize=(6, 5))
    bar_vals = [d['p100_left_ms'], d['p100_right_ms']]
    bars = ax_bm1.bar([0, 1], bar_vals, color=['#1f77b4', '#d62728'], alpha=0.75, width=0.45)
    y_top = max(bar_vals) + 5
    ax_bm1.annotate('', xy=(1, y_top), xytext=(0, y_top),
                    arrowprops=dict(arrowstyle='<->', color='#7d3c98', lw=2))
    flag_marker = '  ⚑ FLAG' if d['flag'] else ''
    ax_bm1.text(0.5, y_top + 1.5, f'IOLD = {d["iold_ms"]:+.1f} ms{flag_marker}',
                ha='center', va='bottom', color='#7d3c98', fontsize=11, fontweight='bold')
    # shade the ±threshold band around right-eye bar to give visual reference
    ref = d['p100_right_ms']
    ax_bm1.axhspan(ref - IOLD_FLAG_MS, ref + IOLD_FLAG_MS,
                   alpha=0.10, color='orange', label=f'±{IOLD_FLAG_MS:.0f} ms threshold band')
    ax_bm1.set_xticks([0, 1])
    ax_bm1.set_xticklabels(['Left Eye', 'Right Eye'], fontsize=12)
    ax_bm1.set_ylabel('P100 latency at Oz (ms)')
    ax_bm1.set_title(f'[{ref_label}] BM1: IOLD — inter-ocular P100 latency')
    ax_bm1.set_ylim(0, y_top + 8)
    ax_bm1.legend(fontsize=9)
    ax_bm1.grid(axis='y', alpha=0.3)
    fig_bm1.tight_layout()
    plt.show()

###############################################################################
# ## BM2 — IOLD per check size
#
# Demyelination preferentially delays high-spatial-frequency (small-check)
# responses, so the per-size IOLD often surfaces lateralised dysfunction that
# the size-pooled IOLD averages out.
#

results['iold_per_size'] = compute_iold_per_size(ch_epochs_corr, event_id, PICK_CH)

print("\n--- BM2: IOLD per check size ---")
for size, d in results['iold_per_size'].items():
    if d is None:
        print(f"[BM2/{size}] Cannot compute — P100 not detected at one or both eyes")
        continue
    direction = "left delayed" if d['iold_ms'] > 0 else "right delayed"
    status = 'FLAG' if d['flag'] else f'within ±{IOLD_FLAG_MS:.0f} ms'
    print(f"[BM2/{size}] P100  Left = {d['p100_left_ms']:.2f} ms   Right = {d['p100_right_ms']:.2f} ms")
    print(f"[BM2/{size}] IOLD  L−R  = {d['iold_ms']:+.2f} ms ({direction})  [{status}]")

# Grouped bar: L/R per check size
sizes_present = [s for s, d in results['iold_per_size'].items() if d is not None]
if sizes_present:
    fig_bm2, ax_bm2 = plt.subplots(figsize=(8, 5))
    x2 = np.arange(len(sizes_present))
    w2 = 0.32
    for i, (eye_prefix, color, eye_label) in enumerate(
            [('left', '#1f77b4', 'Left Eye'), ('right', '#d62728', 'Right Eye')]):
        lats = [results['iold_per_size'][s][f'p100_{eye_prefix}_ms'] for s in sizes_present]
        ax_bm2.bar(x2 + (i - 0.5) * w2, lats, w2, color=color, alpha=0.75, label=eye_label)

    for j, size in enumerate(sizes_present):
        d = results['iold_per_size'][size]
        if d and d.get('iold_ms') is not None:
            flag_str = ' ⚑' if d['flag'] else ''
            top = max(d['p100_left_ms'], d['p100_right_ms'])
            ax_bm2.text(j, top + 1.5, f'IOLD\n{d["iold_ms"]:+.1f}ms{flag_str}',
                        ha='center', fontsize=9, color='#7d3c98', fontweight='bold')

    ax_bm2.set_xticks(x2)
    ax_bm2.set_xticklabels([f'{s.title()} checks\n({CHECK_SIZE_ARCMIN[s]:.0f} arcmin)'
                             for s in sizes_present], fontsize=11)
    ax_bm2.set_ylabel('P100 latency at Oz (ms)')
    ax_bm2.set_title(f'[{ref_label}] BM2: IOLD per check size')
    ax_bm2.legend()
    ax_bm2.grid(axis='y', alpha=0.3)
    fig_bm2.tight_layout()
    plt.show()

###############################################################################
# ## BM3 — Check-size slope
#
# Per-eye P100 latency slope vs. check size (ms / arcmin). Demyelination
# preferentially delays high-spatial-frequency (small-check) responses, so the
# L−R slope difference amplifies asymmetric demyelination beyond what a
# single-check IOLD captures.
#
# Check-size mapping:
#   large → 1.0 deg = 60 arcmin (ISCEV "large check")
#   small → 0.25 deg = 15 arcmin (ISCEV "small check")
#

results['slope'] = compute_check_size_slope(
    ch_epochs_corr, event_id, PICK_CH, CHECK_SIZE_ARCMIN,
)
s = results['slope']

print("\n--- BM3: Check-size slope ---")
for cond_key, lat_ms in s['per_condition_p100_ms'].items():
    print(f"[BM3] {cond_key}: P100 = {lat_ms:.2f} ms")
if s['slope_left_ms_per_arcmin'] is not None:
    print(f"[BM3] Left eye slope:  {s['slope_left_ms_per_arcmin']:+.4f} ms/arcmin")
if s['slope_right_ms_per_arcmin'] is not None:
    print(f"[BM3] Right eye slope: {s['slope_right_ms_per_arcmin']:+.4f} ms/arcmin")
if s['slope_diff'] is not None:
    print(f"[BM3] Slope diff L−R:  {s['slope_diff']:+.4f} ms/arcmin  "
          f"(positive = left more SF-dependent = left more affected)")
elif not s['per_condition_p100_ms']:
    print("[BM3] Insufficient per-condition P100 detections for slope estimation")
else:
    print("[BM3] Need both eyes detected for slope difference")

# Scatter + regression lines
if s['per_condition_p100_ms']:
    fig_bm3, ax_bm3 = plt.subplots(figsize=(7, 5))
    for eye_prefix, color, eye_label, slope_key in [
        ('left_eye',  '#1f77b4', 'Left Eye',  'slope_left_ms_per_arcmin'),
        ('right_eye', '#d62728', 'Right Eye', 'slope_right_ms_per_arcmin'),
    ]:
        pts = []
        for cond_key, lat_ms in s['per_condition_p100_ms'].items():
            if cond_key.startswith(eye_prefix) and lat_ms is not None:
                size_label = cond_key.split('/')[1]
                if size_label in CHECK_SIZE_ARCMIN:
                    pts.append((CHECK_SIZE_ARCMIN[size_label], lat_ms))
        if not pts:
            continue
        pts.sort()
        xs_p, ys_p = zip(*pts)
        ax_bm3.scatter(xs_p, ys_p, color=color, s=100, zorder=5)

        slope_val = s.get(slope_key)
        if slope_val is not None and len(pts) >= 2:
            intercept = np.mean(ys_p) - slope_val * np.mean(xs_p)
            x_fit = np.linspace(min(xs_p) - 5, max(xs_p) + 5, 80)
            ax_bm3.plot(x_fit, slope_val * x_fit + intercept, color=color, lw=2,
                        label=f'{eye_label}: {slope_val:+.3f} ms/arcmin')
        else:
            ax_bm3.plot(xs_p, ys_p, color=color, lw=1.5, label=eye_label)

    if s['slope_diff'] is not None:
        ax_bm3.annotate(f'Slope diff L−R = {s["slope_diff"]:+.3f} ms/arcmin',
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        ha='left', va='top', fontsize=10, color='#7d3c98',
                        fontweight='bold')
    ax_bm3.set_xlabel('Check size (arcmin)')
    ax_bm3.set_ylabel('P100 latency at Oz (ms)')
    ax_bm3.set_title(f'[{ref_label}] BM3: P100 latency vs check size (slope)')
    ax_bm3.legend()
    ax_bm3.grid(True, alpha=0.3)
    fig_bm3.tight_layout()
    plt.show()

###############################################################################
# ## BM4 — P100 amplitude ratio L/R
#
# Inter-ocular amplitude ratio at P100. Less specific than latency but
# computed for free from the same recordings. |log2(L/R)| > 1 (ratio outside
# ~0.5–2.0) flags attenuated drive on the lower-amplitude side.
#
# Caveat: amplitude is sensitive to electrode contact / pulse artifact /
# subject alertness. Treat amplitude ratios as supportive evidence rather than
# standalone biomarkers until several clean baseline sessions bracket the
# day-to-day variance.
#

results['amplitude'] = compute_amplitude_ratio(p100_left, p100_right)

print("\n--- BM4: Amplitude ratio L/R ---")
if results['amplitude'] is None:
    if p100_left is None or p100_right is None:
        print("[BM4] Cannot compute — P100 not detected in one or both eyes")
    else:
        print("[BM4] Right-eye P100 amplitude is zero; ratio undefined")
else:
    a = results['amplitude']
    status = 'FLAG' if a['flag'] else 'within ±1 log₂'
    print(f"[BM4] P100 amplitude  Left  = {a['amp_left_uv']:.2f} µV")
    print(f"[BM4] P100 amplitude  Right = {a['amp_right_uv']:.2f} µV")
    print(f"[BM4] L/R ratio = {a['ratio']:.2f}  (log₂ = {a['log2_ratio']:+.2f})  [{status}]")

    fig_bm4, ax_bm4 = plt.subplots(figsize=(6, 5))
    ax_bm4.bar([0, 1], [a['amp_left_uv'], a['amp_right_uv']],
               color=['#1f77b4', '#d62728'], alpha=0.75, width=0.45)
    ax_bm4.set_xticks([0, 1])
    ax_bm4.set_xticklabels(['Left Eye', 'Right Eye'], fontsize=12)
    ax_bm4.set_ylabel('P100 amplitude at Oz (µV)')
    flag_marker = '  ⚑ FLAG' if a['flag'] else ''
    ax_bm4.set_title(f'[{ref_label}] BM4: P100 amplitude L/R\n'
                     f'ratio = {a["ratio"]:.2f}  (log₂ = {a["log2_ratio"]:+.2f}){flag_marker}')
    ax_bm4.grid(axis='y', alpha=0.3)
    fig_bm4.tight_layout()
    plt.show()

###############################################################################
# ## BM5 — Bootstrap P100 / IOLD confidence intervals
#
# The 8 ms IOLD threshold is only meaningful relative to the precision of
# the L and R latency estimates. A 7 ms IOLD with ±2 ms CI is clinically
# suspicious; a 7 ms IOLD with ±5 ms CI is noise.
#
# Trial-resamples with replacement, recomputes the trimmed-mean evoked at
# PICK_CH, locates the positive peak in the P100 search window. The IOLD CI
# uses pairwise differences of two independent bootstrap samples.
#

N_BOOT = 1000
BOOT_SEED = 0

boot_left  = bootstrap_p100_latency(
    ch_epochs_corr, event_id, PICK_CH, 'left_eye',
    win_ms=P100_WIN_MS, n_boot=N_BOOT, seed=BOOT_SEED,
)
boot_right = bootstrap_p100_latency(
    ch_epochs_corr, event_id, PICK_CH, 'right_eye',
    win_ms=P100_WIN_MS, n_boot=N_BOOT, seed=BOOT_SEED + 1,
)

print("\n--- BM5: Bootstrap P100 / IOLD confidence intervals ---")

if boot_left is not None and boot_right is not None:
    l_med, l_lo, l_hi = (np.percentile(boot_left,  50),
                         np.percentile(boot_left,  2.5),
                         np.percentile(boot_left,  97.5))
    r_med, r_lo, r_hi = (np.percentile(boot_right, 50),
                         np.percentile(boot_right, 2.5),
                         np.percentile(boot_right, 97.5))
    diffs = boot_left - boot_right
    d_med, d_lo, d_hi = (np.percentile(diffs, 50),
                         np.percentile(diffs, 2.5),
                         np.percentile(diffs, 97.5))
    excludes_zero = (d_lo > 0) or (d_hi < 0)
    excl_8ms      = (d_lo > 8.0) or (d_hi < -8.0)
    print(f"[BM5] Left  P100  median = {l_med:.2f} ms   95% CI = [{l_lo:.2f}, {l_hi:.2f}]  "
          f"(±{(l_hi-l_lo)/2:.2f} ms)")
    print(f"[BM5] Right P100  median = {r_med:.2f} ms   95% CI = [{r_lo:.2f}, {r_hi:.2f}]  "
          f"(±{(r_hi-r_lo)/2:.2f} ms)")
    print(f"[BM5] IOLD (L−R)  median = {d_med:+.2f} ms  95% CI = [{d_lo:+.2f}, {d_hi:+.2f}]")
    print(f"[BM5]   {'CI excludes 0 — significant' if excludes_zero else 'CI includes 0 — not separable'}")
    print(f"[BM5]   {'CI excludes ±8 ms — clinically meaningful' if excl_8ms else 'CI overlaps ±8 ms — borderline'}")

    results['bootstrap'] = {
        'n_boot': int(N_BOOT),
        'win_ms': list(P100_WIN_MS),
        'left_p100_median_ms':   _f(l_med),
        'left_p100_ci_lo_ms':    _f(l_lo),
        'left_p100_ci_hi_ms':    _f(l_hi),
        'right_p100_median_ms':  _f(r_med),
        'right_p100_ci_lo_ms':   _f(r_lo),
        'right_p100_ci_hi_ms':   _f(r_hi),
        'iold_median_ms':        _f(d_med),
        'iold_ci_lo_ms':         _f(d_lo),
        'iold_ci_hi_ms':         _f(d_hi),
        'iold_excludes_zero':    bool(excludes_zero),
        'iold_excludes_8ms':     bool(excl_8ms),
    }

    # Bootstrap latency distribution
    fig_bm5, ax_bm5 = plt.subplots(figsize=(10, 5))
    bins5 = np.arange(P100_WIN_MS[0], P100_WIN_MS[1] + 4, 4)
    ax_bm5.hist(boot_left,  bins=bins5, alpha=0.5, color='#1f77b4',
                label=f'Left  ({l_med:.1f} ms,  95% CI [{l_lo:.1f}, {l_hi:.1f}])')
    ax_bm5.hist(boot_right, bins=bins5, alpha=0.5, color='#d62728',
                label=f'Right ({r_med:.1f} ms,  95% CI [{r_lo:.1f}, {r_hi:.1f}])')
    ax_bm5.axvline(l_med, color='#1f77b4', linestyle='--', alpha=0.8)
    ax_bm5.axvline(r_med, color='#d62728', linestyle='--', alpha=0.8)
    ax_bm5.axvspan(l_lo, l_hi, alpha=0.10, color='#1f77b4')
    ax_bm5.axvspan(r_lo, r_hi, alpha=0.10, color='#d62728')
    flag_str5 = '  ⚑ CI excludes 0' if excludes_zero else '  CI includes 0'
    ax_bm5.set_xlabel('P100 latency (ms)')
    ax_bm5.set_ylabel(f'Bootstrap count (N={N_BOOT})')
    ax_bm5.set_title(f'[{ref_label}] BM5: Bootstrap P100 distribution — {PICK_CH}\n'
                     f'IOLD = {d_med:+.2f} ms  95% CI [{d_lo:+.2f}, {d_hi:+.2f}]{flag_str5}')
    ax_bm5.legend(loc='upper right')
    ax_bm5.grid(True, alpha=0.3)
    fig_bm5.tight_layout()
    plt.show()
else:
    print("[BM5] Insufficient trials in one or both eyes — skipping")
    results['bootstrap'] = None

# =========================================================================
# SECTION 2: Morphological Indicators
#
# BM6 examines the shape of the P100 itself rather than its latency or
# amplitude, targeting waveform distortions that can accompany partial
# demyelination or multifocal lesions.
# =========================================================================
print("\n" + "="*70)
print("SECTION 2 — Morphological")
print("="*70)

###############################################################################
# ## BM6 — W-peak (bifurcated P100)
#
# Partial demyelination or multifocal lesions can split the P100 into two
# distinct peaks (a "W" shape), reflecting asynchronous arrival of fast and
# slow fibre populations. The search window is 80–130 ms; two peaks are flagged
# when each rises > 1 µV above the dip between them.
#

print("\n--- BM6: W-peak (bifurcated P100) ---")

wpeak_results = {}
for eye_name, ev_data in [('Left Eye', evoked_left_corr_avg),
                           ('Right Eye', evoked_right_corr_avg)]:
    if ev_data is None:
        continue
    oz_idx = ev_data.ch_names.index(PICK_CH)
    t_ms   = ev_data.times * 1000
    oz_uv  = ev_data.data[oz_idx] * 1e6

    w_mask  = (t_ms >= 80) & (t_ms <= 130)
    oz_win  = oz_uv[w_mask]
    t_win   = t_ms[w_mask]
    peaks_w, _ = find_peaks(oz_win, prominence=0.5, distance=3)

    flagged = False
    if len(peaks_w) >= 2:
        p1_t, p2_t = t_win[peaks_w[0]], t_win[peaks_w[1]]
        p1_v, p2_v = oz_win[peaks_w[0]], oz_win[peaks_w[1]]
        dip_v = float(np.min(oz_win[peaks_w[0]:peaks_w[1] + 1]))
        if p1_v - dip_v > 1.0 and p2_v - dip_v > 1.0:
            flagged = True
            print(f"[BM6 {eye_name}] FLAG: W-peak detected on {PICK_CH}")
            print(f"  Peak 1: {p1_v:.2f} µV at {p1_t:.1f} ms")
            print(f"  Peak 2: {p2_v:.2f} µV at {p2_t:.1f} ms")
            print(f"  Dip:    {dip_v:.2f} µV  (depths: P1−dip={p1_v-dip_v:.2f}, P2−dip={p2_v-dip_v:.2f})")
            wpeak_results[eye_name] = {'flagged': True, 'p1_ms': _f(p1_t), 'p2_ms': _f(p2_t),
                                       'p1_uv': _f(p1_v), 'p2_uv': _f(p2_v), 'dip_uv': _f(dip_v)}
    if not flagged:
        print(f"[BM6 {eye_name}] Normal single P100 morphology")
        wpeak_results[eye_name] = {'flagged': False}

results['wpeak'] = wpeak_results

# Waveform plot with peak annotations
fig_bm6, axes_bm6 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (eye_name, ev_data) in zip(axes_bm6,
                                    [('Left Eye',  evoked_left_corr_avg),
                                     ('Right Eye', evoked_right_corr_avg)]):
    if ev_data is None:
        continue
    oz_idx = ev_data.ch_names.index(PICK_CH)
    t_ms   = ev_data.times * 1000
    oz_uv  = ev_data.data[oz_idx] * 1e6

    ax.plot(t_ms, oz_uv, color='black', lw=2, label=PICK_CH)
    ax.axvspan(80, 130, alpha=0.07, color='green', label='W-peak search (80–130 ms)')

    # Annotate detected peaks in search window
    w_mask  = (t_ms >= 80) & (t_ms <= 130)
    oz_win  = oz_uv[w_mask]
    t_win   = t_ms[w_mask]
    peaks_w, _ = find_peaks(oz_win, prominence=0.5, distance=3)
    for pi, pk in enumerate(peaks_w[:3]):
        ax.annotate(f'P{pi+1}: {oz_win[pk]:.1f} µV\n@ {t_win[pk]:.0f} ms',
                    xy=(t_win[pk], oz_win[pk]),
                    xytext=(t_win[pk] + 8, oz_win[pk] + 1.5),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
                    fontsize=8, color='green')

    for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
        ax.axvline(ms, color=col, linestyle='--', alpha=0.5, label=lbl)

    wres = wpeak_results.get(eye_name, {})
    flag_title = '  ⚑ W-PEAK FLAGGED' if wres.get('flagged') else ''
    ax.set_title(f'[{ref_label}] BM6: W-peak — {eye_name}{flag_title}')
    ax.set_xlabel('Time (ms)')
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    if ax == axes_bm6[0]:
        ax.set_ylabel('Amplitude (µV)')
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
              fontsize=9, loc='upper right')

fig_bm6.tight_layout()
plt.show()

# =========================================================================
# SECTION 3: Post-chiasmatic / Cortical Biomarkers
#
# BM7–BM12 examine how the VEP distributes across scalp channels, targeting
# post-chiasmatic asymmetries and generator-confirmation checks.
#
# Hemisphere note: due to V1 anatomy (calcarine cortex folded into the
# longitudinal fissure), the P100 dipole projects PARADOXICALLY — right
# hemisphere V1 → left scalp (O1), left hemisphere V1 → right scalp (O2).
# P7/P8 (lateral extrastriate) project more directly ipsilaterally and are
# therefore less ambiguous for cortical-side lateralization.
# =========================================================================
print("\n" + "="*70)
print("SECTION 3 — Post-chiasmatic / Cortical")
print("="*70)

###############################################################################
# ## BM7 — Hemispheric asymmetry (O1 vs O2)
#
# Compares P100 latency and amplitude between left (O1) and right (O2)
# occipital channels. A post-chiasmatic lesion typically produces a "crossed
# asymmetry": the P100 is delayed or attenuated over the ipsilateral scalp
# (paradoxical lateralization) regardless of which eye is stimulated.
#
# **Paradoxical lateralization**: deficit at O1 → RIGHT hemisphere lesion;
# deficit at O2 → LEFT hemisphere lesion.
#

print("\n--- BM7: Hemispheric asymmetry O1/O2 ---")

fig_bm7, axes_bm7 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
hemi_per_eye = {}

for ax, (eye_name, ev_avg) in zip(axes_bm7,
                                   [('Left Eye',  evoked_left_corr_avg),
                                    ('Right Eye', evoked_right_corr_avg)]):
    if ev_avg is None:
        continue

    h = compute_hemi_asymmetry(ev_avg, 'O1', 'O2')
    if h is not None:
        lat_status = 'FLAG' if h['lat_flag'] else f'within ±{IOLD_FLAG_MS:.0f} ms'
        amp_status = 'FLAG' if h['amp_flag'] else 'within ±1 log₂'
        print(f"[BM7 {eye_name}] Latency:   O1 = {h['lat_o1']:.2f} ms,  O2 = {h['lat_o2']:.2f} ms  "
              f"(O1−O2 = {h['lat_diff_ms']:+.2f} ms [{lat_status}])")
        print(f"[BM7 {eye_name}] Amplitude: O1 = {h['amp_o1']:.2f} µV,  O2 = {h['amp_o2']:.2f} µV  "
              f"(O1/O2 = {h['amp_ratio']:.2f} [{amp_status}])")
        hemi_per_eye[eye_name] = h
    else:
        ev_o1 = ev_avg.copy().pick(['O1'])
        ev_o2 = ev_avg.copy().pick(['O2'])
        _, p100_o1, _ = get_pr_vep_latencies(ev_o1)
        _, p100_o2, _ = get_pr_vep_latencies(ev_o2)
        if p100_o1 is not None or p100_o2 is not None:
            detected = 'O1' if p100_o1 is not None else 'O2'
            missing  = 'O2' if p100_o1 is not None else 'O1'
            print(f"[BM7 {eye_name}] FLAG: P100 at {detected} but absent at {missing}")
        else:
            print(f"[BM7 {eye_name}] P100 not detected in either O1 or O2")

    # Waveform
    t_h = ev_avg.times * 1000
    for ch, color in [('O1', '#9467bd'), ('O2', '#e377c2')]:
        if ch in ev_avg.ch_names:
            idx = ev_avg.ch_names.index(ch)
            ax.plot(t_h, ev_avg.data[idx] * 1e6, color=color, lw=2, label=ch)
    for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
        ax.axvline(ms, color=col, linestyle='--', alpha=0.6,
                   label=lbl if ax == axes_bm7[0] else '')
    ax.set_title(f'[{ref_label}] BM7: Hemispheric asymmetry — {eye_name}')
    ax.set_xlabel('Time (ms)')
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    if ax == axes_bm7[0]:
        ax.set_ylabel('Amplitude (µV)')
    ax.legend(loc='upper right')

fig_bm7.tight_layout()
plt.show()

results['hemi_o1o2'] = dict(hemi_per_eye)

###############################################################################
# ## BM8 — Inter-ocular Δ-asymmetry (O1 vs O2)
#
# Per-eye O1/O2 differences conflate the lesion signal with stable anatomical
# and electrode-stationary asymmetries (skull thickness, calcarine fold, etc.).
# Subtracting one eye's asymmetry from the other's cancels those stationary
# contributions:
#
#   Δlat  = (O1−O2)|left_eye − (O1−O2)|right_eye
#   Δlog₂ = log₂(O1/O2)|left_eye − log₂(O1/O2)|right_eye
#
# ≈ 0 → purely anatomical / electrode-stationary.
# Large value → eye-dependent skew: the asymmetry depends on which eye drives
# cortex, implying a pathway signal rather than a scalp constant.
#

results['hemi_delta_o1o2'] = compute_hemi_delta_asymmetry(
    hemi_per_eye.get('Left Eye'), hemi_per_eye.get('Right Eye'), 'O1', 'O2',
)

print("\n--- BM8: Inter-ocular Δ-asymmetry O1/O2 ---")
if results['hemi_delta_o1o2'] is None:
    print("[BM8] Need P100 at both O1 and O2 in both eyes — skipping")
else:
    d8 = results['hemi_delta_o1o2']
    print(f"[BM8] (O1−O2) latency:  L-eye = {d8['lat_asym_left']:+.2f} ms,  "
          f"R-eye = {d8['lat_asym_right']:+.2f} ms")
    print(f"[BM8] Δlat  = {d8['d_lat']:+.2f} ms  "
          f"(≈0 ⇒ anatomical; large ⇒ eye-dependent)")
    print(f"[BM8] log₂(O1/O2):      L-eye = {d8['log2_asym_left']:+.2f},  "
          f"R-eye = {d8['log2_asym_right']:+.2f}")
    print(f"[BM8] Δlog₂ = {d8['d_log2']:+.2f}  "
          f"(≈0 ⇒ anatomical; large ⇒ eye-dependent)")

    fig_bm8, axes_bm8 = plt.subplots(1, 2, figsize=(10, 5))

    # Latency asymmetry per eye drive
    ax = axes_bm8[0]
    lat_vals = [d8['lat_asym_left'] or 0, d8['lat_asym_right'] or 0]
    ax.bar(['Left Eye\ndrive', 'Right Eye\ndrive'], lat_vals,
           color=['#1f77b4', '#d62728'], alpha=0.75)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('O1 − O2 latency asymmetry (ms)')
    ax.set_title(f'Δlat = {d8["d_lat"]:+.2f} ms')
    ax.grid(axis='y', alpha=0.3)

    # Amplitude asymmetry per eye drive
    ax = axes_bm8[1]
    log2_vals = [d8['log2_asym_left'] or 0, d8['log2_asym_right'] or 0]
    ax.bar(['Left Eye\ndrive', 'Right Eye\ndrive'], log2_vals,
           color=['#1f77b4', '#d62728'], alpha=0.75)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('log₂(O1/O2) amplitude asymmetry')
    ax.set_title(f'Δlog₂ = {d8["d_log2"]:+.2f}')
    ax.grid(axis='y', alpha=0.3)

    fig_bm8.suptitle(f'[{ref_label}] BM8: Inter-ocular Δ-asymmetry O1/O2\n'
                     '≈0 bars of equal height → stationary anatomy;  '
                     'unequal → eye-dependent (pathway signal)',
                     fontsize=10)
    fig_bm8.tight_layout()
    plt.show()

###############################################################################
# ## BM9 — Lateral extrastriate P100 (P7/P8) + Oz→lateral propagation
#
# P7/P8 pick up a P100 from lateral extrastriate cortex (V2/V3/MT), typically
# delayed 5–15 ms after Oz (intracortical V1→extrastriate propagation) and
# lower amplitude. Two reads:
#
# - P100 detection at P7/P8 confirms an extrastriate response exists at all.
# - Oz→lateral propagation latency should be a small positive number (0–25 ms).
#   Abnormally long or reversed sign suggests intracortical / extrastriate
#   conduction issues distinct from optic-nerve demyelination.
#

print("\n--- BM9: Lateral extrastriate P7/P8 + Oz→lateral propagation ---")

lateral_per_eye = {}
results['lateral_p7p8'] = {}

for eye_name, ev_avg, oz_p100 in [
    ('Left Eye',  evoked_left_corr_avg,  p100_left),
    ('Right Eye', evoked_right_corr_avg, p100_right),
]:
    if ev_avg is None:
        continue
    per_eye = {'oz': oz_p100, 'p7': None, 'p8': None}
    per_eye_results = {}
    for ch in LATERAL_CHANNELS:
        if ch not in ev_avg.ch_names:
            continue
        _, p100_lat, _ = get_pr_vep_latencies(ev_avg.copy().pick([ch]))
        per_eye[ch.lower()] = p100_lat
        if p100_lat is None:
            print(f"[BM9 {eye_name}] {ch}: P100 not detected")
            per_eye_results[ch] = None
            continue
        lat_ms_v = p100_lat['latency'] * 1000.0
        amp_uv_v = p100_lat['amplitude'] * 1e6
        print(f"[BM9 {eye_name}] {ch}: P100 = {amp_uv_v:+.2f} µV at {lat_ms_v:.2f} ms")
        ch_entry = {'lat_ms': _f(lat_ms_v), 'amp_uv': _f(amp_uv_v), 'propagation_ms': None}
        if oz_p100 is not None:
            prop_ms = lat_ms_v - oz_p100['latency'] * 1000.0
            in_range = -2 <= prop_ms <= 25
            status = 'normal' if in_range else 'OUT OF RANGE'
            print(f"[BM9 {eye_name}] {ch}−Oz propagation: {prop_ms:+.2f} ms  "
                  f"[{status}, expected −2 to +25 ms]")
            ch_entry['propagation_ms'] = _f(prop_ms)
            ch_entry['propagation_in_range'] = bool(in_range)
        per_eye_results[ch] = ch_entry
    lateral_per_eye[eye_name] = per_eye
    results['lateral_p7p8'][eye_name] = per_eye_results

# Waveform: Oz, P7, P8 per eye
fig_bm9, axes_bm9 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
lat_ch_colors = {'Oz': 'black', 'P7': '#1f77b4', 'P8': '#d62728'}
lat_ch_lw     = {'Oz': 2.5,    'P7': 1.8,        'P8': 1.8}

for ax, (eye_name, ev_avg) in zip(axes_bm9,
                                   [('Left Eye',  evoked_left_corr_avg),
                                    ('Right Eye', evoked_right_corr_avg)]):
    if ev_avg is None:
        continue
    t_lat = ev_avg.times * 1000
    for ch in ['Oz', 'P7', 'P8']:
        if ch not in ev_avg.ch_names:
            continue
        idx = ev_avg.ch_names.index(ch)
        ax.plot(t_lat, ev_avg.data[idx] * 1e6,
                color=lat_ch_colors[ch], lw=lat_ch_lw[ch], label=ch)

    # Annotate propagation delays
    eye_lat = results['lateral_p7p8'].get(eye_name, {})
    for ch in ['P7', 'P8']:
        ch_entry = eye_lat.get(ch)
        if ch_entry and ch_entry.get('propagation_ms') is not None:
            lat_ms_v = ch_entry['lat_ms']
            prop_ms  = ch_entry['propagation_ms']
            ok_str   = '✓' if ch_entry.get('propagation_in_range', True) else '⚑'
            # find the y value at this latency for annotation placement
            t_idx_ann = int(np.argmin(np.abs(t_lat - lat_ms_v)))
            y_ann = ev_avg.data[ev_avg.ch_names.index(ch), t_idx_ann] * 1e6
            ax.annotate(f'{ch}: +{prop_ms:.0f}ms {ok_str}',
                        xy=(lat_ms_v, y_ann),
                        xytext=(lat_ms_v + 15, y_ann + 1.5),
                        arrowprops=dict(arrowstyle='->', color=lat_ch_colors[ch], lw=1),
                        fontsize=8, color=lat_ch_colors[ch])

    for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
        ax.axvline(ms, color=col, linestyle='--', alpha=0.5,
                   label=lbl if ax == axes_bm9[0] else '')
    ax.set_title(f'[{ref_label}] BM9: Lateral extrastriate — {eye_name}')
    ax.set_xlabel('Time (ms)')
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    if ax == axes_bm9[0]:
        ax.set_ylabel('Amplitude (µV)')
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(dict(zip(lbls, handles)).values(), dict(zip(lbls, handles)).keys(),
              fontsize=9, loc='upper right')

fig_bm9.tight_layout()
plt.show()

###############################################################################
# ## BM10 — Lateral extrastriate asymmetry (P7 vs P8) + inter-ocular contrast
#
# Same logic as BM7/BM8 but on P7/P8. Critical advantage: extrastriate
# generators sit on the lateral cortical surface, so paradoxical lateralization
# is much weaker here than at O1/O2. P7/P8 hemispheric asymmetry is therefore
# a more direct readout of cortical-side asymmetry — useful for detecting
# unilateral lateral-occipital / parieto-occipital involvement.
#

print("\n--- BM10: Lateral asymmetry P7/P8 + inter-ocular contrast ---")

p7p8_asym_per_eye = {}
for eye_name, ev_avg in [('Left Eye',  evoked_left_corr_avg),
                          ('Right Eye', evoked_right_corr_avg)]:
    if ev_avg is None:
        continue
    h = compute_hemi_asymmetry(ev_avg, 'P7', 'P8')
    if h is None:
        print(f"[BM10 {eye_name}] P100 not detected at both P7 and P8 — skipping")
        continue
    lat_status = 'FLAG' if h['lat_flag'] else f'within ±{IOLD_FLAG_MS:.0f} ms'
    amp_status = 'FLAG' if h['amp_flag'] else 'within ±1 log₂'
    print(f"[BM10 {eye_name}] P7 = {h['lat_p7']:.2f} ms,  P8 = {h['lat_p8']:.2f} ms  "
          f"(P7−P8 = {h['lat_diff_ms']:+.2f} ms [{lat_status}])")
    print(f"[BM10 {eye_name}] P7 = {h['amp_p7']:.2f} µV,  P8 = {h['amp_p8']:.2f} µV  "
          f"(P7/P8 = {h['amp_ratio']:.2f} [{amp_status}])")
    p7p8_asym_per_eye[eye_name] = h

results['hemi_p7p8'] = dict(p7p8_asym_per_eye)
results['hemi_delta_p7p8'] = compute_hemi_delta_asymmetry(
    p7p8_asym_per_eye.get('Left Eye'), p7p8_asym_per_eye.get('Right Eye'), 'P7', 'P8',
)

if results['hemi_delta_p7p8'] is not None:
    d10 = results['hemi_delta_p7p8']
    print(f"[BM10] Δlat  (P7−P8): {d10['d_lat']:+.2f} ms   "
          f"(≈0 ⇒ stationary; large ⇒ eye-dependent)")
    print(f"[BM10] Δlog₂(P7/P8):  {d10['d_log2']:+.2f}     "
          f"(≈0 ⇒ stationary; large ⇒ eye-dependent)")

    fig_bm10, axes_bm10 = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes_bm10[0]
    ax.bar(['Left Eye\ndrive', 'Right Eye\ndrive'],
           [d10['lat_asym_left'] or 0, d10['lat_asym_right'] or 0],
           color=['#1f77b4', '#d62728'], alpha=0.75)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('P7 − P8 latency asymmetry (ms)')
    ax.set_title(f'Δlat = {d10["d_lat"]:+.2f} ms')
    ax.grid(axis='y', alpha=0.3)

    ax = axes_bm10[1]
    ax.bar(['Left Eye\ndrive', 'Right Eye\ndrive'],
           [d10['log2_asym_left'] or 0, d10['log2_asym_right'] or 0],
           color=['#1f77b4', '#d62728'], alpha=0.75)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('log₂(P7/P8) amplitude asymmetry')
    ax.set_title(f'Δlog₂ = {d10["d_log2"]:+.2f}')
    ax.grid(axis='y', alpha=0.3)

    fig_bm10.suptitle(f'[{ref_label}] BM10: Inter-ocular Δ-asymmetry P7/P8\n'
                      '≈0 → stationary anatomy;  unequal → eye-dependent (cortical-side signal)',
                      fontsize=10)
    fig_bm10.tight_layout()
    plt.show()

###############################################################################
# ## BM11 — Combined lateral hemisphere composites
#
# Per-hemisphere composites: L-hemi = (O1+P7)/2, R-hemi = (O2+P8)/2.
# Three advantages over O1/O2 alone:
#   - SNR ≈ √2 better (averaging two channels per hemisphere).
#   - Dilutes the contribution of any single bad electrode contact.
#   - Mixes the paradoxically-projected V1 component (O1/O2) with the
#     directly-projected extrastriate component (P7/P8), giving a composite
#     that is less ambiguous about cortical side than O1/O2 alone.
#

print("\n--- BM11: Combined lateral hemisphere composites ---")

fig_bm11, axes_bm11 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
composite_p100 = {}

for ax, (eye_name, ev_avg, color) in zip(axes_bm11, [
    ('Left Eye',  evoked_left_corr_avg,  'blue'),
    ('Right Eye', evoked_right_corr_avg, 'red'),
]):
    if ev_avg is None:
        continue
    if not all(ch in ev_avg.ch_names for ch in ('O1', 'O2', 'P7', 'P8')):
        continue
    idx_o1 = ev_avg.ch_names.index('O1')
    idx_o2 = ev_avg.ch_names.index('O2')
    idx_p7 = ev_avg.ch_names.index('P7')
    idx_p8 = ev_avg.ch_names.index('P8')

    l_hemi = (ev_avg.data[idx_o1] + ev_avg.data[idx_p7]) / 2.0
    r_hemi = (ev_avg.data[idx_o2] + ev_avg.data[idx_p8]) / 2.0
    t_comp = ev_avg.times * 1000

    ax.plot(t_comp, l_hemi * 1e6, color='#1f77b4', lw=2,
            label='L-hemi (O1+P7)/2')
    ax.plot(t_comp, r_hemi * 1e6, color='#d62728', lw=2,
            label='R-hemi (O2+P8)/2')

    win_mask = (t_comp >= P100_WIN_MS[0]) & (t_comp <= P100_WIN_MS[1])
    win_t = t_comp[win_mask]
    l_lat = win_t[np.argmax(l_hemi[win_mask])]
    r_lat = win_t[np.argmax(r_hemi[win_mask])]
    l_amp = float(l_hemi[win_mask].max() * 1e6)
    r_amp = float(r_hemi[win_mask].max() * 1e6)
    composite_p100[eye_name] = {'l_lat': l_lat, 'r_lat': r_lat,
                                  'l_amp': l_amp, 'r_amp': r_amp}
    print(f"[BM11 {eye_name}] L-hemi P100 = {l_amp:+.2f} µV @ {l_lat:.1f} ms  |  "
          f"R-hemi P100 = {r_amp:+.2f} µV @ {r_lat:.1f} ms")

    for ms, col, lbl in zip(LANDMARK_MS, LANDMARK_COLORS, LANDMARK_LABELS):
        ax.axvline(ms, color=col, linestyle='--', alpha=0.6,
                   label=lbl if ax == axes_bm11[0] else '')
    ax.set_title(f'[{ref_label}] BM11: Composite hemispheres — {eye_name}')
    ax.set_xlabel('Time (ms)')
    if ax == axes_bm11[0]:
        ax.set_ylabel('Amplitude (µV)')
    ax.axhline(0, color='black', alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

fig_bm11.tight_layout()
plt.show()

results['composite'] = {eye: {k: _f(v) for k, v in d.items()}
                         for eye, d in composite_p100.items()}
results['composite_delta'] = None
if 'Left Eye' in composite_p100 and 'Right Eye' in composite_p100:
    Lc = composite_p100['Left Eye']
    Rc = composite_p100['Right Eye']
    d_lat_comp  = (Lc['l_lat'] - Lc['r_lat']) - (Rc['l_lat'] - Rc['r_lat'])
    log2_L = np.log2(Lc['l_amp'] / Lc['r_amp']) if Lc['r_amp'] > 0 else float('nan')
    log2_R = np.log2(Rc['l_amp'] / Rc['r_amp']) if Rc['r_amp'] > 0 else float('nan')
    d_log2_comp = log2_L - log2_R
    print(f"[BM11] Composite Δlat  = {d_lat_comp:+.2f} ms   (eye-dependent hemispheric latency skew)")
    print(f"[BM11] Composite Δlog₂ = {d_log2_comp:+.2f}     (eye-dependent hemispheric amplitude skew)")
    results['composite_delta'] = {'d_lat': _f(d_lat_comp), 'd_log2': _f(d_log2_comp)}

###############################################################################
# ## BM12 — Topology QC: Pz gradient + Fz Halliday polarity inversion
#
# Halliday's frontal polarity-inversion check: a genuine V1-generated P100
# produces a *negative* deflection at Fz at the same latency, because the
# posterior-pointing dipole projects with inverted polarity to the frontal
# scalp. This is a strong generator-confirmation that artifact, EMG, or
# alpha contamination cannot mimic. Pz should additionally show a smaller
# positive P100 (gradient: Oz > Pz, Fz < 0).
#
# Only meaningful under linked-mastoid: Fz is zero by construction under
# the Oz−Fz reference scheme.
#

results['topology'] = None
if 'linked' in ref_label.lower() or 'mastoid' in ref_label.lower():
    print("\n--- BM12: Topology QC (Pz gradient + Fz Halliday inversion) ---")

    topology_results = {}
    for eye_name, ev_avg, oz_p100 in [
        ('Left Eye',  evoked_left_corr_avg,  p100_left),
        ('Right Eye', evoked_right_corr_avg, p100_right),
    ]:
        if ev_avg is None or oz_p100 is None:
            print(f"[BM12 {eye_name}] Oz P100 not detected — skipping")
            continue

        oz_lat_s  = oz_p100['latency']
        oz_amp_uv = oz_p100['amplitude'] * 1e6
        t_idx = int(round((oz_lat_s - ev_avg.tmin) * ev_avg.info['sfreq']))
        t_idx = max(0, min(t_idx, ev_avg.data.shape[1] - 1))

        entry = {'oz_amp_uv': _f(oz_amp_uv), 'pz_amp_uv': None, 'fz_amp_uv': None,
                 'pz_gradient_ok': None, 'fz_inversion_ok': None}

        if 'Pz' in ev_avg.ch_names:
            pz_amp = ev_avg.data[ev_avg.ch_names.index('Pz'), t_idx] * 1e6
            pz_ok  = 0 < pz_amp < oz_amp_uv
            pz_str = 'OK (positive, < Oz)' if pz_ok else 'FLAG: gradient broken'
            print(f"[BM12 {eye_name}] Pz @ {oz_lat_s*1000:.1f} ms: {pz_amp:+.2f} µV  "
                  f"(Oz = {oz_amp_uv:+.2f} µV)  [{pz_str}]")
            entry['pz_amp_uv']       = _f(pz_amp)
            entry['pz_gradient_ok']  = bool(pz_ok)

        if 'Fz' in ev_avg.ch_names:
            fz_amp = ev_avg.data[ev_avg.ch_names.index('Fz'), t_idx] * 1e6
            if abs(fz_amp) < 0.2:
                print(f"[BM12 {eye_name}] Fz: {fz_amp:+.2f} µV  "
                      f"[INACTIVE — Fz is the reference]")
                entry['fz_amp_uv']       = _f(fz_amp)
                entry['fz_inversion_ok'] = None
            else:
                fz_ok  = fz_amp < 0
                fz_str = ('OK (inverted ⇒ V1 generator confirmed)'
                          if fz_ok else 'FLAG: same-sign as Oz')
                print(f"[BM12 {eye_name}] Fz @ {oz_lat_s*1000:.1f} ms: {fz_amp:+.2f} µV  "
                      f"(Oz = {oz_amp_uv:+.2f} µV)  [{fz_str}]")
                entry['fz_amp_uv']       = _f(fz_amp)
                entry['fz_inversion_ok'] = bool(fz_ok)

        topology_results[eye_name] = entry

    results['topology'] = topology_results

    if topology_results:
        fig_bm12, ax_bm12 = plt.subplots(figsize=(8, 5))
        eyes_t = list(topology_results.keys())
        x_t    = np.arange(len(eyes_t))
        w_t    = 0.22
        ch_order  = [('oz_amp_uv', 'Oz',   'black'),
                     ('pz_amp_uv', 'Pz',   '#2ca02c'),
                     ('fz_amp_uv', 'Fz',   '#ff7f0e')]
        for i, (key, label, color) in enumerate(ch_order):
            vals = [topology_results.get(eye, {}).get(key) or 0 for eye in eyes_t]
            ax_bm12.bar(x_t + (i - 1) * w_t, vals, w_t,
                        color=color, alpha=0.75, label=label)
        ax_bm12.axhline(0, color='black', lw=0.8)
        ax_bm12.set_xticks(x_t)
        ax_bm12.set_xticklabels(eyes_t)
        ax_bm12.set_ylabel('Amplitude at Oz-P100 latency (µV)')
        ax_bm12.set_title(f'[{ref_label}] BM12: Topology QC\n'
                          'Oz > 0, Pz > 0 (gradient), Fz < 0 (Halliday inversion)')
        ax_bm12.legend()
        ax_bm12.grid(axis='y', alpha=0.3)
        fig_bm12.tight_layout()
        plt.show()
else:
    print("\n[BM12] Topology QC skipped — requires Linked Mastoid reference "
          "(run again with REF_SCHEME = 'Linked Mastoid M1+M2')")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "="*70)
print("BIOMARKER SUMMARY")
print("="*70)

_summary = []

def _row(bm_id, name, flag, value_str):
    _summary.append({'id': bm_id, 'name': name, 'flag': flag, 'value': value_str})

def _fmt_delta(d, lat_key='d_lat', log2_key='d_log2'):
    if not d or d.get(lat_key) is None:
        return 'N/A'
    log2_v = d.get(log2_key)
    return f'Δlat={d[lat_key]:+.2f} ms,  Δlog₂=' + (f'{log2_v:+.2f}' if log2_v is not None else 'N/A')

iold = results.get('iold') or {}
_row('BM1',  'IOLD pooled (Oz)',
     iold.get('flag'),
     f'{iold["iold_ms"]:+.1f} ms' if iold.get('iold_ms') is not None else 'N/A')

for size, d_sz in (results.get('iold_per_size') or {}).items():
    _row(f'BM2/{size}', f'IOLD {size} checks (Oz)',
         d_sz['flag'] if d_sz else None,
         f'{d_sz["iold_ms"]:+.1f} ms' if d_sz and d_sz.get('iold_ms') is not None else 'N/A')

_s = results.get('slope') or {}
_row('BM3',  'Check-size slope diff (Oz)',
     None,
     f'{_s["slope_diff"]:+.4f} ms/arcmin' if _s.get('slope_diff') is not None else 'N/A')

_a = results.get('amplitude') or {}
_row('BM4',  'Amplitude ratio L/R (Oz)',
     _a.get('flag'),
     (f'{_a["ratio"]:.2f}  (log₂={_a["log2_ratio"]:+.2f})'
      if _a.get('ratio') is not None else 'N/A'))

_b = results.get('bootstrap') or {}
_row('BM5',  'Bootstrap IOLD 95% CI',
     (not _b.get('iold_excludes_zero')) if _b else None,
     (f'[{_b["iold_ci_lo_ms"]:+.1f}, {_b["iold_ci_hi_ms"]:+.1f}] ms'
      if _b and _b.get('iold_ci_lo_ms') is not None else 'N/A'))

_w = results.get('wpeak') or {}
any_wpeak = any(v.get('flagged') for v in _w.values())
_row('BM6',  'W-peak (bifurcated P100)',
     any_wpeak if _w else None,
     'Flagged in: ' + ', '.join(e for e, v in _w.items() if v.get('flagged')) if any_wpeak else 'None detected')

for eye_name, h7 in (results.get('hemi_o1o2') or {}).items():
    _row(f'BM7/{eye_name[:1]}', f'Hemi asym O1/O2 — {eye_name}',
         h7.get('lat_flag') or h7.get('amp_flag'),
         (f'lat diff={h7["lat_diff_ms"]:+.1f} ms,  amp ratio={h7["amp_ratio"]:.2f}'
          if h7 else 'N/A'))

_row('BM8',  'Δ-asymmetry O1/O2 (eye-dependent?)',
     None, _fmt_delta(results.get('hemi_delta_o1o2') or {}))

for eye_name, h_lat in results.get('lateral_p7p8', {}).items():
    vals_9 = [f'{ch}: {v["propagation_ms"]:+.0f}ms' if v and v.get('propagation_ms') is not None else f'{ch}: N/A'
              for ch, v in h_lat.items()]
    any_oor = any(v and v.get('propagation_in_range') is False for v in h_lat.values())
    _row(f'BM9/{eye_name[:1]}', f'Lateral P7/P8 propagation — {eye_name}',
         any_oor if h_lat else None,
         '  '.join(vals_9))

_row('BM10', 'Δ-asymmetry P7/P8 (eye-dependent?)',
     None, _fmt_delta(results.get('hemi_delta_p7p8') or {}))

_row('BM11', 'Composite hemi Δ',
     None, _fmt_delta(results.get('composite_delta') or {}))

_topo = results.get('topology') or {}
topo_flags = [e for e, entry in _topo.items()
              if entry and (entry.get('pz_gradient_ok') is False
                            or entry.get('fz_inversion_ok') is False)]
_row('BM12', 'Topology QC (Pz gradient + Fz inversion)',
     bool(topo_flags) if _topo else None,
     'FLAG: ' + ', '.join(topo_flags) if topo_flags else ('OK' if _topo else 'Skipped (not LM ref)'))

print(f"\n  {'BM':<10} {'Name':<48} {'Status':<12} Value")
print("  " + "-"*90)
for row in _summary:
    flag = row['flag']
    status_str = '⚑ FLAGGED' if flag is True else ('N/A' if flag is None else 'OK')
    print(f"  {row['id']:<10} {row['name']:<48} {status_str:<12} {row['value']}")

###############################################################################
# ## Summary dashboard figure
#

n_rows = len(_summary)
fig_sum, ax_sum = plt.subplots(figsize=(13, max(5, n_rows * 0.42 + 1.0)))

bar_colors = ['#e74c3c' if r['flag'] is True
              else ('#95a5a6' if r['flag'] is None
              else '#27ae60')
              for r in _summary]

ax_sum.barh(range(n_rows), [1] * n_rows, color=bar_colors, alpha=0.85, height=0.7)

for i, row in enumerate(reversed(_summary)):
    y = i
    flag = row['flag']
    status_str = '⚑ FLAGGED' if flag is True else ('N/A' if flag is None else 'OK')
    # biomarker label on the left
    ax_sum.text(-0.02, y, f"{row['id']}: {row['name']}",
                ha='right', va='center', fontsize=9)
    # status text inside bar
    ax_sum.text(0.5, y, status_str,
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
    # value on the right
    ax_sum.text(1.02, y, row['value'],
                ha='left', va='center', fontsize=8, color='#444444')

ax_sum.set_xlim(-4.5, 4.5)
ax_sum.set_ylim(-0.5, n_rows - 0.5)
ax_sum.axis('off')
ax_sum.set_title(f'[{ref_label}] Biomarker Summary Dashboard\n'
                 'Green = OK   Red = Flagged   Grey = N/A or not computed',
                 fontsize=11, pad=12)
fig_sum.tight_layout()
plt.show()

###############################################################################
# ## Persist biomarkers to disk
#
# Read-merge-write into ``biomarkers.json``: this run's REF_SCHEME slot is
# updated; other ref schemes already persisted from prior runs are preserved.
# Run the notebook once per REF_SCHEME to populate both keys.
#

biomarker_path = recording_dir / 'biomarkers.json'
if biomarker_path.exists():
    with open(biomarker_path, 'r', encoding='utf-8') as f:
        biomarker_payload = json.load(f)
    biomarker_payload.setdefault('references', {})
else:
    biomarker_payload = {'references': {}}

biomarker_payload.update({
    'subject_id':              SUBJECT_ID,
    'session_nb':              SESSION_NB,
    'device_name':             DEVICE_NAME,
    'experiment':              EXPERIMENT,
    'site':                    SITE,
    'display':                 DISPLAY,
    'montage':                 MONTAGE,
    'analysis_timestamp':      datetime.datetime.now().isoformat(timespec='seconds'),
    'link_panel_lag_ms':       link_panel_lag * 1000.0,
    'link_panel_lag_err_ms':   link_panel_lag_err * 1000.0,
    'pc_lag_ms_mean':          float(pc_lag_s.mean() * 1000.0),
    'pc_lag_ms_std':           float(pc_lag_s.std()  * 1000.0),
    'n_recordings':            len(recording_files),
})
biomarker_payload['references'][ref_label] = results

with open(biomarker_path, 'w', encoding='utf-8') as f:
    json.dump(biomarker_payload, f, indent=2, ensure_ascii=False)
print(f"\n[persist] biomarkers written to: {biomarker_path}")
print(f"[persist] reference schemes now persisted: {list(biomarker_payload['references'])}")
