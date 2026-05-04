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

# Pattern Reversal VEP: Longitudinal Analysis

Aggregates per-session biomarker JSON files written by
``01r__pattern_reversal_viz.py`` into a single per-subject longitudinal
trend view. Each prior session must have been analysed by 01r at least
once (so its ``biomarkers.json`` exists in the recording directory);
this script does **not** recompute biomarkers from raw EEG — it reads
the persisted JSON and plots trends across sessions.

Why this split: the per-session analysis is expensive (load EEG,
filter, epoch, bootstrap), but the trend view is cheap (read JSON,
plot). Persisting biomarkers per session means new longitudinal points
are added in seconds rather than minutes, and individual sessions can
be re-analysed without invalidating the rest of the series.

Outputs:

- A summary DataFrame indexed by session, with one row per (session,
  reference scheme) pair.
- A trend figure showing IOLD, per-eye P100 latency, check-size slope
  difference, and the O1/O2 and P7/P8 Δ-asymmetry contrasts as a
  function of session.
- Bootstrap CI bands around the IOLD trend so a real shift is visually
  separable from session noise.

"""

###############################################################################
# ## Setup
#
#

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

from eegnb import get_recording_dir

# sphinx_gallery_thumbnail_number = 1

###############################################################################
# ## Configure subject / experiment selectors
#
# Change these placeholders to point at the subject + experiment + site whose
# sessions you want to plot. The script will glob every session directory it
# finds under that path and load each session's ``biomarkers.json``.
#

# --- CHANGE THESE PLACEHOLDERS TO POINT AT YOUR OWN SUBJECT ---------------
SUBJECT_ID = 0       # placeholder — your subject number
DEVICE_NAME = 'cyton'
EXPERIMENT = 'visual-PRVEP'
DISPLAY = 'quest-2_120Hz'   # display tag in the site path
MONTAGE = 'cap'              # 'cap' or 'mark-iv'
SITE = f'{DISPLAY}_{MONTAGE}'
# --------------------------------------------------------------------------

# Reference scheme to plot as the primary line (the other is overlaid lighter).
PRIMARY_REF = 'Linked Mastoid M1+M2'    # alternative: 'Fz (ISCEV)'

###############################################################################
# ## Discover session JSON files
#
# Each session's ``biomarkers.json`` lives in the per-session recording dir.
# We use SESSION_NB = 0 just to anchor the path (we then walk one level up to
# find every sibling session directory).
#

anchor_dir = get_recording_dir(DEVICE_NAME, EXPERIMENT, SUBJECT_ID, 0, site=SITE)
subject_dir = anchor_dir.parent  # .../subject{SID}/
print(f"[scan] subject dir: {subject_dir}")

session_dirs = sorted(p for p in subject_dir.glob('session*') if p.is_dir())
print(f"[scan] found {len(session_dirs)} session directories")

session_jsons = []
for sd in session_dirs:
    bj = sd / 'biomarkers.json'
    if bj.exists():
        session_jsons.append(bj)
    else:
        print(f"[skip] {sd.name}: no biomarkers.json (run 01r on it first)")

if not session_jsons:
    raise RuntimeError(
        f"No biomarkers.json files found under {subject_dir}. "
        "Run 01r__pattern_reversal_viz.py against each session first to "
        "generate the per-session biomarker payloads."
    )

print(f"[scan] {len(session_jsons)} session(s) have biomarkers.json")

###############################################################################
# ## Load and flatten into a DataFrame
#
# One row per (session, reference scheme), with biomarker fields flattened
# into named columns. Fields that aren't present in a session land as NaN.
#


def flatten_session(payload):
    """One dict per ref scheme in a session payload, with metadata + biomarkers."""
    rows = []
    meta = {
        'subject_id': payload.get('subject_id'),
        'session_nb': payload.get('session_nb'),
        'analysis_timestamp': payload.get('analysis_timestamp'),
        'site': payload.get('site'),
        'montage': payload.get('montage'),
        'n_trials_kept_overall': payload.get('n_trials_kept'),
        'pc_lag_ms_mean': payload.get('pc_lag_ms_mean'),
    }
    for ref_label, ref_block in (payload.get('references') or {}).items():
        if ref_block is None:
            continue
        row = dict(meta)
        row['ref'] = ref_label
        row['n_trials_kept'] = ref_block.get('n_trials_kept')

        iold = ref_block.get('iold') or {}
        row['p100_left_ms']  = iold.get('p100_left_ms')
        row['p100_right_ms'] = iold.get('p100_right_ms')
        row['p100_left_uv']  = iold.get('p100_left_uv')
        row['p100_right_uv'] = iold.get('p100_right_uv')
        row['iold_ms']       = iold.get('iold_ms')
        row['iold_flag']     = iold.get('flag')

        slope = ref_block.get('slope') or {}
        row['slope_left']  = slope.get('slope_left_ms_per_arcmin')
        row['slope_right'] = slope.get('slope_right_ms_per_arcmin')
        row['slope_diff']  = slope.get('slope_diff')

        amp = ref_block.get('amplitude') or {}
        row['amp_ratio']   = amp.get('ratio')
        row['amp_log2']    = amp.get('log2_ratio')

        boot = ref_block.get('bootstrap') or {}
        row['boot_iold_median'] = boot.get('iold_median_ms')
        row['boot_iold_lo']     = boot.get('iold_ci_lo_ms')
        row['boot_iold_hi']     = boot.get('iold_ci_hi_ms')
        row['boot_left_lo']     = boot.get('left_p100_ci_lo_ms')
        row['boot_left_hi']     = boot.get('left_p100_ci_hi_ms')
        row['boot_right_lo']    = boot.get('right_p100_ci_lo_ms')
        row['boot_right_hi']    = boot.get('right_p100_ci_hi_ms')

        hd_o1o2 = ref_block.get('hemi_delta_o1o2') or {}
        row['delta_lat_o1o2']  = hd_o1o2.get('d_lat')
        row['delta_log2_o1o2'] = hd_o1o2.get('d_log2')

        hd_p7p8 = ref_block.get('hemi_delta_p7p8') or {}
        row['delta_lat_p7p8']  = hd_p7p8.get('d_lat')
        row['delta_log2_p7p8'] = hd_p7p8.get('d_log2')

        comp_d = ref_block.get('composite_delta') or {}
        row['composite_d_lat']  = comp_d.get('d_lat')
        row['composite_d_log2'] = comp_d.get('d_log2')

        rows.append(row)
    return rows


all_rows = []
for jp in session_jsons:
    with open(jp, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    all_rows.extend(flatten_session(payload))

df = pd.DataFrame(all_rows).sort_values(['session_nb', 'ref']).reset_index(drop=True)
print(f"\n[load] longitudinal dataframe: {len(df)} rows, {len(df['session_nb'].unique())} sessions, "
      f"{len(df['ref'].unique())} reference scheme(s)")
print(df[['session_nb', 'ref', 'n_trials_kept', 'iold_ms', 'slope_diff',
          'delta_lat_o1o2', 'delta_lat_p7p8']].to_string(index=False))

###############################################################################
# ## Trend figure
#
# Six panels:
#   1. IOLD over sessions, with bootstrap 95% CI band on the primary ref.
#      ±8 ms clinical threshold drawn as horizontal guides.
#   2. Per-eye P100 latency, with bootstrap CI bands.
#   3. Check-size slope difference (L − R, ms / arcmin).
#   4. O1/O2 inter-ocular Δ-asymmetry (latency).
#   5. P7/P8 inter-ocular Δ-asymmetry (latency).
#   6. Inter-ocular amplitude ratio (log2).
#

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
ax_iold, ax_p100, ax_slope, ax_d_o1o2, ax_d_p7p8, ax_amp = axes.flatten()

ref_styles = {
    PRIMARY_REF: dict(color='#2c3e50', linewidth=2.0, alpha=1.0,  marker='o'),
}
# Lighter style for the secondary reference scheme.
for ref in df['ref'].unique():
    if ref != PRIMARY_REF:
        ref_styles[ref] = dict(color='#95a5a6', linewidth=1.2, alpha=0.6, marker='s', linestyle='--')

# --- 1. IOLD over time -----------------------------------------------------
for ref, sub in df.groupby('ref'):
    style = ref_styles.get(ref, dict(color='gray'))
    ax_iold.plot(sub['session_nb'], sub['iold_ms'], label=f'{ref} (point)', **style)
    if ref == PRIMARY_REF and sub['boot_iold_lo'].notna().any():
        ax_iold.fill_between(sub['session_nb'], sub['boot_iold_lo'], sub['boot_iold_hi'],
                              color=style['color'], alpha=0.15, label=f'{ref} (95% CI)')
ax_iold.axhline(8.0,  color='#c0392b', linestyle=':', alpha=0.6, label='±8 ms clinical threshold')
ax_iold.axhline(-8.0, color='#c0392b', linestyle=':', alpha=0.6)
ax_iold.axhline(0,    color='black',   linestyle='-', alpha=0.3)
ax_iold.set_title('IOLD (L − R P100 latency) over sessions')
ax_iold.set_xlabel('Session number')
ax_iold.set_ylabel('IOLD (ms)')
ax_iold.grid(True, alpha=0.3)
ax_iold.legend(loc='best', fontsize=8)

# --- 2. Per-eye P100 latency ----------------------------------------------
for ref, sub in df.groupby('ref'):
    style = ref_styles.get(ref, dict(color='gray'))
    is_primary = (ref == PRIMARY_REF)
    ax_p100.plot(sub['session_nb'], sub['p100_left_ms'],  color='#2980b9',
                  marker='o' if is_primary else 's',
                  linestyle='-' if is_primary else '--',
                  alpha=style['alpha'], linewidth=style['linewidth'],
                  label=f'Left eye [{ref}]')
    ax_p100.plot(sub['session_nb'], sub['p100_right_ms'], color='#c0392b',
                  marker='o' if is_primary else 's',
                  linestyle='-' if is_primary else '--',
                  alpha=style['alpha'], linewidth=style['linewidth'],
                  label=f'Right eye [{ref}]')
    if is_primary and sub['boot_left_lo'].notna().any():
        ax_p100.fill_between(sub['session_nb'], sub['boot_left_lo'], sub['boot_left_hi'],
                              color='#2980b9', alpha=0.15)
        ax_p100.fill_between(sub['session_nb'], sub['boot_right_lo'], sub['boot_right_hi'],
                              color='#c0392b', alpha=0.15)
ax_p100.set_title('Per-eye P100 latency over sessions')
ax_p100.set_xlabel('Session number')
ax_p100.set_ylabel('P100 latency (ms)')
ax_p100.grid(True, alpha=0.3)
ax_p100.legend(loc='best', fontsize=7)

# --- 3. Check-size slope difference ---------------------------------------
for ref, sub in df.groupby('ref'):
    style = ref_styles.get(ref, dict(color='gray'))
    ax_slope.plot(sub['session_nb'], sub['slope_diff'], label=ref, **style)
ax_slope.axhline(0, color='black', linestyle='-', alpha=0.3)
ax_slope.set_title('Inter-ocular check-size slope difference (L − R)')
ax_slope.set_xlabel('Session number')
ax_slope.set_ylabel('Δ slope (ms / arcmin)')
ax_slope.grid(True, alpha=0.3)
ax_slope.legend(loc='best', fontsize=8)

# --- 4. O1/O2 Δ-asymmetry --------------------------------------------------
for ref, sub in df.groupby('ref'):
    style = ref_styles.get(ref, dict(color='gray'))
    ax_d_o1o2.plot(sub['session_nb'], sub['delta_lat_o1o2'], label=ref, **style)
ax_d_o1o2.axhline(0, color='black', linestyle='-', alpha=0.3)
ax_d_o1o2.set_title('O1/O2 inter-ocular Δ-asymmetry (latency)')
ax_d_o1o2.set_xlabel('Session number')
ax_d_o1o2.set_ylabel('Δlat (ms)  — eye-dependent skew')
ax_d_o1o2.grid(True, alpha=0.3)
ax_d_o1o2.legend(loc='best', fontsize=8)

# --- 5. P7/P8 Δ-asymmetry --------------------------------------------------
for ref, sub in df.groupby('ref'):
    style = ref_styles.get(ref, dict(color='gray'))
    ax_d_p7p8.plot(sub['session_nb'], sub['delta_lat_p7p8'], label=ref, **style)
ax_d_p7p8.axhline(0, color='black', linestyle='-', alpha=0.3)
ax_d_p7p8.set_title('P7/P8 inter-ocular Δ-asymmetry (latency)')
ax_d_p7p8.set_xlabel('Session number')
ax_d_p7p8.set_ylabel('Δlat (ms)  — eye-dependent skew')
ax_d_p7p8.grid(True, alpha=0.3)
ax_d_p7p8.legend(loc='best', fontsize=8)

# --- 6. Amplitude ratio (log2) --------------------------------------------
for ref, sub in df.groupby('ref'):
    style = ref_styles.get(ref, dict(color='gray'))
    ax_amp.plot(sub['session_nb'], sub['amp_log2'], label=ref, **style)
ax_amp.axhline(1.0,  color='#c0392b', linestyle=':', alpha=0.6, label='±1 log2 threshold')
ax_amp.axhline(-1.0, color='#c0392b', linestyle=':', alpha=0.6)
ax_amp.axhline(0,    color='black',   linestyle='-', alpha=0.3)
ax_amp.set_title('Inter-ocular amplitude ratio over sessions')
ax_amp.set_xlabel('Session number')
ax_amp.set_ylabel('log2(L / R)')
ax_amp.grid(True, alpha=0.3)
ax_amp.legend(loc='best', fontsize=8)

fig.suptitle(f'PR-VEP longitudinal trends — subject {SUBJECT_ID:04d}, {SITE}',
             fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()

###############################################################################
# ## Baseline summary
#
# Treat the first 3–5 sessions (or all sessions before a known intervention)
# as the baseline, and report the mean ± std of each biomarker. This bracket
# is what subsequent sessions need to fall outside of to count as a real shift.
#

BASELINE_LAST_SESSION_NB = None  # set to e.g. 4 to mark sessions <= 4 as baseline

if BASELINE_LAST_SESSION_NB is not None:
    baseline_mask = df['session_nb'] <= BASELINE_LAST_SESSION_NB
else:
    baseline_mask = slice(None)
    print("[baseline] BASELINE_LAST_SESSION_NB not set — using ALL sessions as baseline. "
          "Edit the constant above to define a baseline window.")

baseline_df = df.loc[baseline_mask] if BASELINE_LAST_SESSION_NB is not None else df
metrics = ['iold_ms', 'slope_diff', 'amp_log2',
           'delta_lat_o1o2', 'delta_lat_p7p8',
           'p100_left_ms', 'p100_right_ms']

primary = baseline_df[baseline_df['ref'] == PRIMARY_REF]
if len(primary) > 0:
    print(f"\n[baseline] reference = {PRIMARY_REF}, n_sessions = {len(primary)}")
    print(primary[metrics].agg(['mean', 'std']).T.to_string())
