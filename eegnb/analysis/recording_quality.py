"""Per-recording contact quality diagnostic for an eegnb session directory.

Can be used programmatically (returns a formatted string or structured dict)
or from the command line via ``tools/recording_quality.py``.
"""
import pathlib

import numpy as np
import pandas as pd


EEG_CHANNEL_CANDIDATES = [
    'Fz', 'Pz', 'P7', 'P8', 'O1', 'O2', 'Oz', 'M1', 'M2',
    'Cz', 'C3', 'C4', 'F3', 'F4', 'T7', 'T8', 'F7', 'F8',
    'AF7', 'AF8', 'TP9', 'TP10',
]

BIG_SAMPLE_UV    = 200.0
DRIFT_WIN_SECS   = 10
NOISE_FACTOR_FLAG = 1.5


def _detect_sfreq(timestamps: np.ndarray) -> float:
    diffs = np.diff(timestamps)
    diffs = diffs[(diffs > 0) & (diffs < 1.0)]
    return float(1.0 / np.median(diffs)) if len(diffs) else 250.0


def _channel_metrics(x: np.ndarray, sfreq: float) -> dict:
    x = x - np.mean(x)
    win = max(1, int(DRIFT_WIN_SECS * sfreq))
    drift = float(pd.Series(x).rolling(win, center=False).mean().dropna().pipe(
        lambda s: s.max() - s.min()
    )) if len(x) >= win else 0.0
    return {
        'std':     float(np.std(x)),
        'p99':     float(np.percentile(np.abs(x), 99)),
        'max':     float(x.max()),
        'min':     float(x.min()),
        'drift':   drift,
        'pct_big': 100.0 * float(np.mean(np.abs(x) > BIG_SAMPLE_UV)),
    }


def check_session(session_dir: pathlib.Path) -> dict:
    """Return structured quality metrics for a session directory.

    Reads every ``recording_*.csv`` (excluding ``*_timing.csv``) and computes
    per-channel std / p99 / drift metrics.

    Returns a dict with keys:
      report             : str  — formatted text report (same as report_session())
      flagged_channels   : list — channel names flagged in any recording
                                  (std or drift > 1.5× group median)
      shared_ref_suspect : bool — True when ≥ half of channels are flagged,
                                  indicating M1/SRB loose rather than isolated
                                  electrode contacts
    """
    session_dir = pathlib.Path(session_dir).expanduser().resolve()
    recs = sorted(
        p for p in session_dir.glob('recording_*.csv')
        if not p.stem.endswith('_timing') and not p.name.endswith('.excluded')
    )

    lines = []
    w = lines.append

    if not recs:
        w(f'No recording_*.csv files found in {session_dir}')
        return {'report': '\n'.join(lines), 'flagged_channels': [], 'shared_ref_suspect': False}

    w(f'Session: {session_dir.name}')
    w(f'Found {len(recs)} recording(s)')
    w('')
    w('Column legend (all values in µV, raw signal — compare channels relative to each other):')
    w(f'  std      Standard deviation — overall noise level per channel.')
    w(f'  p99|x|   99th percentile of |signal| — sustained noise floor, robust to single spikes.')
    w(f'  max/min  Largest/smallest raw sample — flags electrode pops or movement artifacts.')
    w(f'  drift    Range of 10 s rolling mean — slow DC wander from drying gel or loose contact.')
    w(f'  %>200    % of samples exceeding {BIG_SAMPLE_UV:.0f} µV — mainly useful comparing recordings, not channels.')
    w('')
    w('Interpreting the table: look for channels with std or drift significantly higher than')
    w('the others. A single outlier = bad electrode contact. All channels inflated = loose reference.')
    w('')

    rows = []
    flagged_channels: set = set()

    for p in recs:
        df = pd.read_csv(p)
        eeg_chs = [c for c in df.columns if c in EEG_CHANNEL_CANDIDATES]
        sfreq = _detect_sfreq(df['timestamps'].values) if 'timestamps' in df else 250.0
        rec_min = len(df) / sfreq / 60.0
        w(f'=== {p.stem.split("recording_")[-1]}  '
          f'({rec_min:.1f} min, {len(df)} samples, ~{sfreq:.0f} Hz) ===')

        ch_metrics = {}
        for ch in eeg_chs:
            ch_metrics[ch] = _channel_metrics(df[ch].values.astype(float), sfreq)
            rows.append({'rec': p.stem.split('recording_')[-1], 'ch': ch, **ch_metrics[ch]})

        med_std   = float(np.median([m['std']   for m in ch_metrics.values()]))
        med_drift = float(np.median([m['drift'] for m in ch_metrics.values()]))

        w(f'  {"ch":>4}  {"std":>8}  {"std×":>5}  {"p99|x|":>8}  '
          f'{"max":>8}  {"min":>8}  {"drift":>8}  {"drift×":>6}  {"%>{:.0f}".format(BIG_SAMPLE_UV):>6}')
        for ch, m in ch_metrics.items():
            std_x   = m['std']   / med_std   if med_std   > 0 else 1.0
            drift_x = m['drift'] / med_drift if med_drift > 0 else 1.0
            flagged = std_x > NOISE_FACTOR_FLAG or drift_x > NOISE_FACTOR_FLAG
            if flagged:
                flagged_channels.add(ch)
            flag_str = '⚑' if flagged else ' '
            w(f'  {ch:>4}  {m["std"]:>8.1f}  {std_x:>5.2f}  {m["p99"]:>8.1f}  '
              f'{m["max"]:>8.1f}  {m["min"]:>8.1f}  '
              f'{m["drift"]:>8.1f}  {drift_x:>6.2f}  {m["pct_big"]:>6.2f}  {flag_str}')
        w(f'  (median std={med_std:.1f} µV   median drift={med_drift:.1f} µV   '
          f'flag threshold: ×>{NOISE_FACTOR_FLAG})')

        # Shared-reference diagnosis per recording: if ≥ half of channels are
        # flagged the noise is uniform, pointing at a loose M1/SRB rather than
        # isolated electrode contacts.
        n_flagged_rec = sum(
            1 for m in ch_metrics.values()
            if (m['std'] / med_std if med_std > 0 else 1.0) > NOISE_FACTOR_FLAG
            or (m['drift'] / med_drift if med_drift > 0 else 1.0) > NOISE_FACTOR_FLAG
        )
        if n_flagged_rec >= len(eeg_chs) / 2:
            w(f'  ⚑ SHARED REFERENCE SUSPECT — {n_flagged_rec}/{len(eeg_chs)} channels flagged '
              f'(all-channel inflation → M1/SRB loose)')
        elif n_flagged_rec:
            w(f'  ⚑ {n_flagged_rec} channel(s) flagged — isolated contact issue(s)')
        w('')

    df_all = pd.DataFrame(rows)
    summary = df_all.groupby('rec').agg(
        med_std=('std', 'median'),
        med_p99=('p99', 'median'),
        med_drift=('drift', 'median'),
    )

    w('=== PER-RECORDING SUMMARY (median across EEG channels) ===')
    w(f'  {"rec":<22}  {"med_std":>8}  {"med_p99":>8}  {"med_drift":>10}')
    for rec in summary.index:
        s = summary.loc[rec]
        w(f'  {rec:<22}  {s.med_std:>8.1f}  {s.med_p99:>8.1f}  {s.med_drift:>10.1f}')

    w('')
    w('=== EXCLUSION CANDIDATES ===')
    baseline = float(summary.med_std.min())
    any_exclude = False
    for rec in summary.index:
        factor = float(summary.loc[rec, 'med_std']) / baseline
        flag = factor > NOISE_FACTOR_FLAG
        any_exclude |= flag
        w(f'  {rec}: median_std={summary.loc[rec,"med_std"]:.1f}  '
          f'({factor:.2f}x)  [{"EXCLUDE?" if flag else "ok"}]')

    w('')
    if any_exclude:
        w('⚑ One or more recordings flagged as substantially noisier.')

    # Determine overall shared-reference suspicion across the whole session
    all_ch_names = list(dict.fromkeys(r['ch'] for r in rows))  # ordered unique
    shared_ref_suspect = len(flagged_channels) >= len(all_ch_names) / 2

    return {
        'report':             '\n'.join(lines),
        'flagged_channels':   sorted(flagged_channels),
        'shared_ref_suspect': shared_ref_suspect,
    }


def report_session(session_dir: pathlib.Path) -> str:
    """Return a formatted quality report string for a session directory.

    Reads every ``recording_*.csv`` (excluding ``*_timing.csv``) and reports
    per-channel std / p99 / drift, plus exclusion candidates.
    """
    return check_session(session_dir)['report']
