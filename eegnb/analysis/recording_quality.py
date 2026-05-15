"""Per-recording contact quality diagnostic for an eegnb session directory.

Returns structured DataFrames for notebook display alongside a formatted text
report for console / experiment use.

Notebook usage:
    result = check_session(session_dir)
    display(result['per_channel'])         # styled DataFrame, columns aligned
    display(result['summary'])
    result['flagged_channels']
    result['shared_ref_suspect']

Console / experiment usage:
    print(check_session(session_dir)['report'])
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


def _build_per_channel_df(rows: list) -> pd.DataFrame:
    """Build a tidy per-channel DataFrame with relative metrics and flags."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    med_by_rec = df.groupby('rec')[['std', 'drift']].transform('median')
    df['std_x']    = df['std']   / med_by_rec['std'].where(med_by_rec['std']   > 0, 1.0)
    df['drift_x']  = df['drift'] / med_by_rec['drift'].where(med_by_rec['drift'] > 0, 1.0)
    df['flagged']  = (df['std_x'] > NOISE_FACTOR_FLAG) | (df['drift_x'] > NOISE_FACTOR_FLAG)
    return df[['rec', 'ch', 'std', 'std_x', 'p99', 'max', 'min',
               'drift', 'drift_x', 'pct_big', 'flagged']]


def _build_summary_df(per_channel: pd.DataFrame) -> pd.DataFrame:
    """Per-recording summary with exclusion factor relative to cleanest recording."""
    if per_channel.empty:
        return pd.DataFrame()
    summary = per_channel.groupby('rec').agg(
        med_std=('std', 'median'),
        med_p99=('p99', 'median'),
        med_drift=('drift', 'median'),
        n_flagged=('flagged', 'sum'),
        n_channels=('ch', 'count'),
    )
    baseline = float(summary.med_std.min())
    summary['factor_x']  = summary.med_std / baseline if baseline > 0 else 1.0
    summary['exclude']   = summary.factor_x > NOISE_FACTOR_FLAG
    return summary


def _format_report(session_dir: pathlib.Path,
                   recs: list,
                   per_channel: pd.DataFrame,
                   summary: pd.DataFrame) -> str:
    """Format the structured DataFrames into the text report string."""
    lines = []
    w = lines.append

    if not recs:
        w(f'No recording_*.csv files found in {session_dir}')
        return '\n'.join(lines)

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

    header = (f'  {"ch":>4}  {"std":>8}  {"std×":>5}  {"p99|x|":>8}  '
              f'{"max":>8}  {"min":>8}  {"drift":>8}  {"drift×":>6}  '
              f'{"%>{:.0f}".format(BIG_SAMPLE_UV):>6}  {"flag":>4}')

    for rec_id in summary.index:
        rec_rows = per_channel[per_channel['rec'] == rec_id]
        n_samples = len(rec_rows) and int(rec_rows.iloc[0].get('n_samples', 0))  # placeholder
        rec_meta = next((r for r in recs if r['rec'] == rec_id), None)
        if rec_meta:
            w(f'=== {rec_id}  ({rec_meta["minutes"]:.1f} min, '
              f'{rec_meta["n_samples"]} samples, ~{rec_meta["sfreq"]:.0f} Hz) ===')
        else:
            w(f'=== {rec_id} ===')
        w(header)
        for _, m in rec_rows.iterrows():
            flag_str = '⚑' if m['flagged'] else ' '
            w(f'  {m["ch"]:>4}  {m["std"]:>8.1f}  {m["std_x"]:>5.2f}  {m["p99"]:>8.1f}  '
              f'{m["max"]:>8.1f}  {m["min"]:>8.1f}  '
              f'{m["drift"]:>8.1f}  {m["drift_x"]:>6.2f}  {m["pct_big"]:>6.2f}  {flag_str:>4}')

        s = summary.loc[rec_id]
        w(f'  (median std={s.med_std:.1f} µV   median drift={s.med_drift:.1f} µV   '
          f'flag threshold: ×>{NOISE_FACTOR_FLAG})')
        if s.n_flagged >= s.n_channels / 2:
            w(f'  ⚑ SHARED REFERENCE SUSPECT — {int(s.n_flagged)}/{int(s.n_channels)} channels flagged '
              f'(all-channel inflation → M1/SRB loose)')
        elif s.n_flagged:
            w(f'  ⚑ {int(s.n_flagged)} channel(s) flagged — isolated contact issue(s)')
        w('')

    w('=== PER-RECORDING SUMMARY (median across EEG channels) ===')
    w(f'  {"rec":<22}  {"med_std":>8}  {"med_p99":>8}  {"med_drift":>10}')
    for rec_id in summary.index:
        s = summary.loc[rec_id]
        w(f'  {rec_id:<22}  {s.med_std:>8.1f}  {s.med_p99:>8.1f}  {s.med_drift:>10.1f}')

    w('')
    w('=== EXCLUSION CANDIDATES ===')
    for rec_id in summary.index:
        s = summary.loc[rec_id]
        w(f'  {rec_id}: median_std={s.med_std:.1f}  '
          f'({s.factor_x:.2f}x)  [{"EXCLUDE?" if s.exclude else "ok"}]')

    w('')
    if summary.exclude.any():
        w('⚑ One or more recordings flagged as substantially noisier.')

    return '\n'.join(lines)


def check_session(session_dir: pathlib.Path) -> dict:
    """Return structured quality metrics for a session directory.

    Reads every ``recording_*.csv`` (excluding ``*_timing.csv``) and computes
    per-channel std / p99 / drift metrics.

    Returns a dict with keys:
      per_channel        : DataFrame  — one row per (recording, channel) with
                                        std, std×, p99, max/min, drift, drift×,
                                        pct_big, flagged
      summary            : DataFrame  — one row per recording with med_std,
                                        med_p99, med_drift, n_flagged,
                                        n_channels, factor_x, exclude
      report             : str        — formatted text report for console use
      flagged_channels   : list       — channel names flagged in any recording
                                        (std or drift > 1.5× group median)
      shared_ref_suspect : bool       — True when ≥ half of all unique channels
                                        are flagged across the session,
                                        indicating M1/SRB loose rather than
                                        isolated electrode contacts
    """
    session_dir = pathlib.Path(session_dir).expanduser().resolve()
    rec_paths = sorted(
        p for p in session_dir.glob('recording_*.csv')
        if not p.stem.endswith('_timing') and not p.name.endswith('.excluded')
    )

    if not rec_paths:
        empty = pd.DataFrame()
        return {
            'per_channel':        empty,
            'summary':            empty,
            'report':             f'No recording_*.csv files found in {session_dir}',
            'flagged_channels':   [],
            'shared_ref_suspect': False,
        }

    rows = []
    recs_meta = []

    for p in rec_paths:
        df = pd.read_csv(p)
        eeg_chs = [c for c in df.columns if c in EEG_CHANNEL_CANDIDATES]
        sfreq = _detect_sfreq(df['timestamps'].values) if 'timestamps' in df else 250.0
        rec_id = p.stem.split('recording_')[-1]
        recs_meta.append({
            'rec': rec_id, 'minutes': len(df) / sfreq / 60.0,
            'n_samples': len(df), 'sfreq': sfreq,
        })
        for ch in eeg_chs:
            rows.append({'rec': rec_id, 'ch': ch,
                         **_channel_metrics(df[ch].values.astype(float), sfreq)})

    per_channel = _build_per_channel_df(rows)
    summary     = _build_summary_df(per_channel)
    report      = _format_report(session_dir, recs_meta, per_channel, summary)

    flagged_channels = sorted(per_channel.loc[per_channel.flagged, 'ch'].unique().tolist())
    all_ch_names = list(dict.fromkeys(per_channel['ch']))
    shared_ref_suspect = len(flagged_channels) >= len(all_ch_names) / 2

    return {
        'per_channel':        per_channel,
        'summary':            summary,
        'report':             report,
        'flagged_channels':   flagged_channels,
        'shared_ref_suspect': shared_ref_suspect,
    }


def report_session(session_dir: pathlib.Path) -> str:
    """Return a formatted quality report string for a session directory."""
    return check_session(session_dir)['report']
