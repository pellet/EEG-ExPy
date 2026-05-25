"""Diode-based photon timing correction for VR PR-VEP recordings.

Two stages:

1. ``detect_transitions`` — recovers LCD bright/dark phase transitions
   from a strobing-backlight VR display via rolling-max envelope +
   log-Otsu + adaptive Schmitt trigger, with guard zones around
   dropout/drift epochs.
2. ``fuse_timing`` — combines diode's accurate absolute timing
   (~40 ms median lag) with LibOVR's tight per-trial precision (~2 ms
   std) when both are available.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d


DETECTOR_VERSION = "1.2-dropout-guard"

# Defaults tuned for Cyton 250 Hz ADC + Quest 2 72 Hz LCD strobe.
ENV_WIN_MS       = 100.0    # rolling-max envelope window
HYST_UV          = 5.0      # fixed-hyst fallback / safety-floor margin
DEADZONE_PCT     = (20, 80) # (above_pct, below_pct) for adaptive Schmitt bounds
FWD_WIN_MS       = 300.0    # forward search window after each marker
FUSION_AGREE_MS  = 20.0     # max OVR/diode median diff to enable fusion
FALLBACK_LAG_S   = 0.025    # unmeasured-hardware fallback (flat-monitor regime)
DROPOUT_MIN_MS   = 150.0    # min dead-zone dwell to flag as dropout/drift
DROPOUT_GUARD_MS = 100.0    # guard margin on each side of a dropout region


# ---------------------------------------------------------------------------
# Detector primitives
# ---------------------------------------------------------------------------

def otsu_threshold(values):
    """Otsu's method on log-transformed values.

    Log first so the heavy upper tail of strobe-peak catches doesn't pull
    the threshold deep into the bright class.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 10:
        return float(np.median(values))
    log_vals = np.log(np.maximum(values, 1e-6))
    lo, hi = np.percentile(log_vals, [1, 99])
    if hi <= lo:
        return float(np.exp((lo + hi) / 2))
    candidates = np.linspace(lo, hi, 200)
    best_var, best_t = -np.inf, float((lo + hi) / 2)
    for t in candidates:
        below = log_vals < t
        w0 = float(below.mean())
        if w0 < 0.05 or w0 > 0.95:
            continue
        m0 = float(log_vals[below].mean())
        m1 = float(log_vals[~below].mean())
        var = w0 * (1 - w0) * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, float(t)
    return float(np.exp(best_t))


def adaptive_deadzone(env_uv, threshold_uv, pct=DEADZONE_PCT, min_hyst_uv=HYST_UV):
    """Schmitt-trigger bounds derived from the actual plateau positions.

    ``lo_t`` = ``pct[1]``-th percentile of values below threshold (top of
    the dark plateau). ``hi_t`` = ``pct[0]``-th percentile of values
    above threshold (bottom of the bright plateau). ``min_hyst_uv`` is
    a safety floor when the percentile band collapses on very clean
    recordings.
    """
    below = env_uv[env_uv < threshold_uv]
    above = env_uv[env_uv >= threshold_uv]
    if len(below) < 10 or len(above) < 10:
        return threshold_uv - min_hyst_uv, threshold_uv + min_hyst_uv
    lo_t = float(np.percentile(below, pct[1]))
    hi_t = float(np.percentile(above, pct[0]))
    lo_t = min(lo_t, threshold_uv - min_hyst_uv)
    hi_t = max(hi_t, threshold_uv + min_hyst_uv)
    return lo_t, hi_t


def find_dropout_regions(env_uv, lo_t, hi_t, sfreq, min_dur_ms=DROPOUT_MIN_MS):
    """Contiguous spans where the envelope lingers inside ``[lo_t, hi_t]``
    for longer than ``min_dur_ms``.

    A real phase transition crosses the dead zone in ~one envelope window;
    dropouts and beat-drift epochs trap it for much longer. Returns an
    ``(N, 2)`` array of ``[start, end)`` sample ranges (empty ``(0, 2)``
    if none).
    """
    in_dz = (env_uv >= lo_t) & (env_uv <= hi_t)
    min_samp = max(1, int(min_dur_ms / 1000 * sfreq))

    regions = []
    i = 0
    n = len(in_dz)
    while i < n:
        if in_dz[i]:
            j = i
            while j < n and in_dz[j]:
                j += 1
            if (j - i) >= min_samp:
                regions.append((i, j))
            i = j
        else:
            i += 1

    if regions:
        return np.array(regions, dtype=int)
    return np.empty((0, 2), dtype=int)


def schmitt_state(env_uv, threshold_uv=None, hyst_uv=HYST_UV,
                  lo_t=None, hi_t=None):
    """Schmitt-trigger state machine: 1 once env ≥ ``hi_t``, 0 once env < ``lo_t``.

    Two call modes:
      - ``(env, threshold_uv, hyst_uv)`` — symmetric hysteresis (legacy).
      - ``(env, lo_t=..., hi_t=...)`` — explicit asymmetric bounds,
        used with ``adaptive_deadzone``.
    """
    if lo_t is None or hi_t is None:
        assert threshold_uv is not None, "need either (threshold_uv, hyst_uv) or (lo_t, hi_t)"
        lo_t = threshold_uv - hyst_uv
        hi_t = threshold_uv + hyst_uv
    midpoint = (lo_t + hi_t) / 2.0
    state = np.empty(len(env_uv), dtype=np.int8)
    current = 1 if env_uv[0] >= midpoint else 0
    for i, v in enumerate(env_uv):
        if   current == 0 and v >= hi_t: current = 1
        elif current == 1 and v <  lo_t: current = 0
        state[i] = current
    return state


def detect_transitions(diode_uv, sfreq, env_win_ms=ENV_WIN_MS,
                       deadzone_pct=DEADZONE_PCT, min_hyst_uv=HYST_UV,
                       dropout_min_ms=DROPOUT_MIN_MS,
                       dropout_guard_ms=DROPOUT_GUARD_MS):
    """Polarity-corrected LCD phase transitions from a diode trace.

    Pipeline: centred rolling-max envelope → log-Otsu threshold →
    adaptive dead zone → Schmitt-trigger state → polarity-aware edge
    correction → dropout-guarded trust mask.

    Returns a dict with keys:
        transitions      sample indices, polarity-corrected
        polarities       +1 for dark→bright, -1 for bright→dark
        trusted          bool; False inside dropout guard zones
        dropout_regions  (N, 2) sample ranges of detected dropouts
        n_dropouts, n_untrusted
        threshold_uv, lo_t, hi_t, env_win
        env_lo, env_hi   5th/95th percentile of raw signal (informational)
    """
    env_win = max(2, int(env_win_ms / 1000 * sfreq))
    env = maximum_filter1d(diode_uv, size=env_win)
    threshold = otsu_threshold(env)
    lo_t, hi_t = adaptive_deadzone(env, threshold,
                                   pct=deadzone_pct, min_hyst_uv=min_hyst_uv)
    state = schmitt_state(env, lo_t=lo_t, hi_t=hi_t)

    changes = np.diff(state.astype(int))
    rise = np.where(changes ==  1)[0] + 1
    fall = np.where(changes == -1)[0] + 1

    # Centred rolling-max smears rising edges by -W/2 and falling by +W/2;
    # add/subtract half the window to recover the true transition time.
    half = env_win / 2.0
    rise_corrected = rise + half
    fall_corrected = fall - half

    transitions = np.concatenate([rise_corrected, fall_corrected])
    polarities  = np.concatenate([np.ones(len(rise_corrected)),
                                  -np.ones(len(fall_corrected))]).astype(np.int8)
    order = np.argsort(transitions)
    transitions = transitions[order]
    polarities  = polarities[order]

    dropout_regions = find_dropout_regions(env, lo_t, hi_t, sfreq,
                                           min_dur_ms=dropout_min_ms)
    guard_samp = int(dropout_guard_ms / 1000 * sfreq)
    trusted = np.ones(len(transitions), dtype=bool)
    for d_start, d_end in dropout_regions:
        guarded_lo = d_start - guard_samp
        guarded_hi = d_end + guard_samp
        trusted &= ~((transitions >= guarded_lo) & (transitions <= guarded_hi))

    return {
        'transitions':      transitions,
        'polarities':       polarities,
        'trusted':          trusted,
        'dropout_regions':  dropout_regions,
        'n_dropouts':       len(dropout_regions),
        'n_untrusted':      int((~trusted).sum()),
        'threshold_uv':     float(threshold),
        'lo_t':             float(lo_t),
        'hi_t':             float(hi_t),
        'env_win':          env_win,
        'env_lo':           float(np.percentile(diode_uv, 5)),
        'env_hi':           float(np.percentile(diode_uv, 95)),
    }


def forward_match(trans_samples, marker_samples, fwd_max_samp, trusted=None):
    """For each marker, sample of the first trusted transition in
    ``[marker, marker + fwd_max_samp]``. Unmatched markers return -1.

    Forward-only because the software marker is always written before
    the photon update fires. ``trusted`` (optional) excludes transitions
    near dropout regions.
    """
    matched = np.full(len(marker_samples), -1.0)
    if len(trans_samples) == 0:
        return matched
    if trusted is not None:
        trans_samples = np.asarray(trans_samples)[trusted]
        if len(trans_samples) == 0:
            return matched
    ts = np.sort(trans_samples)
    for i, m in enumerate(marker_samples):
        idx = np.searchsorted(ts, m, side='left')
        if idx < len(ts) and ts[idx] <= m + fwd_max_samp:
            matched[i] = ts[idx]
    return matched


# ---------------------------------------------------------------------------
# High-level: diode correction + OVR fusion
# ---------------------------------------------------------------------------

def correct_events_with_diode(diode_uv, events, sfreq,
                              env_win_ms=ENV_WIN_MS,
                              deadzone_pct=DEADZONE_PCT,
                              min_hyst_uv=HYST_UV,
                              fwd_win_ms=FWD_WIN_MS,
                              dropout_min_ms=DROPOUT_MIN_MS,
                              dropout_guard_ms=DROPOUT_GUARD_MS):
    """Run the diode detector and match each marker to the next trusted
    transition.

    Unmatched markers are filled with the session median lag so every
    event in ``events_corrected`` is usable downstream. Returns ``None``
    if nothing matched at all (caller should fall back to OVR or
    software timing).
    """
    det = detect_transitions(diode_uv, sfreq,
                             env_win_ms=env_win_ms,
                             deadzone_pct=deadzone_pct,
                             min_hyst_uv=min_hyst_uv,
                             dropout_min_ms=dropout_min_ms,
                             dropout_guard_ms=dropout_guard_ms)
    fwd_max_samp = int(fwd_win_ms / 1000 * sfreq)
    matched = forward_match(det['transitions'], events[:, 0], fwd_max_samp,
                            trusted=det['trusted'])
    n_matched = int((matched >= 0).sum())

    if n_matched == 0:
        return None

    shifts = matched[matched >= 0] - events[matched >= 0, 0]
    assert (shifts >= 0).all(), "forward-search matches must be >= marker time"
    median_shift = float(np.median(shifts))

    events_corrected = events.copy()
    n_imputed = 0
    for i in range(len(events)):
        if matched[i] >= 0:
            events_corrected[i, 0] = int(round(matched[i]))
        else:
            events_corrected[i, 0] = int(round(events[i, 0] + median_shift))
            n_imputed += 1

    diode_lag_s = (events_corrected[:, 0] - events[:, 0]) / sfreq

    return {
        'events_corrected': events_corrected,
        'lag_s':            diode_lag_s,
        'matched_lag_s':    shifts / sfreq,           # only matched trials
        'n_matched':        n_matched,
        'n_total':          len(events),
        'n_imputed':        n_imputed,
        'threshold_uv':     det['threshold_uv'],
        'lo_t':             det['lo_t'],
        'hi_t':             det['hi_t'],
        'env_win':          det['env_win'],
        'env_lo':           det['env_lo'],
        'env_hi':           det['env_hi'],
        'n_transitions':    len(det['transitions']),
        'n_trusted':        int(det['trusted'].sum()),
        'n_untrusted':      det['n_untrusted'],
        'n_dropouts':       det['n_dropouts'],
        'dropout_regions':  det['dropout_regions'],
        'n_rise':           int((det['polarities'] ==  1).sum()),
        'n_fall':           int((det['polarities'] == -1).sum()),
        'transitions':      det['transitions'],
        'polarities':       det['polarities'],
        'trusted':          det['trusted'],
    }


def fuse_timing(events, sfreq, diode=None, ovr_lag_s=None, ovr_events=None,
                fusion_agree_ms=FUSION_AGREE_MS, fallback_lag_s=FALLBACK_LAG_S):
    """Pick the best per-trial timing source.

    Priority:
      1. Fusion — diode + OVR available, medians agree within
         ``fusion_agree_ms``: ``pc_lag[i] = diode_med + (ovr[i] - ovr_med)``.
         Diode gives absolute calibration; OVR gives per-trial precision.
      2. Diode alone — OVR missing or disagrees with diode.
      3. OVR alone — diode missing; adds ``fallback_lag_s`` for the
         unmeasured hardware transmission delay.
      4. Uncorrected — no source; adds ``fallback_lag_s`` as a flat lag.

    NaN entries in ``ovr_lag_s`` (dropped compositor frames) are imputed
    before the int-cast — otherwise ``np.round(NaN * sfreq).astype(int)``
    silently produces INT_MIN and corrupts those events.
    """
    out = {
        'detector_version':       DETECTOR_VERSION,
        'fusion_agree_ms':        fusion_agree_ms,
    }

    have_diode = diode is not None
    have_ovr   = ovr_lag_s is not None

    if have_diode and have_ovr:
        diode_med = float(np.median(diode['lag_s']))
        ovr_med   = float(np.nanmedian(ovr_lag_s))
        diff_s    = abs(ovr_med - diode_med)
        ovr_std   = float(np.nanstd(ovr_lag_s))
        diode_std = float(diode['lag_s'].std())

        out.update({
            'diode_median_s': diode_med,
            'ovr_median_s':   ovr_med,
            'median_diff_s':  ovr_med - diode_med,
            'ovr_std_s':      ovr_std,
            'diode_std_s':    diode_std,
        })

        if diff_s < fusion_agree_ms / 1000.0:
            pc_lag_s = diode_med + (ovr_lag_s - ovr_med)
            nan_mask = ~np.isfinite(pc_lag_s)
            n_imputed_ovr = int(nan_mask.sum())
            if n_imputed_ovr:
                pc_lag_s = pc_lag_s.copy()
                pc_lag_s[nan_mask] = diode_med
            events_corrected = events.copy()
            events_corrected[:, 0] = events[:, 0] + np.round(pc_lag_s * sfreq).astype(int)
            out.update({
                'events_corrected':       events_corrected,
                'pc_lag_s':               pc_lag_s,
                'src':                    'fused_diode_ovr',
                'unmeasured_lag_shift_s': 0.0,
                'fused_std_s':            float(np.nanstd(pc_lag_s)),
                'n_ovr_nan_imputed':      n_imputed_ovr,
            })
            return out

        out.update({
            'events_corrected':       diode['events_corrected'],
            'pc_lag_s':               diode['lag_s'],
            'src':                    'photodiode',
            'unmeasured_lag_shift_s': 0.0,
            'fusion_rejected_reason': f"OVR vs diode median diff {diff_s*1000:.1f} ms "
                                      f"> ±{fusion_agree_ms:.0f} ms",
        })
        return out

    if have_diode:
        out.update({
            'events_corrected':       diode['events_corrected'],
            'pc_lag_s':               diode['lag_s'],
            'src':                    'photodiode',
            'unmeasured_lag_shift_s': 0.0,
        })
        return out

    if have_ovr:
        ovr_clean = np.asarray(ovr_lag_s, dtype=float)
        nan_mask = ~np.isfinite(ovr_clean)
        n_imputed_ovr = int(nan_mask.sum())
        if n_imputed_ovr:
            ovr_clean = ovr_clean.copy()
            ovr_clean[nan_mask] = float(np.nanmedian(ovr_lag_s))
        if ovr_events is None:
            ovr_events = events.copy()
            ovr_events[:, 0] = events[:, 0] + np.round(ovr_clean * sfreq).astype(int)
        out.update({
            'events_corrected':       ovr_events,
            'pc_lag_s':               ovr_clean,
            'src':                    'libovr_perfstats',
            'unmeasured_lag_shift_s': fallback_lag_s,
            'n_ovr_nan_imputed':      n_imputed_ovr,
        })
        return out

    out.update({
        'events_corrected':       events.copy(),
        'pc_lag_s':               np.zeros(len(events)),
        'src':                    'uncorrected',
        'unmeasured_lag_shift_s': fallback_lag_s,
    })
    return out
