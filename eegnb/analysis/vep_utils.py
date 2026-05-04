import numpy as np
from mne import Evoked, EvokedArray
from scipy.stats import trim_mean

ISCEV_CHECK_DEG_LARGE = 1.0
ISCEV_CHECK_DEG_SMALL = 0.25

# Default thresholds used by clinical PR-VEP biomarkers. Override per-call
# only if you have a paradigm-specific reason to.
IOLD_FLAG_MS = 8.0          # |L − R P100 latency| above this flags suspected unilateral demyelination
LOG2_AMP_FLAG = 1.0         # |log2(L/R P100 amplitude)| above this flags inter-ocular drive imbalance


def print_latency(peak_name, peak_latency, peak_channel, uv):
    peak_latency = round(peak_latency * 1e3, 2)  # convert to milliseconds
    uv = round(uv * 1e6, 2)  # convert to µV
    print('{} Peak of {} µV at {} ms in peak_channel {}'.format(peak_name, uv, peak_latency, peak_channel))


def get_peak(erp_name, evoked_potential, peak_time_min, peak_time_max, mode):
    """Find peak latency with sub-sample precision using parabolic interpolation.

    MNE's get_peak returns the sample with the largest value, limiting
    resolution to the sample interval (4 ms at 250 Hz).  A parabolic fit
    through the peak sample and its two neighbours recovers the true peak
    location between samples, giving ~0.5 ms precision at 250 Hz.
    """
    # Step 1: find the sample-level peak via MNE
    try:
        peak_channel, sample_latency, _ = evoked_potential.get_peak(
            tmin=peak_time_min, tmax=peak_time_max,
            mode=mode, return_amplitude=True)
    except ValueError as e:
        print(f'{erp_name}: could not find peak ({e})')
        return None

    # Step 2: parabolic interpolation around the peak sample
    ch_idx = evoked_potential.ch_names.index(peak_channel)
    times = evoked_potential.times
    data = evoked_potential.data[ch_idx]

    peak_sample = np.argmin(np.abs(times - sample_latency))

    # Need at least one sample on each side for the fit
    if 0 < peak_sample < len(times) - 1:
        y_prev = data[peak_sample - 1]
        y_peak = data[peak_sample]
        y_next = data[peak_sample + 1]

        # Parabolic interpolation: offset from centre sample
        denom = y_prev - 2 * y_peak + y_next
        if abs(denom) > 1e-30:
            offset = 0.5 * (y_prev - y_next) / denom
            dt = times[peak_sample] - times[peak_sample - 1]
            interp_latency = times[peak_sample] + offset * dt
            interp_uv = y_peak - 0.25 * (y_prev - y_next) * offset
        else:
            interp_latency = sample_latency
            interp_uv = y_peak
    else:
        interp_latency = sample_latency
        interp_uv = data[peak_sample]

    return {
        'name': erp_name,
        'latency': interp_latency,
        'channel': peak_channel,
        'amplitude': interp_uv
    }


def get_pr_vep_latencies(evoked_occipital: Evoked):
    n75 = get_peak(erp_name='N75',   evoked_potential=evoked_occipital,
             peak_time_min=0.060, peak_time_max=0.090, mode='neg')
    p100 = get_peak(erp_name='P100', evoked_potential=evoked_occipital,
                            peak_time_min=0.080, peak_time_max=0.130, mode='pos')
    n145 = get_peak(erp_name='N145',  evoked_potential=evoked_occipital,
             peak_time_min=0.120, peak_time_max=0.170, mode='neg')

    return n75, p100, n145


# ---------------------------------------------------------------------------
# Robust averaging + JSON helper
# ---------------------------------------------------------------------------

def trimmed_average(epochs, proportiontocut=0.1):
    """Per-sample trimmed-mean evoked across trials.

    Standard ``epochs.average()`` is biased by occasional single-trial outliers
    that survive amplitude-based rejection. Dropping the top and bottom
    ``proportiontocut`` fraction at each timepoint produces a more
    representative waveform without raising the rejection threshold or
    discarding whole epochs.
    """
    data = epochs.get_data()  # (n_trials, n_channels, n_times)
    if data.shape[0] == 0:
        return epochs.average()
    avg = trim_mean(data, proportiontocut=proportiontocut, axis=0)
    nave = int(round(data.shape[0] * (1 - 2 * proportiontocut)))
    return EvokedArray(avg, epochs.info, tmin=epochs.tmin, nave=nave,
                       comment=f'trimmed-mean ({int(proportiontocut * 100)}%)')


def json_safe_float(x):
    """Coerce numpy scalars / None / NaN / Inf to JSON-safe Python floats."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


# ---------------------------------------------------------------------------
# PR-VEP biomarkers (pure computation; printing/plotting stays in callers)
# ---------------------------------------------------------------------------

def compute_iold(p100_left, p100_right, flag_threshold_ms=IOLD_FLAG_MS):
    """Inter-ocular latency difference at P100 (signed L − R, ms).

    Returns ``None`` if either eye is missing a P100 detection. Otherwise a
    dict with per-eye latency/amplitude, the signed difference, and a
    threshold-crossing flag suitable for clinical screening.
    """
    if p100_left is None or p100_right is None:
        return None
    p100_left_ms = p100_left['latency'] * 1000.0
    p100_right_ms = p100_right['latency'] * 1000.0
    iold_ms = p100_left_ms - p100_right_ms
    return {
        'p100_left_ms': json_safe_float(p100_left_ms),
        'p100_right_ms': json_safe_float(p100_right_ms),
        'p100_left_uv': json_safe_float(p100_left['amplitude'] * 1e6),
        'p100_right_uv': json_safe_float(p100_right['amplitude'] * 1e6),
        'iold_ms': json_safe_float(iold_ms),
        'flag': bool(abs(iold_ms) > flag_threshold_ms),
    }


def compute_iold_per_size(epochs, event_id, ch_name,
                          sizes=('large', 'small'),
                          eye_prefixes=('left_eye', 'right_eye'),
                          flag_threshold_ms=IOLD_FLAG_MS):
    """Per-check-size IOLD.

    Demyelination preferentially delays the high-spatial-frequency (small-
    check) response, so the per-size IOLD often surfaces lateralised
    dysfunction that the size-pooled IOLD averages out.

    For each size in ``sizes``, trims-mean per-eye epochs at ``ch_name``,
    locates P100, then runs :func:`compute_iold` on the pair. Returns
    ``{size: iold_dict_or_None}``.
    """
    out = {}
    for size in sizes:
        peaks = {}
        for eye in eye_prefixes:
            cond_key = f'{eye}/{size}'
            if cond_key not in event_id or len(epochs[cond_key]) == 0:
                peaks[eye] = None
                continue
            ev = trimmed_average(epochs[cond_key]).copy().pick([ch_name])
            _, p100, _ = get_pr_vep_latencies(ev)
            peaks[eye] = p100
        out[size] = compute_iold(peaks.get('left_eye'), peaks.get('right_eye'),
                                 flag_threshold_ms=flag_threshold_ms)
    return out


def compute_amplitude_ratio(p100_left, p100_right, log2_flag=LOG2_AMP_FLAG):
    """P100 amplitude ratio L/R (rectified) and log2 ratio.

    Returns ``None`` if either eye lacks a P100 or right-eye amplitude is zero
    (ratio undefined).
    """
    if p100_left is None or p100_right is None:
        return None
    amp_l_uv = abs(p100_left['amplitude']) * 1e6
    amp_r_uv = abs(p100_right['amplitude']) * 1e6
    if amp_r_uv <= 0:
        return None
    ratio = amp_l_uv / amp_r_uv
    log2_ratio = float(np.log2(ratio))
    return {
        'amp_left_uv': json_safe_float(amp_l_uv),
        'amp_right_uv': json_safe_float(amp_r_uv),
        'ratio': json_safe_float(ratio),
        'log2_ratio': json_safe_float(log2_ratio),
        'flag': bool(abs(log2_ratio) > log2_flag),
    }


def compute_check_size_slope(epochs, event_id, ch_name, check_size_arcmin,
                             eye_prefixes=('left_eye', 'right_eye'),
                             min_trials=5):
    """Per-eye P100 latency slope vs. check size (ms / arcmin).

    For each (eye × size) condition present in ``event_id`` (keys of the form
    ``"{eye_prefix}/{size_label}"``), trims-mean across trials, locates the
    P100 in the canonical search window, and fits a line of P100 latency vs.
    check size in arcmin. The L − R slope difference amplifies asymmetric
    spatial-frequency-dependent demyelination.

    Returns a dict with per-condition P100 latencies, per-eye slopes (or None
    if fewer than two sizes were detected), and the L − R slope difference (or
    None if either eye is missing a slope).
    """
    out = {
        'per_condition_p100_ms': {},
        'slope_left_ms_per_arcmin': None,
        'slope_right_ms_per_arcmin': None,
        'slope_diff': None,
    }
    rows = []
    for eye in eye_prefixes:
        for size_label, arcmin in check_size_arcmin.items():
            cond_key = f'{eye}/{size_label}'
            if cond_key not in event_id:
                continue
            ep = epochs[cond_key]
            if len(ep) < min_trials:
                continue
            ev = trimmed_average(ep)
            _, p100_c, _ = get_pr_vep_latencies(ev.copy().pick([ch_name]))
            if p100_c is None:
                continue
            lat_ms = p100_c['latency'] * 1000.0
            rows.append({'eye': eye, 'size_label': size_label,
                         'size_arcmin': arcmin, 'p100_ms': lat_ms})
            out['per_condition_p100_ms'][cond_key] = json_safe_float(lat_ms)

    slopes = {}
    for eye in eye_prefixes:
        eye_rows = [r for r in rows if r['eye'] == eye]
        if len(eye_rows) >= 2:
            xs = np.array([r['size_arcmin'] for r in eye_rows])
            ys = np.array([r['p100_ms'] for r in eye_rows])
            slopes[eye] = float(np.polyfit(xs, ys, 1)[0])
    if 'left_eye' in slopes:
        out['slope_left_ms_per_arcmin'] = json_safe_float(slopes['left_eye'])
    if 'right_eye' in slopes:
        out['slope_right_ms_per_arcmin'] = json_safe_float(slopes['right_eye'])
    if 'left_eye' in slopes and 'right_eye' in slopes:
        out['slope_diff'] = json_safe_float(slopes['left_eye'] - slopes['right_eye'])
    return out


def bootstrap_p100_latency(epochs, event_id, ch_name, eye_prefix,
                           win_ms=(60, 160), n_boot=1000, seed=0,
                           proportiontocut=0.1, min_trials=10):
    """Bootstrap distribution of P100 latency for one eye.

    Trial-resamples with replacement, recomputes the trimmed-mean evoked at
    ``ch_name``, and locates the positive max in ``win_ms``. Returns a
    length-``n_boot`` array of latencies (ms), or ``None`` if there aren't
    enough trials. Pair two calls (one per eye) to derive an IOLD CI from the
    pairwise differences.
    """
    keys = [k for k in event_id if k.startswith(eye_prefix)]
    if not keys:
        return None
    ep = epochs[keys].copy().pick([ch_name])
    data = ep.get_data()[:, 0, :]
    if data.shape[0] < min_trials:
        return None
    times_ms = ep.times * 1000.0
    win_mask = (times_ms >= win_ms[0]) & (times_ms <= win_ms[1])
    win_times = times_ms[win_mask]
    n_trials = data.shape[0]
    rng = np.random.default_rng(seed)
    times_all = ep.times * 1000.0          # full time axis in ms
    dt = float(times_all[1] - times_all[0])  # sample period (ms)
    win_indices = np.where(win_mask)[0]    # absolute indices of window samples
    latencies = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_trials, n_trials)
        boot_avg = trim_mean(data[idx], proportiontocut=proportiontocut, axis=0)

        # Grid-level peak within the search window
        local_peak = np.argmax(boot_avg[win_mask])
        abs_peak = win_indices[local_peak]

        # Parabolic interpolation: recovers sub-sample peak location,
        # reducing CI quantisation from ±dt to roughly ±0.3 dt.
        if 0 < abs_peak < len(boot_avg) - 1:
            y_l = boot_avg[abs_peak - 1]
            y_c = boot_avg[abs_peak]
            y_r = boot_avg[abs_peak + 1]
            denom = y_l - 2 * y_c + y_r
            if abs(denom) > 1e-30:
                offset = 0.5 * (y_l - y_r) / denom  # fractional sample offset
                latencies[b] = times_all[abs_peak] + offset * dt
            else:
                latencies[b] = times_all[abs_peak]
        else:
            latencies[b] = times_all[abs_peak]

    return latencies


def compute_hemi_asymmetry(evoked_avg, ch_left, ch_right,
                           lat_flag_ms=IOLD_FLAG_MS, log2_flag=LOG2_AMP_FLAG):
    """P100 latency/amplitude asymmetry between two hemispheric channels.

    Works for any homologous pair (O1/O2, P7/P8, ...). Returns ``None`` if
    either channel fails P100 detection. Sign convention: ``lat_diff =
    ch_left − ch_right``, so positive means the left-named channel is later.
    """
    ev_l = evoked_avg.copy().pick([ch_left])
    ev_r = evoked_avg.copy().pick([ch_right])
    _, p100_l, _ = get_pr_vep_latencies(ev_l)
    _, p100_r, _ = get_pr_vep_latencies(ev_r)
    if p100_l is None or p100_r is None:
        return None
    lat_l = p100_l['latency'] * 1000.0
    lat_r = p100_r['latency'] * 1000.0
    amp_l = abs(p100_l['amplitude']) * 1e6
    amp_r = abs(p100_r['amplitude']) * 1e6
    lat_diff = lat_l - lat_r
    amp_ratio = amp_l / amp_r if amp_r > 0 else float('inf')
    log2_ratio = float(np.log2(amp_ratio)) if amp_r > 0 else float('nan')
    return {
        f'lat_{ch_left.lower()}': json_safe_float(lat_l),
        f'lat_{ch_right.lower()}': json_safe_float(lat_r),
        f'amp_{ch_left.lower()}': json_safe_float(amp_l),
        f'amp_{ch_right.lower()}': json_safe_float(amp_r),
        'lat_diff_ms': json_safe_float(lat_diff),
        'amp_ratio': json_safe_float(amp_ratio),
        'log2_ratio': json_safe_float(log2_ratio),
        'lat_flag': bool(abs(lat_diff) > lat_flag_ms),
        'amp_flag': bool(amp_r > 0 and abs(log2_ratio) > log2_flag),
    }


def compute_hemi_delta_asymmetry(left_eye_hemi, right_eye_hemi,
                                 ch_left, ch_right):
    """Inter-ocular contrast of hemispheric asymmetry.

    Δlat  = (chL − chR)|left_eye − (chL − chR)|right_eye
    Δlog2 = log2(chL/chR)|left_eye − log2(chL/chR)|right_eye

    A purely anatomical chL<chR pattern produces ≈0 here. A non-zero contrast
    implies the asymmetry depends on which eye is driving cortex —
    pathway-specific rather than scalp-stationary.

    Inputs should be raw (un-rounded) per-eye asymmetry dicts. ``compute_hemi_
    asymmetry`` already JSON-rounds, so for ``compute_hemi_delta_asymmetry``
    pass dicts containing the raw float values you want differenced (typically
    re-derived alongside the per-eye call).
    """
    if left_eye_hemi is None or right_eye_hemi is None:
        return None
    lat_key_l = f'lat_{ch_left.lower()}'
    lat_key_r = f'lat_{ch_right.lower()}'
    amp_key_l = f'amp_{ch_left.lower()}'
    amp_key_r = f'amp_{ch_right.lower()}'
    L, R = left_eye_hemi, right_eye_hemi
    lat_asym_L = L[lat_key_l] - L[lat_key_r]
    lat_asym_R = R[lat_key_l] - R[lat_key_r]
    d_lat = lat_asym_L - lat_asym_R
    log2_asym_L = (np.log2(L[amp_key_l] / L[amp_key_r])
                   if L[amp_key_r] > 0 else float('nan'))
    log2_asym_R = (np.log2(R[amp_key_l] / R[amp_key_r])
                   if R[amp_key_r] > 0 else float('nan'))
    d_log2 = log2_asym_L - log2_asym_R
    return {
        'lat_asym_left': json_safe_float(lat_asym_L),
        'lat_asym_right': json_safe_float(lat_asym_R),
        'd_lat': json_safe_float(d_lat),
        'log2_asym_left': json_safe_float(log2_asym_L),
        'log2_asym_right': json_safe_float(log2_asym_R),
        'd_log2': json_safe_float(d_log2),
    }
