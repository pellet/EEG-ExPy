import numpy as np
from mne import Evoked

ISCEV_CHECK_DEG_LARGE = 1.0
ISCEV_CHECK_DEG_SMALL = 0.25


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
