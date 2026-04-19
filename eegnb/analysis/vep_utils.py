import numpy as np
from mne import Evoked


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

    print_latency(erp_name, interp_latency, peak_channel, interp_uv)
    return interp_latency


def plot_vep(evoked_occipital: Evoked):
    # Fixed absolute windows — independent of each other so a missed N75
    # doesn't cascade into a missed P100 or N145.
    get_peak(erp_name='N75',   evoked_potential=evoked_occipital,
             peak_time_min=0.060, peak_time_max=0.090, mode='neg')
    p100_latency = get_peak(erp_name='P100', evoked_potential=evoked_occipital,
                            peak_time_min=0.080, peak_time_max=0.130, mode='pos')
    get_peak(erp_name='N145',  evoked_potential=evoked_occipital,
             peak_time_min=0.120, peak_time_max=0.170, mode='neg')

    fig = evoked_occipital.plot(show=False)

    ax = fig.get_axes()[0]
    ax.axvline(x=0,     color='r', linestyle='--', label='stim')
    ax.axvline(x=0.100, color='r', linestyle='--', label='100 ms')
    if p100_latency is not None:
        ax.axvline(x=p100_latency, color='g', linestyle='-', label='p100')

    fig.legend(loc="lower right")

    return fig
