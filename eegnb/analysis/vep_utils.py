from mne import Evoked


def print_latency(peak_name, peak_latency, peak_channel, uv):
    peak_latency = round(peak_latency * 1e3, 2)  # convert to milliseconds
    uv = round(uv * 1e6, 2)  # convert to µV
    print('{} Peak of {} µV at {} ms in peak_channel {}'.format(peak_name, uv, peak_latency, peak_channel))


def get_peak(erp_name, evoked_potential, peak_time_min, peak_time_max, mode):
    # print('{} peak min {} max {}'.format(erp_name, peak_time_min, peak_time_max))
    peak_channel, peak_latency, uv = evoked_potential.get_peak(tmin=peak_time_min,
                                                               tmax=peak_time_max,
                                                               mode=mode, return_amplitude=True)
    print_latency(erp_name, peak_latency, peak_channel, uv)
    return peak_latency


def plot_vep(evoked_occipital: Evoked):
    n75_peak_width = 0.05
    n75_latency = get_peak(erp_name='N75',
                           evoked_potential=evoked_occipital,
                           peak_time_min=0.06,
                           peak_time_max=0.075 + n75_peak_width,
                           mode='neg')
    p100_peak_width = 0.1
    p100_latency = get_peak(erp_name='P100',
                            evoked_potential=evoked_occipital,
                            peak_time_min=n75_latency,
                            peak_time_max=n75_latency + p100_peak_width,
                            mode='pos')
    n145_peak_width = 0.12
    n145_latency = get_peak(erp_name='N145',
                            evoked_potential=evoked_occipital,
                            peak_time_min=p100_latency,
                            peak_time_max=p100_latency + n145_peak_width,
                            mode='neg')

    plt = evoked_occipital.plot(spatial_colors=True, show=False)

    # Get the axes from the figure
    axes = plt.get_axes()  # This gets all Axes objects

    # Add vertical lines as markers to each subplot
    ax = axes[0]
    ax.axvline(x=0, color='r', linestyle='--', label='stim')
    ax.axvline(x=0.100, color='r', linestyle='--', label='100 ms')
    #ax.axvline(x=n75_latency, color='g', linestyle='-', label='n75')
    ax.axvline(x=p100_latency, color='g', linestyle='-', label='p100')
    #ax.axvline(x=n145_latency, color='g', linestyle='-', label='n145')

    # Add a legend to each subplot
    # ax.legend()

    # plt.show()

    # Add a legend
    plt.legend(loc="lower right")

    return plt
