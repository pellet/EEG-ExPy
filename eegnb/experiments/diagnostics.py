"""Pre- and post-experiment diagnostic checks.

Used by ``eegnb.experiments.Experiment`` to validate the experimental setup
(display frame rate, electrode contact) before the trial loop starts, and to
summarise recording quality afterwards.

Each function is a pure(-ish) routine that takes runtime objects and returns
a result dict or string. The Experiment class is responsible for calling them
at the right point and rendering the output. Keeping these out of
Experiment.py makes that class about *what* the experiment does rather than
*how* the setup is validated.
"""
from time import sleep
import pathlib

import numpy as np


# ---------------------------------------------------------------------------
# Monitor (flat display)
# ---------------------------------------------------------------------------

def build_flat_monitor(screen_num=0):
    """Create a PsychoPy ``Monitor`` from detected screen properties.

    Avoids the 'Monitor specification not found' warning that PsychoPy emits
    when a flat ``visual.Window`` can't locate a saved calibration file.

    Note: there is no equivalent ``measure_frame_rate`` for monitors — flat
    displays deliver their nominal rate reliably (no encoded transport
    pipeline), and PsychoPy's ``Window.getActualFrameRate`` already provides
    the measurement when needed. Frame-rate validation lives on the ``VR``
    class because it's only informative where target Hz is decoupled from
    actual delivery (Quest Link).
    """
    import logging
    from psychopy import monitors
    from psychopy import logging as psy_logging

    # Temporarily elevate console logger to ERROR to suppress the
    # "Monitor specification not found" warning during initialization.
    if hasattr(psy_logging, 'console') and psy_logging.console:
        old_level = psy_logging.console.level
        psy_logging.console.setLevel(logging.ERROR)
    else:
        old_level = None

    try:
        mon = monitors.Monitor('eegnb_auto', autoLog=False)
    finally:
        if old_level is not None:
            psy_logging.console.setLevel(old_level)

    mon.setDistance(60)
    try:
        import pyglet
        screen = pyglet.canvas.Display().get_screens()[screen_num]
        mon.setSizePix([screen.width, screen.height])
    except Exception:
        mon.setSizePix([1920, 1080])
    
    # Persist the monitor specification so PsychoPy finds it on disk next time
    mon.save()
    return mon


# ---------------------------------------------------------------------------
# Pre-experiment signal quality check
#
# A brief read of the EEG amplifier's incoming signal to catch obviously
# broken channels (loose electrode, dead reference) before a session
# starts.
#
# This pre-flight check catches gross failures
# (no contact, broken wire, badly seated electrode); subtler problems
# like high-but-uniform noise or slow drift are caught by the post-session
# quality report instead.
# ---------------------------------------------------------------------------

# Flag a channel only when both thresholds are exceeded — keeps the warning
# rate low for normal-but-noisy sessions.
SIGNAL_NOISE_FLAG_UV     = 200.0   # absolute floor: nothing usable above this
SIGNAL_NOISE_REL_FACTOR  = 3.0     # relative to montage median


def check_signal_quality(eeg, n_seconds=3):
    """Read a brief EEG buffer and flag clearly broken channels.

    Used before a recording starts to catch hardware problems (loose
    electrode, dead reference) before the recording begins.
    The function reads ``n_seconds`` of live signal from the amplifier,
    detrends each channel (1-second rolling-mean subtraction so DC drift
    doesn't dominate), and computes the standard deviation as a baseline-
    noise estimate.

    A channel is flagged only when its noise is BOTH:
      - above ``SIGNAL_NOISE_FLAG_UV`` µV in absolute terms, AND
      - more than ``SIGNAL_NOISE_REL_FACTOR`` × the group median.

    Both conditions are required so that a session where every channel is
    a bit noisy doesn't trigger spurious warnings — only individually
    broken channels are surfaced.

    Note: this is *not* an impedance check. Real impedance measurement
    requires putting the amplifier into a dedicated test mode and is not
    supported by all BrainFlow backends. Baseline noise is a proxy that
    works for the common failure mode (loose / dry electrodes).

    Returns:
        Dict with:
            ``stds``     per-channel detrended std (µV)
            ``median``   group median (µV)
            ``flagged``  list of ``(channel, std_uv)`` for broken contacts
            ``skipped``  True if not run (non-brainflow backend or error)
    """
    result = {'stds': {}, 'median': None, 'flagged': [], 'skipped': True}

    if not eeg or getattr(eeg, 'backend', None) != 'brainflow':
        return result

    try:
        sfreq = int(getattr(eeg, 'sfreq', 250))
        n_samples = sfreq * n_seconds

        eeg._start_brainflow()
        sleep(n_seconds)
        raw = eeg.board.get_current_board_data(n_samples)
        ch_names, eeg_data, _ = eeg._brainflow_extract(raw)

        # Stop so the main eeg.start() can restart cleanly later
        eeg.board.stop_stream()
        eeg.board.release_session()
        eeg_data = np.array(eeg_data)
        win = sfreq
        for ch_name, x in zip(ch_names, eeg_data):
            if len(x) < win:
                std = float(np.std(x))
            else:
                rolling = np.convolve(x, np.ones(win) / win, mode='same')
                std = float(np.std(x - rolling))
            result['stds'][ch_name] = round(std, 1)

        if result['stds']:
            med = float(np.median(list(result['stds'].values())))
            result['median'] = round(med, 1)
            
            # If the median itself is huge, the reference/ground is likely bad,
            # so the relative check (3x median) will hide the noise. In this case,
            # just flag anything over the absolute threshold.
            bad_ref_mode = med > SIGNAL_NOISE_FLAG_UV
            
            for ch, s in result['stds'].items():
                if bad_ref_mode:
                    if s > SIGNAL_NOISE_FLAG_UV:
                        result['flagged'].append((ch, s))
                else:
                    if s > SIGNAL_NOISE_FLAG_UV and s > SIGNAL_NOISE_REL_FACTOR * med:
                        result['flagged'].append((ch, s))

        result['skipped'] = False
        print(f"[signal-check] {len(ch_names)} ch over {n_seconds}s — "
              f"median noise = {result['median']} µV, "
              f"flagged: {[c for c, _ in result['flagged']] or 'none'}")
        return result
    except Exception as e:
        print(f"[signal-check] skipped — {e}")
        return result


# ---------------------------------------------------------------------------
# Post-session report
# ---------------------------------------------------------------------------

def post_session_report(save_fn):
    """Build the recording quality report string for display after a session."""
    from eegnb.analysis.recording_quality import report_session
    return report_session(pathlib.Path(save_fn).parent)


# ---------------------------------------------------------------------------
# Diagnostics screen formatting
# ---------------------------------------------------------------------------

def format_diagnostic_warnings(*, device_name=None, display=None, signal_check=None):
    """Build warnings for the pre-experiment diagnostics screen.

    Returns a list of strings — empty if everything's fine.
    """
    warnings = []

    if device_name and 'synthetic' in device_name.lower():
        warnings.append(
            "[!] SYNTHETIC EEG DEVICE — NO REAL DATA WILL BE RECORDED\n"
            "    Set eeg_device = EEG(device, ...) in your run script before re-running."
        )

    if display and not display.get('ok', True):
        warnings.append(
            f"[!] DISPLAY WARNING — frame delivery severely degraded\n"
            f"    Target: {display['target_hz']:.0f} Hz  -  "
            f"Measured: {display['actual_hz']:.1f} Hz  "
            f"({display['deviation_pct']:.1f}% off)\n"
            f"    Likely cause: wrong GPU selected, or GPU acceleration disabled.\n"
            f"    Fix: set python.exe to NVIDIA in NVIDIA Control Panel and\n"
            f"    Windows Graphics Settings, then restart."
        )

    if signal_check and signal_check.get('flagged'):
        n_flagged = len(signal_check['flagged'])
        n_total = len(signal_check.get('stds', {}))
        ch_info = ", ".join(f"{ch} ({std:.0f} µV)" for ch, std in signal_check['flagged'])
        
        msg = (
            f"[!] SIGNAL QUALITY WARNING\n"
            f"    Channels with abnormally high noise: {ch_info}\n"
            f"    (group median is {signal_check['median']} µV)\n"
        )
        
        if n_total > 0 and n_flagged >= (n_total / 2.0):
            msg += (
                f"    Likely cause: Bad Reference (M1) or Ground (A2) connection.\n"
                f"    When noise is universally high across the head, the shared\n"
                f"    reference is usually loose or dry. Re-seat M1 and A2."
            )
        else:
            msg += (
                f"    Likely cause: loose electrode or dry paste. Re-seat the\n"
                f"    listed contact(s) before continuing."
            )
        warnings.append(msg)

    return warnings
