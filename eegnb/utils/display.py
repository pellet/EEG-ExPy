"""Display-related helpers shared across experiments and example scripts."""

import logging

# Common panel refresh rates seen on consumer monitors and HMDs.
STANDARD_REFRESH_HZ = (60, 72, 75, 90, 100, 120, 144, 165, 240)


def snap_refresh_rate(measured_hz: float, tolerance_pct: float = 3.0) -> int:
    """Snap a noisy measured refresh rate to the nearest standard panel rate.

    PsychoPy's getActualFrameRate() and libovr's displayRefreshRate both
    occasionally report a value 1-2 Hz off the nominal panel rate due to
    measurement jitter or runtime quirks. This helper rounds them back to
    something physically meaningful.

    Falls back to the rounded measured value (with a warning) when the
    measurement is more than tolerance_pct away from any standard rate, so
    truly unusual panels (e.g. 85 Hz CRTs) still pass through correctly.
    """
    snapped = min(STANDARD_REFRESH_HZ, key=lambda h: abs(h - measured_hz))
    if abs(snapped - measured_hz) / snapped * 100 > tolerance_pct:
        rounded = int(round(measured_hz))
        logging.warning(
            "[display] measured refresh rate %.2f Hz didn't match any "
            "standard rate within %.1f%%; using rounded %d Hz",
            measured_hz, tolerance_pct, rounded,
        )
        return rounded
    return snapped
