"""Process-priority / OS-scheduler helpers for time-critical sections.

Centralised so the three independent knobs that affect frame-pacing —
Windows scheduler tick resolution, process priority, and Python GC —
are managed together. PsychoPy's ``core.rush`` ONLY sets process
priority on Windows (verified against the upstream source); it does
NOT touch the scheduler tick or GC, so the other two need explicit
handling.

Usage:
    from eegnb.utils.realtime import force_high_res_timer, high_priority_section

    # At module-import time so the resolution is high before any later
    # imports do any sleeps:
    force_high_res_timer()

    # Around the time-critical trial loop:
    with high_priority_section():
        run_trial_loop()
"""

from __future__ import annotations

import gc
import logging
import sys
from contextlib import contextmanager

from psychopy import core

_diag = logging.getLogger(__name__)


def force_high_res_timer() -> bool:
    """Set the Windows scheduler tick to 1 ms via ``timeBeginPeriod(1)``.

    Windows' default tick is 15.625 ms. Any ``time.sleep`` (including
    those inside libovr's ``waitToBeginFrame`` and BrainFlow's serial
    poll) rounds up to that tick, mathematically locking a 72 Hz
    (13.89 ms) render loop to half-rate. Calling this at module import
    means the resolution is high BEFORE psychxr / brainflow / any
    sleeping code touches it.

    No-op on non-Windows platforms (Linux/macOS already use a 1 ms or
    finer scheduler tick natively).

    Returns ``True`` if the call succeeded, ``False`` otherwise.
    """
    if sys.platform != 'win32':
        return False
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        _diag.info("[timer] called timeBeginPeriod(1)")
        return True
    except Exception as e:
        _diag.warning("[timer] timeBeginPeriod failed: %s", e)
        return False


def query_timer_resolution_ms():
    """Return current system timer resolution in ms (None on non-Windows
    or query failure). Resolution is reported in 100-ns units by
    ``NtQueryTimerResolution``."""
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes
        ntdll = ctypes.windll.ntdll
        _min = wintypes.ULONG(); _max = wintypes.ULONG(); _cur = wintypes.ULONG()
        ntdll.NtQueryTimerResolution(
            ctypes.byref(_min), ctypes.byref(_max), ctypes.byref(_cur)
        )
        return _cur.value / 10000.0  # 100-ns → ms
    except Exception:
        return None


@contextmanager
def high_priority_section():
    """Context manager that puts the OS / Python runtime into time-
    critical mode for the wrapped block. Use around any precision-
    timed section: the trial loop, vr.validate_frame_rate, etc.

    What it does:
      - ``timeBeginPeriod(1)`` → 1 ms Windows scheduler tick (idempotent,
        process-wide; needed for libovr's waitToBeginFrame sleeps to
        resolve below the 15.6 ms default)
      - ``core.rush(True)``    → SetPriorityClass(HIGH_PRIORITY_CLASS) on Windows
      - ``gc.disable()``       → suspends Python's generational GC
    What it does NOT do:
      - Stop reference-count-driven deallocation (Python's main GC mode).
      - Stop GC inside C extensions (BrainFlow, numpy, etc.).
      - Call ``timeEndPeriod``. The 1 ms tick stays on for the rest of
        the process — re-entering this context is a no-op for the timer,
        only rush + gc are toggled per-section.

    Process priority + GC are reversed on exit, even if the block raises.
    """
    force_high_res_timer()
    core.rush(True)
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        core.rush(False)
