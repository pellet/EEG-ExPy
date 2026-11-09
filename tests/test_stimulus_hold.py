import itertools
import math

import pytest

pytest.importorskip("psychopy")

import eegnb.experiments.Experiment as experiment_module
from eegnb.experiments.visual_n170.n170 import VisualN170
from eegnb.experiments.visual_p300.p300 import VisualP300
from eegnb.experiments.visual_vep.pattern_reversal_vep import VisualPatternReversalVEP

REFRESH_S = 1 / 120


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 50e-6
        return self.now


class _Window:
    def __init__(self, clock):
        self.clock = clock
        self.drawn = []
        self.frames = []

    def flip(self):
        self.clock.now = (math.floor(self.clock.now / REFRESH_S) + 1) * REFRESH_S
        self.frames.append(tuple(self.drawn))
        self.drawn = []

    def setBuffer(self, eye):
        self.drawn.append(eye)

    def setDefaultView(self):
        pass

    def sync_vr_clock(self):
        pass

    def check_eye_separation(self):
        pass


class _Stim:
    def __init__(self, window, name):
        self.window = window
        self.name = name

    def draw(self):
        self.window.drawn.append(self.name)


class _Recorder:
    backend = "brainflow"

    def __init__(self):
        self.markers = []

    def push_sample(self, marker, timestamp):
        self.markers.append(marker)


def _n170(window, eeg):
    exp = VisualN170(duration=3, eeg=eeg)
    exp.houses = [_Stim(window, "house")]
    exp.faces = [_Stim(window, "face")]
    return exp


def _p300(window, eeg):
    exp = VisualP300(duration=3, eeg=eeg)
    exp.nontargets = [_Stim(window, "nontarget")]
    exp.targets = [_Stim(window, "target")]
    return exp


def _pattern_reversal(window, eeg):
    exp = VisualPatternReversalVEP(display_refresh_rate=120, eeg=eeg, block_duration_seconds=3,
                                   block_trial_size=10, n_blocks=1)
    exp.black_background = _Stim(window, "black")
    exp.grey_background = _Stim(window, "grey")
    exp.stim = {eye: {"checkerboards": [_Stim(window, f"{eye} checks 0"),
                                        _Stim(window, f"{eye} checks 1")],
                      "fixation": _Stim(window, f"{eye} fixation")}
                for eye in ("left", "right", "monoscopic")}
    return exp


EXPERIMENTS = [_n170, _p300, _pattern_reversal]
EXPERIMENT_IDS = ["n170", "p300", "pattern_reversal"]
PATHS = pytest.mark.parametrize("use_vr", [False, True], ids=["monitor", "headset"])


def _run(build, use_vr, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(experiment_module, "time", clock)
    window = _Window(clock)
    eeg = _Recorder()
    exp = build(window, eeg)
    exp.use_vr = use_vr
    exp.window = window
    if use_vr:
        exp.vr = window
    exp._clear_user_input = lambda: None
    exp._user_input = lambda input_type: False
    assert exp._run_trial_loop(clock(), exp.duration)
    holds = [(frame, len(list(group))) for frame, group in itertools.groupby(window.frames)
             if frame]
    return exp, holds, eeg.markers


@PATHS
@pytest.mark.parametrize("build", EXPERIMENTS, ids=EXPERIMENT_IDS)
def test_the_stimulus_is_drawn_on_every_refresh_while_it_is_held(monkeypatch, build, use_vr):
    exp, holds, markers = _run(build, use_vr, monkeypatch)
    assert len(holds) >= 3
    for frame, refreshes in holds[:-1]:
        assert refreshes >= round(exp.soa / REFRESH_S)


@PATHS
@pytest.mark.parametrize("build", EXPERIMENTS, ids=EXPERIMENT_IDS)
def test_the_marker_is_sent_once_per_trial_not_once_per_redraw(monkeypatch, build, use_vr):
    exp, holds, markers = _run(build, use_vr, monkeypatch)
    assert len(markers) == len(holds)
    assert sum(refreshes for frame, refreshes in holds) > 10 * len(markers)
