"""VR's runtime-free surface: geometry from the view fov, missed-vsync
counting, clock sync, and telemetry rows that match their header."""

import csv
import ctypes
import enum
import math
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("psychopy")

from eegnb.devices.vr import VR


class _SessionState(enum.Enum):
    IDLE = 1
    READY = 2
    SYNCHRONIZED = 3
    VISIBLE = 4
    FOCUSED = 5
    STOPPING = 6


class _Fov:
    def __init__(self, left, right, up, down):
        self.angle_left, self.angle_right = left, right
        self.angle_up, self.angle_down = up, down


class _Pos:
    def __init__(self, x):
        self.x = x


class _Pose:
    def __init__(self, x):
        self.position = _Pos(x)


class _View:
    def __init__(self, fov, x):
        self.fov, self.pose = fov, _Pose(x)


def _bare(views=None):
    win = VR.__new__(VR)
    win._views = views
    win._eye_size = (1616, 1648)
    win._frame_index = 8
    win.timing_data = []
    win.last_predicted_display_time_s = 12.5
    win.last_runtime_time_s = 12.48
    win.last_display_period_ms = 8.333
    win.last_frame_step_ms = 8.354
    win._missed_vsyncs = 3
    win._missed_events = 2
    win._longest_step_ms = 16.7
    win._zero_steps = 0
    win.app_dropped_max = 3
    win.mirror_swap_every = 1
    win.runtime_name = "test 0.0"
    win.system_name = "test hmd"
    win.swapchain_format = 0x8C43
    win.frameIntervals = []
    win._closed = True
    win._input_enabled = False
    win._monoscopic = False

    class _XR:
        SessionState = _SessionState
    win._xr = _XR()
    win._session_state = _SessionState.FOCUSED
    return win


QUEST2_VIEWS = [
    _View(_Fov(-0.9075712, 0.7853982, 0.8377581, -0.8726646), -0.0316),
    _View(_Fov(-0.7853982, 0.9075712, 0.8377581, -0.8726646), 0.0316),
]


def test_optical_axis_offsets_mirror_between_eyes():
    left, right = _bare(QUEST2_VIEWS).compute_optical_axis_offsets()
    assert left == pytest.approx(-right)
    assert left > 0


def test_optical_axis_offsets_match_the_asymmetric_lens_formula():
    fov = QUEST2_VIEWS[0].fov
    tan_l, tan_r = -math.tan(fov.angle_left), math.tan(fov.angle_right)
    expected = (tan_l - tan_r) / (tan_l + tan_r)
    assert _bare(QUEST2_VIEWS).compute_optical_axis_offsets()[0] == pytest.approx(expected)


def test_pixels_per_degree_is_tangent_pixels_scaled_to_degrees():
    ppd_h, ppd_v = _bare(QUEST2_VIEWS).pixels_per_degree()
    fov = QUEST2_VIEWS[0].fov
    assert ppd_h == pytest.approx(
        1616 / (math.tan(fov.angle_right) - math.tan(fov.angle_left)) * math.pi / 180)
    assert 10 < ppd_h < 15 and 10 < ppd_v < 15


def test_log_display_info_reports_ipd_from_the_view_poses():
    win = _bare(QUEST2_VIEWS)
    ppd, ipd_mm = win.log_display_info()
    assert ppd == 12
    assert ipd_mm == pytest.approx(63.2)
    assert win.timing_data[0][0] == '# ipd_mm'


class _LayerView:
    def __init__(self):
        self.pose = self.fov = None
        self.sub_image = SimpleNamespace(
            swapchain=None, image_rect=SimpleNamespace(offset=[0, 0], extent=[0, 0]))


def _located(monoscopic):
    win = _bare(QUEST2_VIEWS)
    win._monoscopic = monoscopic
    win.session = win.space = win._view_config = None
    win._swapchains = ['left', 'right']
    win._xr.ViewLocateInfo = lambda **kw: kw
    win._xr.locate_views = lambda session, view_locate_info: (None, QUEST2_VIEWS)
    win._xr.CompositionLayerProjectionView = _LayerView
    win._xr.CompositionLayerProjection = lambda space: SimpleNamespace(space=space, views=None)
    win._locate_views(0)
    return win


def _fovs(views):
    return [(v.fov.angle_left, v.fov.angle_right, v.fov.angle_up, v.fov.angle_down)
            for v in views]


@pytest.mark.parametrize("monoscopic", [True, False])
def test_layers_carry_each_eyes_own_fov(monoscopic):
    assert _fovs(_located(monoscopic)._layer.views) == _fovs(QUEST2_VIEWS)


def test_monoscopic_draws_in_one_symmetric_fov_covering_both_eyes():
    win = _located(monoscopic=True)
    fovs = win.eye_fov
    assert fovs[0] == fovs[1]
    left, right, up, down = fovs[0]
    assert left == pytest.approx(-right) and up == pytest.approx(-down)
    for view in QUEST2_VIEWS:
        assert left <= view.fov.angle_left + 1e-9 and right >= view.fov.angle_right - 1e-9
        assert up >= view.fov.angle_up - 1e-9 and down <= view.fov.angle_down + 1e-9
    ppd_h, ppd_v = win.pixels_per_degree()
    assert ppd_h == pytest.approx(ppd_v)
    assert win.compute_optical_axis_offsets() == pytest.approx((0.0, 0.0))


def test_monoscopic_remap_shows_each_drawn_direction_in_the_same_direction_to_both_eyes():
    win = _located(monoscopic=True)
    w, h = win._eye_size
    left, right, up, down = (math.tan(a) for a in win.eye_fov[0])
    pixel = (right - left) / w
    for x, y in [(w / 2, h / 2), (0.3 * w, 0.7 * h), (0.6 * w, 0.2 * h)]:
        drawn = (left + x / w * (right - left), down + y / h * (up - down))
        for view in QUEST2_VIEWS:
            x0, y0, x1, y1 = win._mono_source_rect(view)
            eye_x, eye_y = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
            fov = view.fov
            shown = (math.tan(fov.angle_left) + eye_x * (math.tan(fov.angle_right)
                                                         - math.tan(fov.angle_left)),
                     math.tan(fov.angle_down) + eye_y * (math.tan(fov.angle_up)
                                                         - math.tan(fov.angle_down)))
            assert shown == pytest.approx(drawn, abs=pixel)


class _GL:
    GL_FRAMEBUFFER, GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER = 1, 2, 3
    GL_COLOR_ATTACHMENT0 = GL_TEXTURE_2D = GL_SCISSOR_TEST = GL_COLOR_BUFFER_BIT = GL_LINEAR = 0

    def __init__(self):
        self.attached = {}
        self.blits = []

    def glBindFramebuffer(self, target, fbo):
        pass

    def glFramebufferTexture2D(self, target, attachment, textarget, texture, level):
        self.attached[target] = texture

    def glDisable(self, cap):
        pass

    def glBlitFramebuffer(self, *args):
        self.blits.append((self.attached[self.GL_READ_FRAMEBUFFER],
                           self.attached[self.GL_FRAMEBUFFER], args[:4]))


class _Layer(ctypes.Structure):
    _fields_ = [('layer_flags', ctypes.c_uint64)]


def test_monoscopic_submit_copies_the_drawn_image_into_each_eye(monkeypatch):
    gl = _GL()
    monkeypatch.setattr("eegnb.devices.vr.GL", gl)
    win = _located(monoscopic=True)
    win._mono_texture = 'drawn'
    win._swapchain_images = [[SimpleNamespace(image='left eye')],
                             [SimpleNamespace(image='right eye')]]
    win._acquired = {}
    win._draw_fbo = win._copy_fbo = None
    win._passthrough_on = False
    win._layer = _Layer(0)
    win._mirror_swap_counter = 0
    win.mirror_swap_every = 0
    win._frame_step_ns = None
    win._frame_state = SimpleNamespace(should_render=True, predicted_display_time=1,
                                       predicted_display_period=8333333)
    win._xr_time_to_s = lambda t: 0.0
    win._xr.SwapchainImageAcquireInfo = win._xr.SwapchainImageReleaseInfo = lambda: None
    win._xr.SwapchainImageWaitInfo = lambda timeout: None
    win._xr.INFINITE_DURATION = 0
    win._xr.acquire_swapchain_image = lambda swapchain, acquire_info: 0
    win._xr.wait_swapchain_image = lambda swapchain, wait_info: None
    win._xr.release_swapchain_image = lambda swapchain, release_info: None
    win._xr.EnvironmentBlendMode = SimpleNamespace(OPAQUE=1)
    win._xr.FrameEndInfo = lambda **kw: kw
    win._xr.end_frame = lambda session, frame_end_info: None
    win._submit_frame()
    assert gl.blits == [('drawn', 'left eye', win._mono_source_rect(QUEST2_VIEWS[0])),
                        ('drawn', 'right eye', win._mono_source_rect(QUEST2_VIEWS[1]))]


def _teardown_log(monkeypatch, win):
    log = []
    monkeypatch.setattr("eegnb.devices.vr.GL", SimpleNamespace(
        glDeleteTextures=lambda n, names: log.append(('texture', n, list(names))),
        glDeleteFramebuffers=lambda n, names: log.append(('framebuffer', n, list(names)))))
    for kind in ('swapchain', 'passthrough_layer_fb', 'passthrough_fb', 'action_set',
                 'space', 'session', 'instance'):
        setattr(win._xr, f'destroy_{kind}',
                lambda handle, kind=kind: log.append((kind, handle)))
    return log


def test_teardown_frees_the_mono_texture_and_both_framebuffers_before_the_session(monkeypatch):
    win = _bare(QUEST2_VIEWS)
    log = _teardown_log(monkeypatch, win)
    win.winHandle = SimpleNamespace(switch_to=lambda: log.append(('switch_to',)))
    win._mono_texture, win._draw_fbo, win._copy_fbo = 7, 3, 4
    win._swapchains = ['left', 'right']
    win.session, win.instance = 'session', 'instance'
    win._destroy_xr()
    assert log == [('switch_to',), ('texture', 1, [7]), ('framebuffer', 1, [3]),
                   ('framebuffer', 1, [4]), ('swapchain', 'left'), ('swapchain', 'right'),
                   ('session', 'session'), ('instance', 'instance')]
    assert (win._mono_texture, win._draw_fbo, win._copy_fbo) == (None, None, None)


def test_teardown_after_construction_failed_before_the_window_skips_the_gl_objects(monkeypatch):
    win = _bare(QUEST2_VIEWS)
    log = _teardown_log(monkeypatch, win)
    win.instance = 'instance'
    win._destroy_xr()
    assert log == [('instance', 'instance')]


def test_geometry_without_views_degrades_rather_than_raising():
    win = _bare(None)
    assert win.compute_optical_axis_offsets() == (0.0, 0.0)
    assert win.pixels_per_degree() == (None, None)
    assert win.log_display_info() == (None, None)


def _row():
    win = _bare(QUEST2_VIEWS)
    win.log_telemetry(3, 100.0, marker=2)
    header = (["trial_idx", "software_time", "submitted_frame_index"]
              + list(win._DERIVED_TIMING_FIELDS))
    assert len(win.timing_data[0]) == len(header)
    return dict(zip(header, win.timing_data[0]))


def test_telemetry_row_carries_the_submitted_frames_prediction():
    row = _row()
    assert row["submitted_frame_index"] == 7
    assert row["marker"] == 2
    assert row["frame_step_ms"] == 8.354
    assert row["predicted_display_time_s"] == 12.5
    assert row["runtime_time_s"] == 12.48
    assert row["missed_vsyncs"] == 3
    assert row["session_state"] == "FOCUSED"


def test_summary_reports_the_missed_vsync_counts():
    summary = _bare(QUEST2_VIEWS).get_session_summary()
    assert summary["app_dropped"] == 3
    assert summary["missed_vsyncs"] == 3
    assert summary["missed_vsync_events"] == 2
    assert summary["longest_step_ms"] == 16.7
    assert summary["session_status"]["is_visible"] is True
    assert summary["session_status"]["hmd_mounted"] is None
    assert "asw_activations" not in summary
    assert "phase_stats" not in summary


def test_session_that_stops_mid_run_raises(monkeypatch):
    win = _bare(QUEST2_VIEWS)
    win._session_running = False
    win._session_state = _SessionState.STOPPING
    win._session_lost = False
    win.resume_after_stop = False
    win._context = SimpleNamespace(pump=lambda: None)
    monkeypatch.setattr(win, "_poll_events", lambda: None)
    with pytest.raises(RuntimeError, match="STOPPING"):
        win._begin_frame()


def test_session_stopped_on_an_instruction_screen_waits_and_resumes(monkeypatch):
    win = _bare(QUEST2_VIEWS)
    win._session_running = False
    win._session_lost = False
    win._session_state = _SessionState.STOPPING
    win._passthrough_on = False
    win._last_pdt = 123
    win.resume_after_stop = True
    win.session = None
    win._context = SimpleNamespace(pump=lambda: None)
    polls = []

    def poll():
        polls.append(1)
        if len(polls) == 3:
            win._session_running = True
            win._session_state = _SessionState.FOCUSED

    monkeypatch.setattr(win, "_poll_events", poll)
    period = 8_333_333
    win._xr.wait_frame = lambda session: SimpleNamespace(
        predicted_display_time=10 * period, predicted_display_period=period, should_render=False)
    win._xr.begin_frame = lambda session: None
    monkeypatch.setattr(win, "_locate_views", lambda display_time: None)
    monkeypatch.setattr("eegnb.devices.vr.GL", SimpleNamespace(
        glBindFramebuffer=lambda *args: None, GL_FRAMEBUFFER=0))
    win._begin_frame()
    assert len(polls) == 3
    assert win._frame_state is not None
    assert win._missed_vsyncs == 3


def test_lost_session_raises_even_on_an_instruction_screen(monkeypatch):
    win = _bare(QUEST2_VIEWS)
    win._session_running = False
    win._session_lost = True
    win._session_state = _SessionState.STOPPING
    win.resume_after_stop = True
    win._context = SimpleNamespace(pump=lambda: None)
    monkeypatch.setattr(win, "_poll_events", lambda: None)
    with pytest.raises(RuntimeError, match="STOPPING"):
        win._begin_frame()


def test_late_frames_are_counted_as_missed_vsyncs(monkeypatch):
    win = _bare(QUEST2_VIEWS)
    win._missed_vsyncs = win._missed_events = win.app_dropped_max = 0
    win._longest_step_ms = 0.0
    win._frame_index = 0
    win._last_pdt = None
    win._session_running = True
    win._context = SimpleNamespace(pump=lambda: None)
    win.session = None
    period = 8_333_333
    times = iter([0, period, 3 * period, 3 * period + period])

    def wait_frame(session):
        return SimpleNamespace(predicted_display_time=next(times),
                               predicted_display_period=period, should_render=False)

    win._xr.wait_frame = wait_frame
    win._xr.begin_frame = lambda session: None
    monkeypatch.setattr(win, "_poll_events", lambda: None)
    monkeypatch.setattr(win, "_locate_views", lambda display_time: None)
    monkeypatch.setattr("eegnb.devices.vr.GL", SimpleNamespace(
        glBindFramebuffer=lambda *args: None, GL_FRAMEBUFFER=0))
    for _ in range(4):
        win._begin_frame()
    assert win._missed_vsyncs == 1
    assert win._missed_events == 1
    assert win._longest_step_ms == pytest.approx(2 * period / 1e6)


def test_frames_the_headset_is_not_showing_are_not_counted_as_late(monkeypatch):
    win = _bare(QUEST2_VIEWS)
    win._missed_vsyncs = win._missed_events = win.app_dropped_max = 0
    win._longest_step_ms = 0.0
    win._frame_index = 0
    win._last_pdt = None
    win._session_running = True
    win._context = SimpleNamespace(pump=lambda: None)
    win.session = None
    period = 8_333_333
    frames = iter([(0, _SessionState.FOCUSED), (period, _SessionState.SYNCHRONIZED),
                   (2000 * period, _SessionState.SYNCHRONIZED),
                   (2004 * period, _SessionState.FOCUSED), (2005 * period, _SessionState.FOCUSED)])
    states = []

    def wait_frame(session):
        pdt, state = next(frames)
        win._session_state = state
        return SimpleNamespace(predicted_display_time=pdt,
                               predicted_display_period=period, should_render=False)

    win._xr.wait_frame = wait_frame
    win._xr.begin_frame = lambda session: None
    monkeypatch.setattr(win, "_poll_events", lambda: None)
    monkeypatch.setattr(win, "_locate_views", lambda display_time: None)
    monkeypatch.setattr("eegnb.devices.vr.GL", SimpleNamespace(
        glBindFramebuffer=lambda *args: None, GL_FRAMEBUFFER=0))
    for _ in range(5):
        win._begin_frame()
    assert win._missed_vsyncs == 0
    assert win._missed_events == 0


def test_clock_offset_is_cached_and_matches_wall_minus_perf_counter():
    win = _bare(QUEST2_VIEWS)
    win.runtime_to_wallclock_offset = None
    win.runtime_to_wallclock_bracket = None
    offset = win.sync_vr_clock()
    assert offset == pytest.approx(time.time() - time.perf_counter(), abs=0.05)
    assert win.runtime_to_wallclock_bracket >= 0
    assert win.sync_vr_clock() == offset


def test_save_telemetry_writes_a_header_matching_every_row(tmp_path):
    win = _bare(QUEST2_VIEWS)
    win.runtime_to_wallclock_offset = None
    win.log_telemetry(3, 100.0, marker=2)
    win.save_telemetry(tmp_path / "recording.csv")

    with (tmp_path / "recording_timing.csv").open(newline="") as handle:
        rows = list(csv.reader(line for line in handle if not line.startswith('#')))
    header, row = rows
    assert len(header) == len(row)
    assert header[:3] == ["trial_idx", "software_time", "submitted_frame_index"]
    sample = dict(zip(header, row))
    assert sample["predicted_display_time_s"] == "12.5"
    assert sample["marker"] == "2"


def test_save_telemetry_accepts_a_string_path(tmp_path):
    win = _bare(QUEST2_VIEWS)
    win.runtime_to_wallclock_offset = None
    win.save_telemetry(str(tmp_path / "recording.csv"))
    assert (tmp_path / "recording_timing.csv").is_file()


def test_save_telemetry_without_a_path_writes_nothing():
    assert _bare(QUEST2_VIEWS).save_telemetry(None) is None
