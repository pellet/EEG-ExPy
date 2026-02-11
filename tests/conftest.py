"""
Shared pytest fixtures for EEG-ExPy integration tests.

This module provides reusable fixtures for mocking EEG devices,
PsychoPy components, and controller inputs.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from pathlib import Path


class MockEEG:
    """
    Mock EEG device that simulates the eegnb.devices.eeg.EEG interface.

    Tracks all interactions including start/stop calls, marker pushes,
    and provides synthetic data on request.
    """

    def __init__(self, device_name="synthetic"):
        self.device_name = device_name
        self.sfreq = 256
        self.channels = ['TP9', 'AF7', 'AF8', 'TP10']
        self.n_channels = 4
        self.backend = "brainflow"

        # Track state
        self.started = False
        self.stopped = False
        self.markers = []
        self.save_fn = None
        self.duration = None

        # Call counters for assertions
        self.start_count = 0
        self.stop_count = 0
        self.push_sample_count = 0

    def start(self, save_fn, duration):
        """Start EEG recording."""
        self.started = True
        self.save_fn = save_fn
        self.duration = duration
        self.start_count += 1

    def push_sample(self, marker, timestamp):
        """Push a stimulus marker to the EEG stream."""
        self.markers.append({
            'marker': marker,
            'timestamp': timestamp
        })
        self.push_sample_count += 1

    def stop(self):
        """Stop EEG recording."""
        self.started = False
        self.stopped = True
        self.stop_count += 1

    def get_recent(self, n_samples=256):
        """Get recent EEG data samples (synthetic)."""
        return np.random.randn(n_samples, self.n_channels)

    def reset(self):
        """Reset the mock state for reuse in tests."""
        self.started = False
        self.stopped = False
        self.markers = []
        self.save_fn = None
        self.duration = None
        self.start_count = 0
        self.stop_count = 0
        self.push_sample_count = 0


class MockWindow:
    """
    Mock PsychoPy Window for headless testing.

    Simulates window operations without requiring a display.
    """

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.mouseVisible = True
        self.size = kwargs.get('size', [1600, 800])
        self.fullscr = kwargs.get('fullscr', False)
        self.screen = kwargs.get('screen', 0)
        self.units = kwargs.get('units', 'height')
        self.color = kwargs.get('color', 'black')

        # Track operations
        self.flip_count = 0

    def flip(self):
        """Flip the window buffer."""
        if not self.closed:
            self.flip_count += 1

    def close(self):
        """Close the window."""
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MockImageStim:
    """Mock PsychoPy ImageStim for visual stimulus testing."""

    def __init__(self, win, image=None, **kwargs):
        self.win = win
        self.image = image
        self.size = kwargs.get('size', None)
        self.pos = kwargs.get('pos', (0, 0))
        self.opacity = kwargs.get('opacity', 1.0)

        self.draw_count = 0

    def draw(self):
        """Draw the image stimulus."""
        self.draw_count += 1

    def setImage(self, image):
        """Set a new image."""
        self.image = image

    def setOpacity(self, opacity):
        """Set stimulus opacity."""
        self.opacity = opacity


class MockTextStim:
    """Mock PsychoPy TextStim for text display testing."""

    def __init__(self, win, text='', **kwargs):
        self.win = win
        self.text = text
        self.height = kwargs.get('height', 0.1)
        self.pos = kwargs.get('pos', (0, 0))
        self.color = kwargs.get('color', 'white')
        self.wrapWidth = kwargs.get('wrapWidth', None)

        self.draw_count = 0

    def draw(self):
        """Draw the text stimulus."""
        self.draw_count += 1

    def setText(self, text):
        """Update text content."""
        self.text = text


class MockClock:
    """Mock PsychoPy Clock for timing control in tests."""

    def __init__(self):
        self.time = 0.0
        self.reset_count = 0

    def getTime(self):
        """Get current time."""
        return self.time

    def reset(self):
        """Reset clock to zero."""
        self.time = 0.0
        self.reset_count += 1

    def add(self, seconds):
        """Manually advance time (for testing)."""
        self.time += seconds


# Global fixtures

@pytest.fixture
def mock_eeg():
    """Fixture providing a fresh MockEEG instance for each test."""
    return MockEEG()


@pytest.fixture
def mock_eeg_muse2():
    """Fixture providing a Muse2-specific mock EEG device."""
    eeg = MockEEG(device_name="muse2")
    eeg.channels = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
    eeg.n_channels = 5
    return eeg


@pytest.fixture
def temp_save_fn(tmp_path):
    """Fixture providing a temporary file path for test recordings."""
    return str(tmp_path / "test_n170_recording")


@pytest.fixture
def mock_psychopy(mocker):
    """
    Fixture that mocks all PsychoPy components for headless testing.

    Returns a dictionary with references to all mocked components
    for assertion and control in tests.
    """
    # Mock window
    mock_window = mocker.patch('psychopy.visual.Window', MockWindow)

    # Mock visual stimuli
    mock_image = mocker.patch('psychopy.visual.ImageStim', MockImageStim)
    mock_text = mocker.patch('psychopy.visual.TextStim', MockTextStim)

    # Mock event system - return empty by default (no keys pressed)
    mock_keys = mocker.patch('psychopy.event.getKeys')
    mock_keys.return_value = []

    # Mock core timing
    mock_wait = mocker.patch('psychopy.core.wait')
    mock_clock_class = mocker.patch('psychopy.core.Clock', MockClock)

    # Mock mouse
    mock_mouse = mocker.patch('psychopy.event.Mouse')

    return {
        'Window': mock_window,
        'ImageStim': mock_image,
        'TextStim': mock_text,
        'get_keys': mock_keys,
        'wait': mock_wait,
        'Clock': mock_clock_class,
        'Mouse': mock_mouse,
    }


@pytest.fixture
def mock_psychopy_with_spacebar(mock_psychopy):
    """
    Fixture that mocks PsychoPy with automatic spacebar press.

    Useful for tests that need to start the experiment automatically.
    """
    # First call returns empty, second returns space, then escape
    mock_psychopy['get_keys'].side_effect = [
        [],           # Initial call
        ['space'],    # Start experiment
        [],           # During experiment
        ['escape']    # End experiment
    ] * 50  # Repeat pattern for multiple calls

    return mock_psychopy


@pytest.fixture
def mock_vr_disabled(mocker):
    """Fixture to disable VR input for tests."""
    # Patch the BaseExperiment.get_vr_input method to always return False
    mock = mocker.patch('eegnb.experiments.Experiment.BaseExperiment.get_vr_input')
    mock.return_value = False
    return mock


@pytest.fixture
def mock_vr_button_press(mocker):
    """Fixture to simulate VR controller button press."""
    mock = mocker.patch('eegnb.experiments.Experiment.BaseExperiment.get_vr_input')
    # First call False, second True (button press), then False again
    mock.side_effect = [False, True, False] * 50
    return mock


@pytest.fixture
def stimulus_images(tmp_path):
    """
    Fixture providing mock stimulus image files for testing.

    Creates a temporary directory structure with dummy face and house images.
    """
    stim_dir = tmp_path / "stimuli" / "visual" / "face_house"
    stim_dir.mkdir(parents=True)

    # Create dummy image files (we'll just create empty files for testing)
    # In real tests with image loading, you'd create actual small test images
    faces = []
    houses = []

    for i in range(3):
        face_file = stim_dir / f"face_{i:02d}.jpg"
        house_file = stim_dir / f"house_{i:02d}.jpg"

        face_file.touch()
        house_file.touch()

        faces.append(str(face_file))
        houses.append(str(house_file))

    return {
        'dir': str(stim_dir),
        'faces': faces,
        'houses': houses
    }


# Pytest configuration hooks

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers",
        "requires_display: mark test as requiring a display (skip in CI)"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib settings after each test."""
    yield
    # Cleanup after test
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass
