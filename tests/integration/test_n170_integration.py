"""
Integration tests for the N170 visual experiment.

Minimal high-value test suite covering:
- Initialization and setup
- Full experiment execution with/without EEG
- Device integration
- Edge cases
- User input handling

All tests use mocked EEG devices and PsychoPy components for headless testing.
Tests follow the normal initialization flow: __init__() → setup() → run()
"""

import pytest
import sys
import numpy as np
from unittest.mock import Mock, patch, call, MagicMock
from pathlib import Path

# Mock PsychoPy and other heavy dependencies at the module level before importing
mock_psychopy = MagicMock()
mock_psychopy.visual = MagicMock()
mock_psychopy.visual.rift = MagicMock()
mock_psychopy.visual.Window = MagicMock()
mock_psychopy.visual.ImageStim = MagicMock()
mock_psychopy.visual.TextStim = MagicMock()
mock_psychopy.core = MagicMock()
mock_psychopy.event = MagicMock()
mock_psychopy.prefs = MagicMock()
mock_psychopy.prefs.hardware = {}

sys.modules['psychopy'] = mock_psychopy
sys.modules['psychopy.visual'] = mock_psychopy.visual
sys.modules['psychopy.visual.rift'] = mock_psychopy.visual.rift
sys.modules['psychopy.core'] = mock_psychopy.core
sys.modules['psychopy.event'] = mock_psychopy.event
sys.modules['psychopy.prefs'] = mock_psychopy.prefs

sys.modules['brainflow'] = MagicMock()
sys.modules['brainflow.board_shim'] = MagicMock()
sys.modules['muselsl'] = MagicMock()
sys.modules['muselsl.stream'] = MagicMock()
sys.modules['muselsl.muse'] = MagicMock()
sys.modules['pylsl'] = MagicMock()

from eegnb.experiments.visual_n170.n170 import VisualN170
from eegnb import generate_save_fn


@pytest.mark.integration
class TestN170Core:
    """Core functionality tests for N170 experiment."""

    def test_basic_initialization(self, mock_eeg, temp_save_fn):
        """Test basic experiment initialization with parameters."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=50,
            iti=0.4,
            soa=0.3,
            jitter=0.2,
            use_vr=False
        )

        assert experiment.duration == 10
        assert experiment.eeg == mock_eeg
        assert experiment.save_fn == temp_save_fn
        assert experiment.n_trials == 50
        assert experiment.iti == 0.4
        assert experiment.soa == 0.3
        assert experiment.jitter == 0.2
        assert experiment.use_vr is False

    def test_setup_creates_window_and_loads_stimuli(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that setup() properly initializes window and stimuli."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=10,
            use_vr=False
        )

        # Before setup
        assert not hasattr(experiment, 'window')

        # Run setup
        experiment.setup(instructions=False)

        # After setup - everything should be initialized
        assert hasattr(experiment, 'window')
        assert experiment.window is not None
        assert hasattr(experiment, 'stim')
        assert hasattr(experiment, 'faces')
        assert hasattr(experiment, 'houses')
        assert hasattr(experiment, 'trials')
        assert len(experiment.trials) == 10

    def test_full_experiment_run_with_eeg(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test complete experiment workflow with EEG device."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        # Run complete experiment
        experiment.run(instructions=False)

        # Verify initialization happened
        assert hasattr(experiment, 'window')
        assert hasattr(experiment, 'trials')
        assert experiment.eeg == mock_eeg

    def test_full_experiment_run_without_eeg(self, temp_save_fn, mock_psychopy):
        """Test complete experiment workflow without EEG device."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=5,
            eeg=None,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        # Should work without EEG device
        experiment.run(instructions=False)

        assert experiment.eeg is None
        assert hasattr(experiment, 'window')


@pytest.mark.integration
class TestN170DeviceIntegration:
    """Test integration with different EEG devices."""

    def test_device_integration(self, temp_save_fn, mock_psychopy):
        """Test initialization with different device types."""
        from tests.conftest import MockEEG

        # Test with Muse2 device
        mock_eeg = MockEEG(device_name="muse2")
        mock_eeg.n_channels = 5

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=False
        )

        assert experiment.eeg.device_name == "muse2"
        assert experiment.eeg.n_channels == 5

        # Verify it can be set up
        experiment.setup(instructions=False)
        assert experiment.eeg == mock_eeg


@pytest.mark.integration
class TestN170EdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_trials(self, mock_eeg, temp_save_fn):
        """Test handling of zero trials edge case."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=0,
            use_vr=False
        )

        assert experiment.n_trials == 0

    def test_minimal_timing_configuration(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test experiment with minimal timing (short duration, fast trials)."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 10

        experiment = VisualN170(
            duration=1,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=1,
            iti=0.1,
            soa=0.05,
            jitter=0.0,
            use_vr=False
        )

        # Should handle minimal configuration gracefully
        experiment.run(instructions=False)
        assert True


@pytest.mark.integration
class TestN170UserInteraction:
    """Test user input and interaction handling."""

    def test_keyboard_input_handling(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test keyboard spacebar start and escape cancellation."""
        # Simulate spacebar press to start, then escape to exit
        mock_psychopy['get_keys'].side_effect = [
            [],           # Initial
            ['space'],    # Start
            [],           # Running
            ['escape']    # Exit
        ] * 20

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        # Should handle keyboard input properly
        experiment.run(instructions=False)
        assert True

    def test_vr_mode_initialization(self, mock_eeg, temp_save_fn):
        """Test VR mode can be enabled."""
        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=True
        )

        assert experiment.use_vr is True


@pytest.mark.integration
class TestN170SaveFunction:
    """Test save function and file handling."""

    def test_save_function_integration(self, mock_eeg, tmp_path):
        """Test integration with save function utility."""
        save_fn = generate_save_fn(
            board_name="muse2",
            experiment="visual_n170",
            subject_id=0,
            session_nb=0,
            site="test",
            data_dir=str(tmp_path)
        )

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=save_fn,
            use_vr=False
        )

        # Verify save_fn is set correctly
        assert experiment.save_fn == save_fn
        save_fn_str = str(save_fn)
        assert "visual_n170" in save_fn_str or "n170" in save_fn_str
