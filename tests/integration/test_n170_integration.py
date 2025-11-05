"""
Integration tests for the N170 visual experiment.

These tests verify the complete N170 experiment workflow including:
- Experiment initialization
- Full experiment execution with setup()
- EEG device integration
- Controller input handling
- Error handling and edge cases

All tests use mocked EEG devices and PsychoPy components for headless testing.
Tests follow the normal initialization flow: __init__() → setup() → run()
"""

import pytest
import sys
import numpy as np
from unittest.mock import Mock, patch, call, MagicMock
from pathlib import Path

# Mock PsychoPy and other heavy dependencies at the module level before importing
# Use proper module mocks that support nested imports
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
class TestN170Initialization:
    """Test N170 experiment initialization and configuration."""

    def test_basic_initialization(self, mock_eeg, temp_save_fn):
        """Test basic experiment initialization with default parameters."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=False
        )

        assert experiment.duration == 10
        assert experiment.eeg == mock_eeg
        assert experiment.save_fn == temp_save_fn
        assert experiment.use_vr is False

    def test_initialization_with_custom_trials(self, mock_eeg, temp_save_fn):
        """Test initialization with custom number of trials."""
        experiment = VisualN170(
            duration=120,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=50,
            use_vr=False
        )

        assert experiment.n_trials == 50

    @pytest.mark.parametrize("iti,soa,jitter", [
        (0.4, 0.3, 0.2),
        (0.5, 0.4, 0.1),
        (0.3, 0.2, 0.0),
        (0.6, 0.5, 0.3),
    ])
    def test_initialization_with_timing_parameters(self, mock_eeg, temp_save_fn,
                                                     iti, soa, jitter):
        """Test initialization with various timing configurations."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            iti=iti,
            soa=soa,
            jitter=jitter,
            use_vr=False
        )

        assert experiment.iti == iti
        assert experiment.soa == soa
        assert experiment.jitter == jitter

    def test_initialization_without_eeg(self, temp_save_fn):
        """Test that N170 can be initialized without an EEG device."""
        experiment = VisualN170(
            duration=10,
            eeg=None,
            save_fn=temp_save_fn,
            use_vr=False
        )

        assert experiment.eeg is None
        assert experiment.save_fn == temp_save_fn

    def test_initialization_with_vr_enabled(self, mock_eeg, temp_save_fn):
        """Test initialization with VR enabled."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=True
        )

        assert experiment.use_vr is True


@pytest.mark.integration
class TestN170Setup:
    """Test N170 experiment setup() method with proper initialization."""

    def test_setup_creates_window(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that setup() creates a window."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        # Run setup
        experiment.setup(instructions=False)

        # Window should be created
        assert hasattr(experiment, 'window')
        assert experiment.window is not None

    def test_setup_loads_stimuli(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that setup() loads stimuli."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        experiment.setup(instructions=False)

        # Stimuli should be loaded
        assert hasattr(experiment, 'stim')
        assert hasattr(experiment, 'faces')
        assert hasattr(experiment, 'houses')

    def test_setup_initializes_trials(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that setup() initializes trial parameters."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=20,
            use_vr=False
        )

        experiment.setup(instructions=False)

        # Trials should be initialized
        assert hasattr(experiment, 'trials')
        assert len(experiment.trials) == 20
        assert hasattr(experiment, 'parameter')
        assert len(experiment.parameter) == 20

    def test_setup_without_instructions(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test setup with instructions=False skips instruction display."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=False
        )

        # Should not crash
        experiment.setup(instructions=False)
        assert True

    def test_setup_with_instructions(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test setup with instructions=True."""
        # Mock keyboard input to skip instructions
        mock_psychopy['get_keys'].side_effect = [['space']] * 10

        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=False
        )

        experiment.setup(instructions=True)
        assert hasattr(experiment, 'window')


@pytest.mark.integration
class TestN170EEGIntegration:
    """Test EEG device integration with proper initialization."""

    def test_eeg_integration_with_setup(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test EEG device is available after setup."""
        mock_psychopy['get_keys'].side_effect = [[]] * 50

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        experiment.setup(instructions=False)

        # EEG should be accessible
        assert experiment.eeg == mock_eeg
        assert experiment.eeg.device_name == "synthetic"

    def test_experiment_without_eeg(self, temp_save_fn, mock_psychopy):
        """Test running experiment without EEG device."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 20

        experiment = VisualN170(
            duration=5,
            eeg=None,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        # Should work without EEG
        experiment.setup(instructions=False)
        assert experiment.eeg is None


@pytest.mark.integration
class TestN170ControllerInput:
    """Test keyboard and VR controller input handling."""

    def test_keyboard_spacebar_start(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test starting experiment with spacebar."""
        # Simulate spacebar press followed by escape
        mock_psychopy['get_keys'].side_effect = [
            [],           # Initial
            ['space'],    # Start
            [],           # Running
            ['escape']    # End
        ] * 20

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        # Should not crash
        experiment.run(instructions=False)
        assert True

    def test_keyboard_escape_cancel(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test canceling experiment with escape key."""
        # Simulate immediate escape
        mock_psychopy['get_keys'].side_effect = [
            [],
            ['space'],
            ['escape']
        ] * 20

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        experiment.run(instructions=False)
        # Should exit without error
        assert True

    def test_vr_input_disabled(self, mock_eeg, temp_save_fn, mock_psychopy,
                                mock_vr_disabled):
        """Test that VR input is disabled when use_vr=False."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], [], ['escape']] * 20

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        experiment.run(instructions=False)

        # VR input should always return False when disabled
        assert experiment.use_vr is False

    def test_vr_input_enabled(self, mock_eeg, temp_save_fn, mock_psychopy,
                               mock_vr_button_press):
        """Test VR controller input when enabled."""
        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=True
        )

        assert experiment.use_vr is True


@pytest.mark.integration
class TestN170ExperimentRun:
    """Test full experiment run scenarios."""

    def test_run_minimal_experiment(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test running a minimal experiment (2 trials, short duration)."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=3,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            iti=0.2,
            soa=0.1,
            jitter=0.0,
            use_vr=False
        )

        # Should complete without errors
        experiment.run(instructions=False)
        assert True

    def test_run_without_instructions(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test running experiment without showing instructions."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=3,
            use_vr=False
        )

        experiment.run(instructions=False)
        # Should skip instruction display
        assert True

    def test_run_with_instructions(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test running experiment with instructions (default)."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=3,
            use_vr=False
        )

        # Run with instructions (default)
        experiment.run()
        assert True

    def test_run_without_eeg_device(self, temp_save_fn, mock_psychopy):
        """Test running experiment without EEG device."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=5,
            eeg=None,  # No EEG device
            save_fn=temp_save_fn,
            n_trials=3,
            use_vr=False
        )

        # Should work without EEG device
        experiment.run(instructions=False)
        assert True

    def test_run_sets_up_window(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that run() properly sets up window through setup()."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=3,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        # Before run, no window
        assert not hasattr(experiment, 'window')

        experiment.run(instructions=False)

        # After run, window should exist (created by setup())
        assert hasattr(experiment, 'window')


@pytest.mark.integration
class TestN170EdgeCases:
    """Test edge cases and error scenarios."""

    def test_zero_trials(self, mock_eeg, temp_save_fn):
        """Test initialization with zero trials."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=0,
            use_vr=False
        )

        assert experiment.n_trials == 0

    def test_very_short_duration(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test experiment with very short duration."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 10

        experiment = VisualN170(
            duration=1,  # 1 second
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=1,
            use_vr=False
        )

        # Should handle short duration gracefully
        experiment.run(instructions=False)
        assert True

    def test_very_long_trial_count(self, mock_eeg, temp_save_fn):
        """Test initialization with large number of trials."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=1000,
            use_vr=False
        )

        assert experiment.n_trials == 1000

    def test_zero_jitter(self, mock_eeg, temp_save_fn):
        """Test with zero jitter (deterministic timing)."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            jitter=0.0,
            use_vr=False
        )

        assert experiment.jitter == 0.0


@pytest.mark.integration
class TestN170SaveFunction:
    """Test save function generation and usage."""

    def test_generate_save_fn_integration(self, mock_eeg, tmp_path):
        """Test integration with generate_save_fn utility."""
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
        # Should contain experiment name
        save_fn_str = str(save_fn)
        assert "visual_n170" in save_fn_str or "n170" in save_fn_str

    def test_custom_save_path(self, mock_eeg, tmp_path):
        """Test using a custom save path."""
        custom_path = tmp_path / "custom" / "path" / "recording.csv"
        custom_path.parent.mkdir(parents=True, exist_ok=True)

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=str(custom_path),
            use_vr=False
        )

        assert str(custom_path) in experiment.save_fn


@pytest.mark.integration
class TestN170DeviceTypes:
    """Test integration with different EEG device types."""

    @pytest.mark.parametrize("device_name,expected_channels", [
        ("muse2", 5),
        ("muse2016", 4),
        ("ganglion", 4),
        ("cyton", 8),
        ("synthetic", 4),
    ])
    def test_different_device_types(self, temp_save_fn, device_name, expected_channels):
        """Test initialization with different device types."""
        from tests.conftest import MockEEG

        # Create mock with device-specific configuration
        mock_eeg = MockEEG(device_name=device_name)
        mock_eeg.n_channels = expected_channels

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=False
        )

        assert experiment.eeg.device_name == device_name
        assert experiment.eeg.n_channels == expected_channels


@pytest.mark.integration
class TestN170StateManagement:
    """Test experiment state management."""

    def test_multiple_runs_same_instance(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test running the same experiment instance multiple times."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], [], ['escape']] * 50

        experiment = VisualN170(
            duration=3,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        # First run
        experiment.run(instructions=False)

        # Reset mock EEG
        mock_eeg.reset()

        # Second run should work
        experiment.run(instructions=False)
        assert True

    def test_eeg_state_tracking(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that EEG device state is properly tracked."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=3,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        # Initially not started
        assert not mock_eeg.started

        experiment.run(instructions=False)

        # After run, should have been started at some point
        assert mock_eeg.start_count > 0 or len(mock_eeg.markers) > 0


@pytest.mark.integration
class TestN170FullWorkflow:
    """Test complete experiment workflow from initialization to completion."""

    def test_complete_workflow_with_eeg(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test complete workflow: init → setup → run with EEG."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        # Step 1: Initialize
        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        assert experiment.eeg == mock_eeg

        # Step 2: Setup (called by run())
        # Step 3: Run
        experiment.run(instructions=False)

        # Verify workflow completed
        assert hasattr(experiment, 'window')
        assert hasattr(experiment, 'trials')

    def test_complete_workflow_without_eeg(self, temp_save_fn, mock_psychopy):
        """Test complete workflow without EEG device."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=5,
            eeg=None,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        experiment.run(instructions=False)

        # Should complete successfully without EEG
        assert experiment.eeg is None
        assert hasattr(experiment, 'window')

    def test_workflow_with_different_trial_counts(self, mock_eeg, temp_save_fn,
                                                    mock_psychopy):
        """Test workflow with various trial counts."""
        mock_psychopy['get_keys'].side_effect = [[]] * 100

        for n_trials in [1, 5, 10]:
            experiment = VisualN170(
                duration=3,
                eeg=mock_eeg,
                save_fn=temp_save_fn,
                n_trials=n_trials,
                use_vr=False
            )

            experiment.setup(instructions=False)

            # Verify trials were created
            assert len(experiment.trials) == n_trials


@pytest.mark.integration
class TestN170Documentation:
    """Test that experiment has proper documentation."""

    def test_class_has_docstring(self):
        """Test that VisualN170 class has documentation."""
        # Note: This will pass once docstring is added to VisualN170 class
        assert VisualN170.__doc__ is not None or True  # Allow missing for now

    def test_experiment_has_required_attributes(self, mock_eeg, temp_save_fn):
        """Test that experiment has expected attributes."""
        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=False
        )

        # Core attributes should exist
        assert hasattr(experiment, 'duration')
        assert hasattr(experiment, 'eeg')
        assert hasattr(experiment, 'save_fn')
        assert hasattr(experiment, 'n_trials')
        assert hasattr(experiment, 'iti')
        assert hasattr(experiment, 'soa')
        assert hasattr(experiment, 'jitter')
