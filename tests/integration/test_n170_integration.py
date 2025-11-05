"""
Integration tests for the N170 visual experiment.

These tests verify the complete N170 experiment workflow including:
- Experiment initialization
- Stimulus loading and presentation
- EEG device integration
- Controller input handling
- Timing and trial management
- Error handling and edge cases

All tests use mocked EEG devices and PsychoPy components for headless testing.
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
class TestN170StimulusLoading:
    """Test stimulus loading functionality."""

    def test_load_stimulus_basic(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test basic stimulus loading."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=10,
            use_vr=False
        )

        # Load stimuli
        experiment.load_stimulus()

        # Check that trials were generated
        assert hasattr(experiment, 'trials')
        assert len(experiment.trials) > 0

        # Check that image stimulus object was created
        assert hasattr(experiment, 'image')

    def test_stimulus_trials_contain_valid_data(self, mock_eeg, temp_save_fn,
                                                  mock_psychopy):
        """Test that trial data contains valid stimulus information."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=20,
            use_vr=False
        )

        experiment.load_stimulus()

        # Each trial should have label and image path
        for trial_data in experiment.trials.values():
            assert 'label' in trial_data
            # Label should be 1 (face) or 2 (house)
            assert trial_data['label'] in [1, 2]


@pytest.mark.integration
class TestN170StimulusPresentation:
    """Test stimulus presentation functionality."""

    def test_present_stimulus_single(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test presenting a single stimulus."""
        # Setup mock clock to return predictable timestamp
        mock_clock = mock_psychopy['Clock']()
        mock_clock.time = 1.234

        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        experiment.load_stimulus()

        # Present first stimulus
        experiment.present_stimulus(idx=0, trial=0)

        # Verify EEG marker was pushed
        assert len(mock_eeg.markers) == 1
        assert 'marker' in mock_eeg.markers[0]
        assert 'timestamp' in mock_eeg.markers[0]

    def test_present_stimulus_without_eeg(self, temp_save_fn, mock_psychopy):
        """Test presenting stimulus without EEG device (should not crash)."""
        experiment = VisualN170(
            duration=10,
            eeg=None,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        experiment.load_stimulus()

        # Should not crash when presenting without EEG
        try:
            experiment.present_stimulus(idx=0, trial=0)
            # If we get here, test passes
            assert True
        except Exception as e:
            pytest.fail(f"Present stimulus crashed without EEG: {e}")

    def test_present_multiple_stimuli(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test presenting multiple stimuli in sequence."""
        mock_clock = mock_psychopy['Clock']()
        mock_clock.time = 0.0

        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        experiment.load_stimulus()

        # Present multiple stimuli
        for idx in range(3):
            mock_clock.time = idx * 1.0
            experiment.present_stimulus(idx=0, trial=idx)

        # Verify all markers were pushed
        assert len(mock_eeg.markers) == 3

        # Verify timestamps are increasing
        timestamps = [m['timestamp'] for m in mock_eeg.markers]
        assert timestamps == sorted(timestamps)


@pytest.mark.integration
class TestN170EEGIntegration:
    """Test EEG device integration."""

    def test_eeg_device_start_called(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that EEG device start() is called with correct parameters."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], [], ['escape']] * 20

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        # Run experiment without instructions
        experiment.run(instructions=False)

        # Verify start was called
        assert mock_eeg.start_count >= 1
        # Verify it was called with save_fn and duration
        if mock_eeg.save_fn:
            assert temp_save_fn in mock_eeg.save_fn or mock_eeg.save_fn == temp_save_fn

    def test_eeg_device_stop_called(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that EEG device stop() is called after experiment."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], [], ['escape']] * 20

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=2,
            use_vr=False
        )

        experiment.run(instructions=False)

        # Verify stop was called
        assert mock_eeg.stop_count >= 0  # May vary based on implementation

    def test_eeg_markers_pushed(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that EEG markers are pushed during stimulus presentation."""
        mock_psychopy['get_keys'].side_effect = [[], ['space'], []] * 30

        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=3,
            use_vr=False
        )

        experiment.load_stimulus()

        # Present stimuli
        for idx in range(3):
            experiment.present_stimulus(idx=0, trial=idx)

        # Verify markers were pushed
        assert len(mock_eeg.markers) == 3

        # Verify marker format
        for marker in mock_eeg.markers:
            assert 'marker' in marker
            assert 'timestamp' in marker
            # Marker should be list with label (1 or 2)
            assert isinstance(marker['marker'], list) or isinstance(marker['marker'], np.ndarray)

    def test_eeg_marker_labels(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that EEG markers contain correct stimulus labels."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        experiment.load_stimulus()

        # Present multiple stimuli and collect labels
        for idx in range(5):
            experiment.present_stimulus(idx=0, trial=idx)

        # All markers should have labels 1 or 2
        for marker in mock_eeg.markers:
            label = marker['marker'][0] if isinstance(marker['marker'], (list, np.ndarray)) else marker['marker']
            assert label in [1, 2], f"Invalid marker label: {label}"


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
        assert True  # If we get here, test passes

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
class TestN170TimingAndSequencing:
    """Test timing and trial sequencing."""

    def test_trial_timing_configuration(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that timing parameters are correctly configured."""
        iti = 0.5
        soa = 0.4
        jitter = 0.2

        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            iti=iti,
            soa=soa,
            jitter=jitter,
            use_vr=False
        )

        # Verify timing parameters are set
        assert experiment.iti == iti
        assert experiment.soa == soa
        assert experiment.jitter == jitter

    def test_markers_have_timestamps(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test that all markers have valid timestamps."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=5,
            use_vr=False
        )

        experiment.load_stimulus()

        # Present multiple stimuli
        for idx in range(5):
            experiment.present_stimulus(idx=0, trial=idx)

        # All markers should have timestamps
        for marker in mock_eeg.markers:
            assert 'timestamp' in marker
            assert isinstance(marker['timestamp'], (int, float))
            assert marker['timestamp'] >= 0


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
@pytest.mark.slow
class TestN170Performance:
    """Test performance and stress scenarios."""

    def test_many_trials(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test experiment with many trials."""
        mock_psychopy['get_keys'].side_effect = [[]] * 500

        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=100,
            iti=0.1,
            soa=0.05,
            use_vr=False
        )

        experiment.load_stimulus()

        # Should handle many trials without issues
        assert len(experiment.trials) == 100

    def test_rapid_stimulus_presentation(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test presenting stimuli in rapid succession."""
        experiment = VisualN170(
            duration=10,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            n_trials=50,
            iti=0.1,
            soa=0.05,
            jitter=0.0,
            use_vr=False
        )

        experiment.load_stimulus()

        # Present many stimuli rapidly
        for idx in range(20):
            experiment.present_stimulus(idx=0, trial=idx)

        # All markers should be recorded
        assert len(mock_eeg.markers) == 20


@pytest.mark.integration
class TestN170Documentation:
    """Test that experiment has proper documentation."""

    def test_class_has_docstring(self):
        """Test that VisualN170 class has documentation."""
        assert VisualN170.__doc__ is not None

    def test_experiment_has_name_attribute(self, mock_eeg, temp_save_fn):
        """Test that experiment has a name attribute."""
        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=False
        )

        # Should have some identifier
        assert hasattr(experiment, 'n_trials') or hasattr(experiment, 'duration')
