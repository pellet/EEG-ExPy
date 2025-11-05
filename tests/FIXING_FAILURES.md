# Fixing N170 Integration Test Failures

## Current Status

**Test Results**: 33/44 tests passing (75% pass rate)
**Failing Tests**: 11 tests

## Root Causes Analysis

### Issue #1: Window Not Initialized (10 tests failing)

**Problem**: Tests call `load_stimulus()` or `present_stimulus()` directly without initializing the window.

**Technical Details**:
- The `self.window` attribute is created in `BaseExperiment.setup()` (line 92-94 in Experiment.py)
- `setup()` is normally called by `run()` before `load_stimulus()` is invoked
- Tests are calling these methods directly, bypassing the normal initialization flow

**Stack Trace Example**:
```
AttributeError: 'VisualN170' object has no attribute 'window'
    at eegnb/experiments/visual_n170/n170.py:35
    in load_image = lambda fn: visual.ImageStim(win=self.window, image=fn)
```

**Affected Tests**:
1. `TestN170StimulusLoading::test_load_stimulus_basic`
2. `TestN170StimulusLoading::test_stimulus_trials_contain_valid_data`
3. `TestN170StimulusPresentation::test_present_stimulus_single`
4. `TestN170StimulusPresentation::test_present_stimulus_without_eeg`
5. `TestN170StimulusPresentation::test_present_multiple_stimuli`
6. `TestN170EEGIntegration::test_eeg_markers_pushed`
7. `TestN170EEGIntegration::test_eeg_marker_labels`
8. `TestN170TimingAndSequencing::test_markers_have_timestamps`
9. `TestN170Performance::test_many_trials`
10. `TestN170Performance::test_rapid_stimulus_presentation`

### Issue #2: Missing Class Docstring (1 test failing)

**Problem**: The `VisualN170` class has no docstring.

**Current Code** (n170.py:21):
```python
class VisualN170(Experiment.BaseExperiment):

    def __init__(self, duration=120, eeg: Optional[EEG]=None, save_fn=None,
```

**Affected Tests**:
1. `TestN170Documentation::test_class_has_docstring`

## Solutions

### Solution 1: Call setup() in Tests (Recommended for Quick Fix)

**Approach**: Modify tests to call `setup()` before using stimulus methods.

**Implementation**:

```python
def test_load_stimulus_basic(self, mock_eeg, temp_save_fn, mock_psychopy):
    """Test basic stimulus loading."""
    experiment = VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=10,
        use_vr=False
    )

    # Initialize window and load stimuli
    experiment.setup(instructions=False)  # <-- ADD THIS LINE

    # Now load_stimulus has been called by setup()
    assert hasattr(experiment, 'trials')
    assert len(experiment.trials) > 0
    assert hasattr(experiment, 'image')
```

**Pros**:
- Simple fix
- Tests real initialization flow
- Minimal changes to test code

**Cons**:
- Tests become less granular (testing multiple steps together)
- Relies on full setup process

### Solution 2: Create Window Manually in Tests (More Granular)

**Approach**: Manually create the window in tests that need it.

**Implementation**:

```python
def test_load_stimulus_basic(self, mock_eeg, temp_save_fn, mock_psychopy):
    """Test basic stimulus loading."""
    experiment = VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=10,
        use_vr=False
    )

    # Manually create window for testing
    from tests.conftest import MockWindow
    experiment.window = MockWindow()  # <-- ADD THIS

    # Also need to initialize trials
    experiment.parameter = np.random.binomial(1, 0.5, experiment.n_trials)
    experiment.trials = DataFrame(dict(
        parameter=experiment.parameter,
        timestamp=np.zeros(experiment.n_trials)
    ))

    # Now we can test load_stimulus
    experiment.load_stimulus()

    assert hasattr(experiment, 'trials')
    assert hasattr(experiment, 'faces')
    assert hasattr(experiment, 'houses')
```

**Pros**:
- More granular testing
- Can test specific initialization steps
- More control over test setup

**Cons**:
- More test code
- Duplicates initialization logic
- May miss integration issues

### Solution 3: Add Helper Fixture (Best for Reusability)

**Approach**: Create a fixture that returns a fully initialized experiment.

**Implementation in conftest.py**:

```python
@pytest.fixture
def initialized_n170_experiment(mock_eeg, temp_save_fn, mock_psychopy):
    """Fixture providing a fully initialized N170 experiment."""
    experiment = VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=10,
        use_vr=False
    )

    # Initialize with setup
    experiment.setup(instructions=False)

    return experiment
```

**Usage in tests**:

```python
def test_load_stimulus_basic(self, initialized_n170_experiment):
    """Test basic stimulus loading."""
    experiment = initialized_n170_experiment

    # Stimulus already loaded by setup()
    assert hasattr(experiment, 'stim')
    assert hasattr(experiment, 'faces')
    assert hasattr(experiment, 'houses')
```

**Pros**:
- DRY principle (Don't Repeat Yourself)
- Easy to use across multiple tests
- Consistent initialization

**Cons**:
- Less explicit about what's being tested
- May make tests less independent

### Solution 4: Mock the Window Attribute (Quick Fix for Mocking)

**Approach**: Mock `self.window` in tests without full initialization.

**Implementation**:

```python
def test_load_stimulus_basic(self, mock_eeg, temp_save_fn, mock_psychopy, mocker):
    """Test basic stimulus loading."""
    experiment = VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=10,
        use_vr=False
    )

    # Mock the window attribute
    from tests.conftest import MockWindow
    experiment.window = MockWindow()

    # Mock the stimulus file paths
    mocker.patch('glob.glob', return_value=[
        '/fake/path/face_1.jpg',
        '/fake/path/face_2.jpg',
    ])

    # Now load_stimulus should work
    experiment.load_stimulus()

    assert hasattr(experiment, 'faces')
```

**Pros**:
- Quick fix
- Can test in isolation
- Good for unit testing

**Cons**:
- More mocking = further from real behavior
- May miss real integration issues

### Solution 5: Fix the Missing Docstring

**Approach**: Add a docstring to the VisualN170 class.

**Implementation**:

Edit `eegnb/experiments/visual_n170/n170.py`:

```python
class VisualN170(Experiment.BaseExperiment):
    """
    Visual N170 oddball experiment.

    Presents faces and houses in a random sequence to elicit the N170
    event-related potential (ERP) component. The N170 is a negative deflection
    in the EEG signal occurring approximately 170ms after stimulus onset,
    with larger amplitude for faces compared to other visual stimuli.

    Parameters
    ----------
    duration : int, optional
        Duration of the recording in seconds (default: 120)
    eeg : EEG, optional
        EEG device instance for recording (default: None)
    save_fn : str, optional
        Path to save the recording data (default: None)
    n_trials : int, optional
        Number of trials to present (default: 2010)
    iti : float, optional
        Inter-trial interval in seconds (default: 0.4)
    soa : float, optional
        Stimulus onset asynchrony in seconds (default: 0.3)
    jitter : float, optional
        Random jitter added to timing in seconds (default: 0.2)
    use_vr : bool, optional
        Whether to use VR display (default: False)

    References
    ----------
    Bentin, S., et al. (1996). Electrophysiological studies of face perception
    in humans. Journal of Cognitive Neuroscience, 8(6), 551-565.
    """

    def __init__(self, duration=120, eeg: Optional[EEG]=None, save_fn=None,
```

**Pros**:
- Simple fix
- Improves code documentation
- Makes the test pass

**Cons**:
- None (this should always be done!)

## Recommended Fix Strategy

### Phase 1: Quick Wins (Immediate)

1. **Add class docstring** (fixes 1 test)
   - Edit n170.py line 21
   - Add comprehensive docstring
   - Takes 2 minutes

2. **Add initialized_n170_experiment fixture** (fixes 10 tests)
   - Add to conftest.py
   - Handles window and stimulus initialization
   - Update failing tests to use fixture

### Phase 2: Refactor Tests (Follow-up)

1. **Reorganize test structure**
   - Separate unit tests (with mocks) from integration tests
   - Use fixtures for common setup
   - Document test prerequisites

2. **Add more specific fixtures**
   - `experiment_with_window` - Just window initialized
   - `experiment_with_stimuli` - Full setup
   - `experiment_minimal` - No initialization (current default)

3. **Update test documentation**
   - Document which tests require initialization
   - Add examples of each testing approach

## Implementation Files

### File 1: conftest.py (Add fixture)

Add this fixture to `tests/conftest.py`:

```python
@pytest.fixture
def initialized_n170_experiment(mock_eeg, temp_save_fn, mock_psychopy):
    """
    Fixture providing a fully initialized N170 experiment ready for testing.

    The experiment has:
    - Window initialized
    - Stimuli loaded
    - Trials configured
    - Ready to present stimuli or run
    """
    from eegnb.experiments.visual_n170.n170 import VisualN170

    experiment = VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=10,
        iti=0.2,
        soa=0.1,
        jitter=0.0,
        use_vr=False
    )

    # Initialize experiment (creates window, loads stimuli)
    experiment.setup(instructions=False)

    return experiment


@pytest.fixture
def experiment_with_window(mock_eeg, temp_save_fn, mock_psychopy):
    """
    Fixture providing an N170 experiment with window initialized but stimuli not loaded.

    Useful for testing load_stimulus() in isolation.
    """
    from eegnb.experiments.visual_n170.n170 import VisualN170
    from pandas import DataFrame

    experiment = VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=10,
        use_vr=False
    )

    # Create window manually
    experiment.window = MockWindow()

    # Initialize trial parameters
    experiment.parameter = np.random.binomial(1, 0.5, experiment.n_trials)
    experiment.trials = DataFrame(dict(
        parameter=experiment.parameter,
        timestamp=np.zeros(experiment.n_trials)
    ))
    experiment.markernames = [1, 2]

    return experiment
```

### File 2: test_n170_integration.py (Update failing tests)

Update tests to use the new fixtures. Example:

```python
@pytest.mark.integration
class TestN170StimulusLoading:
    """Test stimulus loading functionality."""

    def test_load_stimulus_basic(self, experiment_with_window):
        """Test basic stimulus loading."""
        experiment = experiment_with_window

        # Load stimuli
        stim = experiment.load_stimulus()

        # Check that stimuli were loaded
        assert stim is not None
        assert len(stim) == 2  # [houses, faces]
        assert hasattr(experiment, 'faces')
        assert hasattr(experiment, 'houses')

    def test_stimulus_trials_contain_valid_data(self, initialized_n170_experiment):
        """Test that trial data contains valid stimulus information."""
        experiment = initialized_n170_experiment

        # Trials should already be set up
        assert hasattr(experiment, 'trials')
        assert len(experiment.trials) == 10

        # Each trial should have parameter (label)
        for idx in range(len(experiment.trials)):
            label = experiment.trials["parameter"].iloc[idx]
            assert label in [0, 1]  # Binary parameter


@pytest.mark.integration
class TestN170StimulusPresentation:
    """Test stimulus presentation functionality."""

    def test_present_stimulus_single(self, initialized_n170_experiment, mock_eeg):
        """Test presenting a single stimulus."""
        experiment = initialized_n170_experiment

        # Present first stimulus
        experiment.present_stimulus(idx=0)

        # Should not crash
        assert True

    def test_present_stimulus_without_eeg(self, initialized_n170_experiment):
        """Test presenting stimulus without EEG device (should not crash)."""
        experiment = initialized_n170_experiment
        experiment.eeg = None  # Remove EEG

        # Should work without EEG
        try:
            experiment.present_stimulus(idx=0)
            assert True
        except Exception as e:
            pytest.fail(f"Present stimulus crashed without EEG: {e}")
```

### File 3: n170.py (Add docstring)

Add this docstring to the VisualN170 class:

```python
class VisualN170(Experiment.BaseExperiment):
    """
    Visual N170 oddball experiment for face vs. house discrimination.

    This experiment presents faces and houses in a random sequence to elicit
    the N170 event-related potential, a face-sensitive component occurring
    approximately 170ms post-stimulus.

    The N170 is characterized by a negative deflection in the EEG signal
    with larger amplitude for faces compared to other visual stimuli.

    Parameters
    ----------
    duration : int, default=120
        Duration of the recording in seconds
    eeg : EEG, optional
        EEG device instance for recording. If None, runs without EEG recording
    save_fn : str, optional
        Path to save the recording data
    n_trials : int, default=2010
        Number of trials to present
    iti : float, default=0.4
        Inter-trial interval in seconds
    soa : float, default=0.3
        Stimulus onset asynchrony in seconds
    jitter : float, default=0.2
        Random jitter added to timing in seconds (0-jitter range)
    use_vr : bool, default=False
        Whether to use VR display via Oculus Rift

    Attributes
    ----------
    faces : list of ImageStim
        Face stimulus images loaded from face_house/faces directory
    houses : list of ImageStim
        House stimulus images loaded from face_house/houses directory

    Examples
    --------
    >>> from eegnb.devices.eeg import EEG
    >>> from eegnb.experiments import VisualN170
    >>>
    >>> # Run without EEG
    >>> experiment = VisualN170(duration=60, n_trials=100)
    >>> experiment.run()
    >>>
    >>> # Run with EEG device
    >>> eeg = EEG(device='muse2')
    >>> experiment = VisualN170(duration=120, eeg=eeg, save_fn='/tmp/recording')
    >>> experiment.run()

    References
    ----------
    .. [1] Bentin, S., Allison, T., Puce, A., Perez, E., & McCarthy, G. (1996).
           Electrophysiological studies of face perception in humans.
           Journal of Cognitive Neuroscience, 8(6), 551-565.

    .. [2] Rossion, B., & Jacques, C. (2008). Does physical interstimulus
           variance account for early electrophysiological face sensitive
           responses in the human brain? Ten lessons on the N170.
           NeuroImage, 39(4), 1959-1979.
    """
```

## Summary of Changes Needed

| File | Change | Lines | Difficulty | Impact |
|------|--------|-------|------------|--------|
| `tests/conftest.py` | Add 2 new fixtures | ~60 | Easy | Fixes 10 tests |
| `tests/integration/test_n170_integration.py` | Update 10 test methods | ~50 | Easy | Fixes 10 tests |
| `eegnb/experiments/visual_n170/n170.py` | Add class docstring | ~45 | Easy | Fixes 1 test |
| **Total** | **3 files** | **~155** | **Easy** | **Fixes 11 tests** |

## Estimated Time

- **Quick fix (docstring only)**: 5 minutes → 1 test passing (34/44)
- **Full fix (all changes)**: 45-60 minutes → All tests passing (44/44)

## Testing the Fixes

After implementing changes:

```bash
# Test docstring fix
pytest tests/integration/test_n170_integration.py::TestN170Documentation -v

# Test stimulus loading fixes
pytest tests/integration/test_n170_integration.py::TestN170StimulusLoading -v

# Test all fixes
pytest tests/integration/test_n170_integration.py -v

# Check coverage
pytest tests/integration/test_n170_integration.py --cov=eegnb.experiments.visual_n170 --cov-report=term
```

## Next Steps

1. ✅ Implement docstring fix (2 minutes)
2. ✅ Add fixtures to conftest.py (15 minutes)
3. ✅ Update failing tests (30 minutes)
4. ✅ Run full test suite (2 minutes)
5. ✅ Commit and push changes (5 minutes)

**Total time: ~1 hour to achieve 100% test pass rate**
