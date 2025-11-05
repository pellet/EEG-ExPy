# EEG-ExPy Integration Tests

This directory contains integration tests for the EEG-ExPy experiment framework, with a focus on high-coverage testing of the N170 visual experiment.

## Overview

The test suite provides comprehensive integration testing with mocked EEG devices and PsychoPy components for headless testing in CI/CD environments. The tests verify complete experiment workflows including initialization, stimulus presentation, EEG integration, controller input handling, and error scenarios.

## Directory Structure

```
tests/
├── README.md                              # This file
├── conftest.py                            # Shared pytest fixtures and mock classes
├── fixtures/                              # Additional test fixtures (future)
│   └── __init__.py
├── integration/                           # Integration tests
│   ├── __init__.py
│   └── test_n170_integration.py          # N170 experiment tests
├── test_empty.py                          # Placeholder test
└── test_run_experiments.py                # Manual integration test (not run in CI)
```

## Test Architecture

### Mock Infrastructure (conftest.py)

The test suite uses custom mock classes that simulate the behavior of real hardware and UI components:

#### **MockEEG**
Simulates the `eegnb.devices.eeg.EEG` interface:
- Tracks start/stop calls
- Records marker pushes with timestamps
- Provides synthetic EEG data
- Configurable for different device types (Muse2, Ganglion, Cyton, etc.)

```python
def test_example(mock_eeg):
    mock_eeg.start("/tmp/recording", duration=10)
    mock_eeg.push_sample(marker=[1], timestamp=1.5)
    assert len(mock_eeg.markers) == 1
```

#### **MockWindow**
Simulates PsychoPy Window for headless testing:
- No display required
- Tracks flip() calls
- Supports context manager protocol

#### **MockImageStim / MockTextStim**
Simulates PsychoPy visual stimuli:
- Tracks draw() calls
- Supports image/text updates
- Lightweight for fast testing

#### **MockClock**
Provides deterministic timing control:
- Manual time advancement
- Predictable timestamps for testing

### Fixtures

#### Core Fixtures

- **`mock_eeg`**: Fresh MockEEG instance for each test
- **`mock_eeg_muse2`**: Muse2-specific configuration
- **`temp_save_fn(tmp_path)`**: Temporary file path for recordings
- **`mock_psychopy(mocker)`**: Complete PsychoPy mock setup
- **`mock_psychopy_with_spacebar`**: Auto-starts experiments with spacebar
- **`mock_vr_disabled`**: Disables VR controller input
- **`mock_vr_button_press`**: Simulates VR button press
- **`stimulus_images(tmp_path)`**: Creates temporary stimulus directory

#### Using Fixtures

```python
def test_experiment_with_eeg(mock_eeg, temp_save_fn, mock_psychopy):
    experiment = VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        use_vr=False
    )
    experiment.run(instructions=False)
    assert mock_eeg.start_count > 0
```

## N170 Integration Tests

The N170 test suite (`test_n170_integration.py`) contains **44 tests** organized into 12 test classes:

### Test Coverage

#### ✅ TestN170Initialization (8 tests)
- Basic initialization with default parameters
- Custom trial counts
- Timing parameter configurations (ITI, SOA, jitter)
- Initialization without EEG device
- VR enabled/disabled modes

#### ✅ TestN170EdgeCases (4 tests)
- Zero trials
- Very short durations
- Very long trial counts
- Zero jitter (deterministic timing)

#### ✅ TestN170SaveFunction (2 tests)
- Integration with `generate_save_fn()` utility
- Custom save paths

#### ✅ TestN170DeviceTypes (5 tests)
- Muse2, Muse2016, Ganglion, Cyton, Synthetic devices
- Device-specific channel configurations

#### ✅ TestN170StateManagement (2 tests)
- Multiple runs of same experiment instance
- EEG device state tracking

#### ✅ TestN170ControllerInput (4 tests)
- Keyboard spacebar start
- Escape key cancellation
- VR input enabled/disabled

#### ✅ TestN170ExperimentRun (4 tests)
- Minimal experiment execution
- With/without instructions
- Without EEG device

#### ✅ TestN170Documentation (1 test)
- Class docstring presence

#### ⚠️ TestN170StimulusLoading (2 tests)
- Requires window initialization (needs enhancement)

#### ⚠️ TestN170StimulusPresentation (3 tests)
- Requires window and stimulus loading (needs enhancement)

#### ⚠️ TestN170EEGIntegration (4 tests)
- Partially working (2/4 passing)

#### ⚠️ TestN170TimingAndSequencing (2 tests)
- Partially working

#### ⚠️ TestN170Performance (2 tests)
- Slow tests for stress testing (marked with `@pytest.mark.slow`)

### Current Status

- **✅ 33/44 tests passing (75%)**
- **❌ 11/44 tests failing (25%)**

Failing tests primarily involve stimulus loading and presentation, which require the experiment window to be initialized. These can be fixed by:
1. Adding window initialization to tests
2. Enhancing mocks to support more complex interactions
3. Refactoring experiment code to separate concerns

## Running Tests

### Run All Integration Tests

```bash
pytest tests/integration/
```

### Run N170 Tests Only

```bash
pytest tests/integration/test_n170_integration.py
```

### Run Specific Test Class

```bash
pytest tests/integration/test_n170_integration.py::TestN170Initialization
```

### Run With Coverage Report

```bash
pytest tests/integration/ --cov=eegnb --cov-report=html
```

### Run Fast Tests Only (Skip Slow Tests)

```bash
pytest tests/integration/ -m "not slow"
```

### Verbose Output

```bash
pytest tests/integration/ -v
```

### Show Test Names Without Running

```bash
pytest tests/integration/ --collect-only
```

## Test Markers

Tests can be marked with pytest markers for selective execution:

- `@pytest.mark.integration`: Integration test (all tests in this suite)
- `@pytest.mark.slow`: Slow-running test (skip in quick test runs)
- `@pytest.mark.requires_display`: Requires display (currently none)

## Dependencies

### Required Packages

```bash
pip install pytest pytest-cov pytest-mock numpy
```

### Optional (for full experiment functionality)

```bash
pip install -r requirements.txt
```

The test suite is designed to work with minimal dependencies by mocking heavy dependencies like PsychoPy, BrainFlow, and MuseLSL.

## CI/CD Integration

The tests are designed to run in GitHub Actions with:
- Ubuntu, Windows, macOS support
- Headless display via Xvfb on Linux
- Python 3.8, 3.10 compatibility
- Automatic coverage reporting

### GitHub Actions Configuration

```yaml
- name: Run integration tests
  run: pytest tests/integration/ --cov=eegnb --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Writing New Tests

### Basic Test Template

```python
@pytest.mark.integration
class TestNewFeature:
    """Test description."""

    def test_basic_functionality(self, mock_eeg, temp_save_fn, mock_psychopy):
        """Test basic functionality."""
        # Arrange
        experiment = VisualN170(
            duration=5,
            eeg=mock_eeg,
            save_fn=temp_save_fn,
            use_vr=False
        )

        # Act
        result = experiment.some_method()

        # Assert
        assert result is not None
        assert mock_eeg.start_count > 0
```

### Parametrized Test Template

```python
@pytest.mark.parametrize("duration,n_trials", [
    (5, 10),
    (10, 20),
    (15, 30),
])
def test_various_configurations(mock_eeg, temp_save_fn, duration, n_trials):
    """Test with various configurations."""
    experiment = VisualN170(
        duration=duration,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=n_trials
    )
    assert experiment.duration == duration
    assert experiment.n_trials == n_trials
```

## Best Practices

### 1. Use Fixtures for Reusable Components

```python
@pytest.fixture
def configured_experiment(mock_eeg, temp_save_fn):
    return VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=5
    )

def test_with_fixture(configured_experiment):
    assert configured_experiment.duration == 10
```

### 2. Mock at the Right Level

- Mock external dependencies (PsychoPy, BrainFlow)
- Don't mock the code you're testing
- Use `mocker.patch()` for temporary mocks in specific tests

### 3. Test Behavior, Not Implementation

```python
# Good: Test observable behavior
def test_markers_are_recorded(mock_eeg):
    experiment.present_stimulus(0)
    assert len(mock_eeg.markers) > 0

# Avoid: Testing internal implementation details
def test_internal_variable_name(experiment):
    assert hasattr(experiment, '_internal_var')  # Fragile
```

### 4. Use Descriptive Test Names

```python
# Good
def test_experiment_starts_eeg_device_when_run():
    pass

# Avoid
def test_run():
    pass
```

### 5. Keep Tests Independent

Each test should:
- Set up its own state
- Not depend on other tests
- Clean up after itself (handled by fixtures)

### 6. Test Edge Cases

```python
def test_zero_trials(mock_eeg, temp_save_fn):
    """Test handling of edge case: zero trials."""
    experiment = VisualN170(
        duration=10,
        eeg=mock_eeg,
        save_fn=temp_save_fn,
        n_trials=0  # Edge case
    )
    assert experiment.n_trials == 0
```

## Troubleshooting

### Import Errors

If you see import errors for PsychoPy, BrainFlow, etc.:
```python
# These are mocked at module level in test files
import sys
from unittest.mock import MagicMock
sys.modules['psychopy'] = MagicMock()
```

### Fixture Not Found

Ensure conftest.py is in the tests/ directory and fixtures are properly defined.

### Tests Pass Locally But Fail in CI

- Check for hardcoded paths
- Ensure tests don't require display
- Verify all dependencies are in requirements.txt

### Timeout Errors

For slow tests:
```python
@pytest.mark.timeout(60)  # 60 second timeout
def test_slow_operation():
    pass
```

## Coverage Goals

Current coverage for N170 module: **~69%**

Target coverage goals:
- **Critical paths**: 90%+ (initialization, EEG integration)
- **Overall module**: 80%+
- **Edge cases**: 70%+

View coverage report:
```bash
pytest --cov=eegnb.experiments.visual_n170 --cov-report=html
open htmlcov/index.html
```

## Future Enhancements

### Planned Improvements

1. **Complete Stimulus Loading Tests**
   - Mock stimulus file loading
   - Test with actual small test images

2. **Add More Experiment Types**
   - P300 integration tests
   - SSVEP integration tests
   - Auditory oddball tests

3. **Performance Benchmarking**
   - Time critical operations
   - Memory usage tracking
   - Frame rate validation

4. **Real Hardware Integration**
   - Optional tests with synthetic EEG device
   - BrainFlow synthetic board integration

5. **Visual Regression Testing**
   - Capture and compare stimulus rendering
   - Ensure UI consistency

## Contributing

When adding new tests:

1. Place tests in appropriate test class or create new class
2. Use existing fixtures when possible
3. Add docstrings to all test methods
4. Mark slow tests with `@pytest.mark.slow`
5. Update this README with new test categories
6. Ensure tests pass locally before committing

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)
- [EEG-ExPy documentation](https://neurotechx.github.io/eeg-notebooks/)

## Questions or Issues?

- Open an issue on GitHub
- Check existing tests for examples
- Review conftest.py for available fixtures
- Consult the pytest documentation

---

**Test Suite Status**: 🟢 Operational (33/44 tests passing)
**Last Updated**: 2025-11-05
**Maintainer**: EEG-ExPy Team
