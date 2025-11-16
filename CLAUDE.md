# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EEG-ExPy is a Python library for cognitive neuroscience experiments using consumer-grade EEG devices. It provides a unified framework for running classic EEG experiments (N170, P300, SSVEP, etc.) with devices like Muse, OpenBCI, and others. The goal is to democratize cognitive neuroscience research by making it accessible with affordable hardware.

## Claude Code Configuration

The `.claude/` directory contains configuration for Claude Code sessions:

### `.claude/config.json`
- **GitHub CLI (gh)**: Version 2.62.0, configured for workflow monitoring
  - Path: `~/.local/bin/gh`
  - Optional tool (not required)
  - Setup instructions in `.claude/hooks/SessionStart`

## Development Commands

### Environment Setup
```bash
# Install in development mode with all dependencies
pip install --use-pep517 .[full]

# Install specific dependency groups
pip install -e .[streaming]     # For EEG streaming
pip install -e .[stimpres]      # For stimulus presentation
pip install -e .[streamstim]    # Both streaming and stimulus
pip install -e .[docsbuild]     # For documentation building
```

### Testing and Quality Control
```bash
# Run tests
make test
# OR: pytest --cov=eegnb --cov-report=term --cov-report=xml --cov-report=html --nbval-lax --current-env

# Run type checking (excludes visual_cueing due to errors)
make typecheck
# OR: python -m mypy --exclude 'examples/visual_cueing'

# Run single test file
pytest tests/test_specific.py
```

### Documentation
```bash
# Build documentation
make docs
# OR: cd doc && make html

# Clean documentation build
make clean
# OR: cd doc && make clean
```

### CLI Usage
```bash
# Run experiments via CLI
eegnb runexp --experiment visual_n170 --eegdevice muse2016 --recdur 120
# OR: eegexpy runexp --experiment visual_n170 --eegdevice muse2016 --recdur 120

# Interactive mode
eegnb runexp --prompt
```

## Architecture

### Core Components

1. **BaseExperiment Class** (`eegnb/experiments/Experiment.py:29`)
   - Abstract base class for all experiments
   - Key parameters: `n_trials`, `iti` (inter-trial interval), `soa` (stimulus onset asynchrony), `jitter`
   - Subclasses must implement `load_stimulus()` and `present_stimulus()` methods
   - Supports both standard displays and VR via `use_vr` parameter

2. **EEG Device Abstraction** (`eegnb/devices/eeg.py`)
   - Unified interface for multiple EEG devices via BrainFlow and muselsl
   - Supported devices: Muse, OpenBCI (Cyton/Ganglion), g.tec Unicorn, BrainBit, synthetic
   - Handles device-specific streaming and marker insertion

3. **CLI Interface** (`eegnb/cli/__main__.py`)
   - Command-line interface using Click
   - Entry points: `eegnb` and `eegexpy`
   - Supports interactive prompts and direct parameter specification

### Experiment Implementation Pattern

Each experiment inherits from `BaseExperiment` and implements:
- `load_stimulus()`: Load visual/auditory stimuli
- `present_stimulus(idx)`: Present stimulus for trial idx and push EEG markers

Example from N170 experiment (`eegnb/experiments/visual_n170/n170.py:21`):
```python
class VisualN170(Experiment.BaseExperiment):
    def load_stimulus(self):
        # Load face and house images
        self.faces = list(map(load_image, glob(os.path.join(FACE_HOUSE, "faces", "*_3.jpg"))))
        self.houses = list(map(load_image, glob(os.path.join(FACE_HOUSE, "houses", "*.3.jpg"))))
        return [self.houses, self.faces]

    def present_stimulus(self, idx: int):
        label = self.trials["parameter"].iloc[idx]
        image = choice(self.faces if label == 1 else self.houses)
        image.draw()
        if self.eeg:
            self.eeg.push_sample(marker=self.markernames[label], timestamp=time())
        self.window.flip()
```

### Available Experiments

Located in `eegnb/experiments/`:
- `visual_n170/` - Face/object recognition ERP
- `visual_p300/` - Oddball paradigm ERP
- `visual_ssvep/` - Steady-state visual evoked potentials
- `auditory_oddball/` - Auditory ERP paradigms
- `visual_cueing/` - Spatial attention cueing
- `visual_gonogo/` - Go/No-go task
- `somatosensory_p300/` - Tactile P300

### Dependencies and Requirements

The `requirements.txt` is organized into sections:
- **Analysis**: Core analysis tools (MNE, scikit-learn, pandas)
- **Streaming**: EEG device streaming (brainflow, muselsl, pylsl)
- **Stimpres**: Stimulus presentation (PsychoPy, psychtoolbox)
- **Docsbuild**: Documentation building (Sphinx, etc.)

### Platform-Specific Notes

- **macOS**: Uses `pyobjc==7.3` for GUI support
- **Windows**: Includes `psychxr` for VR support, `pywinhook` for input handling
- **Linux**: Requires `pyo>=1.0.3` for audio, additional system dependencies for PsychoPy

### Data Storage

- Default data directory: `eegnb.DATA_DIR`
- Filename generation: `eegnb.generate_save_fn()`
- Analysis utilities in `eegnb/analysis/`

### Configuration

- PsychoPy preferences set to PTB audio library with high precision latency mode
- Test configuration in `pyproject.toml` with pytest coverage and nbval for notebooks
- Makefile provides convenient commands for common tasks

### Testing Strategy

- pytest with coverage reporting
- nbval for notebook validation
- Excludes `examples/**.py` and `baseline_task.py` from tests
- Type checking with mypy (excludes visual_cueing)

This architecture enables rapid development of new experiments while maintaining consistency across the experimental framework and supporting diverse EEG hardware platforms.
