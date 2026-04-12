********************************
_
*********************************

Visual Pattern Reversal VEP
===========================

The Pattern Reversal VEP (PR-VEP) is the most widely studied visual
evoked potential paradigm. A checkerboard pattern swaps its black and
white squares at a regular rate (typically 2 reversals per second) while
the participant fixates a central dot. Each reversal elicits a
stereotyped waveform whose most prominent feature is the **P100**, a
positive deflection occurring ~100ms after the reversal at midline
occipital electrodes. The other components are a small N75 before it
and an N145 after it.

In this notebook, we will attempt to detect the P100 with the OpenBCI
Cyton, with the most critical electrode at Oz, followed by O1 and O2,
then POz. Fp1 and Fp2 are optional channels for detecting eye movement
artefacts. We use monocular pattern reversal blocks and run the analysis
pipeline to pull out the per-eye P100 latency and the interocular
latency difference.


Stimulus Parameters
-------------------

Parameters follow the ISCEV "large check" option [Odom2016]_:

- **Check size**: 1° of visual angle (0.5 cpd)
- **Reversal rate**: 2 reversals per second
- **Field size**: 16° (monitor) / 20° (VR)
- **Contrast**: High contrast black/white, mean luminance held constant
- **Fixation**: Central red dot
- **Recording**: Monocular, alternating left and right eye per block

Four blocks of 50 seconds by default, giving ~100 reversals per eye per
block.


Monitor vs VR
-------------

The experiment supports both standard monitor presentation and Meta
Quest (VR) presentation via ``use_vr=True``.

**VR mode is preferred** for two reasons:

- Each eye sees the checkerboard independently, so there is no manual
  eye closure and no light leakage.
- The OpenXR compositor supplies a per-frame predicted photon time
  (``tracking_state.headPose.time``), which is attached to the EEG
  marker in place of ``time.time()``. This cancels most of the
  output-side display latency — render queue, compositor buffering,
  scan-out, HMD persistence — on a per-frame basis, which matters for
  P100 latency where shifts of 10–20ms are meaningful.

In monitor mode the software marker is the only timing source, so any
fixed display-pipeline latency has to be handled separately (see below).
A proof-of-concept photodiode sync patch is drawn in the bottom-left
corner of the window in monitor mode — a 50px square whose polarity
flips with each reversal. Taping a photodiode over that square and
routing its TTL into a spare channel would give hardware timing ground
truth; the code is in place but the hardware path has not been wired
up yet.


Electrode Placement
-------------------

The P100 is generated in occipital cortex. Priority electrode placement
for the OpenBCI Cyton is:

1. **Oz** — the primary electrode; highest amplitude P100
2. **O1, O2** — lateral occipital; provide left/right asymmetry information
3. **POz** — parieto-occipital midline; useful fallback or supplement
4. **Fp1, Fp2** — optional; placed on the forehead to record eye movement
   artefacts (EOG) for rejection during analysis


Timing Notes
------------

Measured P100 latency is the true P100 latency plus the display-pipeline
delay, plus the EEG device's input delay, plus any clock-alignment
error. For the Cyton the USB-serial latency is typically ~30–40ms, so
if you need *absolute* latencies you need to characterise and subtract
it; for *relative* comparisons (between-eye, within-subject across
sessions) it cancels out and you can ignore it.

Two sidecar files are written alongside each recording to let you check
timing after the fact:

- ``{save_fn}_timing.csv`` — per-trial software and compositor
  timestamps and their delta
- ``{save_fn}_frame_stats.json`` — per-frame intervals and dropped-frame
  count (150%-of-refresh threshold)

During each trial loop, Python garbage collection is disabled and
process priority is raised via ``psychopy.core.rush(True)`` to reduce
the chance of a dropped frame during a critical flip. Both are reset
between blocks.


Running the Experiment
----------------------

.. code-block:: python

   from eegnb.devices.eeg import EEG
   from eegnb.experiments.visual_vep import VisualPatternReversalVEP

   eeg = EEG(device='cyton')
   experiment = VisualPatternReversalVEP(
       display_refresh_rate=60,   # must match display and be divisible by 2Hz
       eeg=eeg,
       save_fn='my_vep_recording.csv',
       use_vr=True,               # False for monitor mode
   )
   experiment.run()


API Reference
-------------

.. autoclass:: eegnb.experiments.visual_vep.VisualPatternReversalVEP
   :members:
   :undoc-members:
   :show-inheritance:


References
----------

.. [Odom2016] Odom JV, Bach M, Brigell M, Holder GE, McCulloch DL, Mizota A,
   Tormene AP; International Society for Clinical Electrophysiology of Vision.
   **ISCEV standard for clinical visual evoked potentials: (2016 update).**
   *Documenta Ophthalmologica* 133(1):1-9. doi:10.1007/s10633-016-9553-y
