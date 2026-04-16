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


**PR-VEP Experiment Notebook Examples:**

.. include:: ../auto_examples/visual_vep/index.rst


Running the Experiment
----------------------

.. code-block:: python

   from eegnb.devices.eeg import EEG
   from eegnb.experiments.visual_vep import VisualPatternReversalVEP

   eeg = EEG(device='cyton')
   experiment = VisualPatternReversalVEP(
       display_refresh_rate=120,  # must match display and be divisible by 2; higher rates give better latency precision
       eeg=eeg,
       save_fn='my_vep_recording.csv',
       use_vr=True,               # False for monitor mode
   )
   experiment.run()


Participant Preparation
-----------------------

The PR-VEP is sensitive to the optical quality of the retinal image.
Participants who normally wear glasses or contact lenses **must** wear
their corrective lenses during the test. Uncorrected refractive error
blurs the checkerboard's high spatial frequency edges, which attenuates
the P100 amplitude and can increase its latency — mimicking a genuine
neural conduction delay. This is especially important when comparing
latencies between eyes or across sessions.

ISCEV guidelines require that visual acuity be documented for each
recording session. If a participant's corrected acuity is worse than
6/9 (20/30), note it alongside the data so that downstream analysis can
account for it.


Stimulus Parameters
-------------------

Parameters follow the ISCEV "large check" option [Odom2016]_:

- **Check size**: 1° of visual angle (0.5 cpd)
- **Reversal rate**: 2 reversals per second (one reversal per two display frames)
- **Field size**: 16° (monitor) / 20° (VR)
- **Contrast**: High contrast black/white, mean luminance held constant
- **Fixation**: Central red dot
- **Recording**: Monocular, alternating left and right eye per block

Eight blocks of 50 seconds by default, giving ~100 reversals per eye per
block (400 per eye total).

The experiment requires a display refresh rate that is divisible by two,
since each reversal occupies exactly two frames. Any such refresh rate is
supported — 60 Hz, 90 Hz, 120 Hz, 144 Hz, etc. A higher refresh rate
reduces the temporal jitter between the true reversal onset and the
nearest frame boundary, which directly translates to more precise P100
latency estimates. For example, at 60 Hz each frame is ~16.7 ms wide,
whereas at 120 Hz it is ~8.3 ms — halving the worst-case timing error.
VR headsets running at 90 Hz or above are therefore preferred over a
standard 60 Hz monitor when absolute latency precision matters.


Monitor vs VR
-------------

The experiment supports both standard monitor presentation and Meta
Quest (VR) presentation via ``use_vr=True``.

**VR mode is preferred** for two reasons:

- Each eye sees the checkerboard independently, so there is no manual
  eye closure and no light leakage.
- The OpenXR compositor supplies a per-frame predicted photon time
  (``tracking_state.headPose.timeInSeconds``), which is attached to the EEG
  marker in place of ``time.time()``. This cancels most of the
  output-side display latency — render queue, compositor buffering,
  scan-out, HMD persistence — on a per-frame basis, which matters for
  P100 latency where even small shifts are clinically meaningful.

In monitor mode the software marker is the only timing source, so any
fixed display-pipeline latency has to be handled separately (see below).
A proof-of-concept photodiode sync patch is drawn in the bottom-left
corner of the window in monitor mode — a 50px square whose polarity
flips with each reversal. Taping a photodiode over that square and
routing its TTL into a spare channel would give hardware timing ground
truth; the code is in place but the hardware path is a work in progress —
instructions for wiring a photodiode to a Cyton digital input pin will
be added in a future update.


Electrode Placement
-------------------

The P100 is generated in occipital cortex. Priority electrode placement
for the OpenBCI Cyton is:

1. **Oz** — the primary electrode; highest amplitude P100
2. **O1, O2** — lateral occipital; provide left/right asymmetry information
3. **POz** — parieto-occipital midline; useful fallback or supplement
4. **Fp1, Fp2** — optional; placed on the forehead to record eye movement
   artefacts (EOG) for rejection during analysis


Latency Resolution
------------------

The precision of a P100 latency estimate depends on three factors:

1. **Display refresh rate** — determines the worst-case stimulus timing
   jitter (see *Stimulus Parameters* above). At 120 Hz this is ~4.2 ms
   per frame.

2. **EEG sampling rate** — the Cyton samples at 250 Hz, giving 4 ms
   between samples. Without interpolation, the peak latency is locked to
   the nearest sample and cannot resolve shifts smaller than 4 ms.

3. **Number of trials** — averaging more reversals reduces noise in the
   ERP waveform, tightening the confidence interval around the peak
   estimate. The default is 8 blocks of 100 reversals (400 per eye).

To achieve sub-sample precision the analysis pipeline uses **parabolic
interpolation**: a parabola is fitted through the peak sample and its
two neighbours, and the vertex of the fit is taken as the true peak
location. At 250 Hz this brings effective resolution to ~0.5 ms — well
below the sample interval. The interpolated peak finder is used by
default in ``vep_utils.plot_vep()``.

For studies that require detecting latency shifts of 1–2 ms (e.g.
within-subject longitudinal comparisons), the combination of 120 Hz
display, parabolic interpolation, and the default 8-block design is
recommended.


Longitudinal Tracking
---------------------

To monitor P100 latency over time — for example during nerve recovery or
neuroplasticity studies — record multiple sessions using the same subject
and session numbering scheme and compare the per-eye P100 across them.

Before attributing a latency change to an intervention, establish a
**baseline**: record at least 3–5 sessions over 1–2 weeks under the same
conditions. This gives you the natural session-to-session variability for
your setup and participant, so you can distinguish a real shift from
measurement noise.

The ``02r__pattern_reversal_longitudinal.py`` example notebook
demonstrates the full workflow: discovering sessions, extracting per-eye
P100 latencies with parabolic interpolation, printing a summary table,
and plotting latency trends and interocular differences over time.


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


References
----------

.. [Odom2016] Odom JV, Bach M, Brigell M, Holder GE, McCulloch DL, Mizota A,
   Tormene AP; International Society for Clinical Electrophysiology of Vision.
   **ISCEV standard for clinical visual evoked potentials: (2016 update).**
   *Documenta Ophthalmologica* 133(1):1-9. doi:10.1007/s10633-016-9553-y
