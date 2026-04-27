Visual Pattern Reversal VEP
===========================

The Pattern Reversal VEP (PR-VEP) is the most widely studied visual
evoked potential paradigm. A checkerboard pattern swaps its black and
white squares at a regular rate (typically 2 reversals per second) while
the participant fixates on a central cross. Each reversal elicits a
stereotyped triphasic waveform whose most prominent feature is the
**P100**, a positive deflection occurring ~100 ms after the reversal at
midline occipital electrodes. The other components are a smaller N75
before it and an N145 after it.

This implementation runs both ISCEV check-size variants in a single
session (large + small) and supports stereoscopic monocular presentation
through a Meta Quest HMD via PsychoPy / psychxr / Meta-Link. Trials are
analysed under two reference schemes (Oz−Fz ISCEV-standard and linked
mastoid M1+M2) and a wide range of differential biomarkers are reported,
including inter-ocular latency difference (IOLD), check-size slope
difference, hemispheric asymmetry, inter-ocular Δ-asymmetry contrasts,
lateral extrastriate readouts at P7/P8, and bootstrap confidence
intervals on the P100 latency.

**PR-VEP Experiment Notebook Examples:**

.. include:: ../auto_examples/visual_vep/index.rst


Running the Experiment
----------------------

.. code-block:: python

   from eegnb.devices.eeg import EEG
   from eegnb.experiments.visual_vep import VisualPatternReversalVEP

   eeg = EEG(device='cyton')
   experiment = VisualPatternReversalVEP(
       eeg=eeg,
       save_fn='my_vep_recording.csv',
       block_duration_seconds=50,
       block_trial_size=100,
       reps_per_condition=2,    # 4 conditions × 2 reps = 8 blocks total
       use_vr=True,             # False for monitor mode
   )
   experiment.run()

The display refresh rate is auto-detected from the active window
(``displayRefreshRate`` in VR, ``getActualFrameRate()`` on a monitor) and
must be an integer multiple of the 2 reversals/sec rate; 60, 90, 120, and
144 Hz are all supported.


Participant Preparation
-----------------------

The PR-VEP is sensitive to the optical quality of the retinal image.
Participants who normally wear glasses or contact lenses **must** wear
their corrective lenses during the test. Uncorrected refractive error
blurs the checkerboard's high spatial frequency edges, which attenuates
the P100 amplitude and can increase its latency — mimicking a genuine
neural conduction delay. This is especially important when comparing
latencies between eyes or across sessions, and even more so for the
small-check (high spatial-frequency) variant where blur dominates.

ISCEV guidelines require that visual acuity be documented for each
recording session. If a participant's corrected acuity is worse than
6/9 (20/30), note it alongside the data so that downstream analysis can
account for it.


Stimulus Parameters
-------------------

Two ISCEV check-size variants are presented per session [Odom2016]_:

- **Large checks**: 1.0° of visual angle (60 arcmin, 0.5 cpd) — the
  standard clinical PR-VEP stimulus, dominant low-spatial-frequency
  drive of the foveal P100.
- **Small checks**: 0.25° of visual angle (15 arcmin, 2.0 cpd) — the
  high-spatial-frequency variant. Demyelinated optic-nerve fibres
  preferentially delay this response, so the latency *difference*
  between large- and small-check P100 amplifies subtle demyelination
  that the large-check IOLD alone misses.

Other parameters:

- **Reversal rate**: 2 reversals per second (ISCEV standard, range
  1–3 rev/s) [Odom2016]_ — fast enough for a discrete transient P100,
  slow enough to stay well below photosensitive-seizure trigger
  frequencies.
- **Field size**: 16° square (``ISCEV_FIELD_DEG = 16.0``), rendered at
  the runtime-derived pixels-per-degree of the active display.
- **Contrast**: high-contrast black/white, mean luminance held constant
  through the inter-trial interval to avoid adaptation.
- **Fixation**: thin red cross, 0.25° × 0.05° (15 × 3 arcmin), sized in
  pixels via ``ppd`` so it stays sub-check at every variant and does not
  mask foveal stimulation.

Block scheduling: 4 conditions (left/right eye × large/small check) ×
``reps_per_condition`` repetitions, shuffled at construction time. With
the default ``reps_per_condition=2`` this gives 8 blocks of 50 s, ~100
reversals per block, ~200 reversals per (eye × size) condition, and
~400 reversals per eye total.

Block-start markers (100, 101, 102, 103) are pushed at the first
reversal of each block, encoding the full condition (bit 0 = eye, bit 1
= size class). Per-reversal markers are ``1`` for left-eye blocks and
``2`` for right-eye blocks. Both decode to the per-condition labels in
the analysis pipeline.


Monitor vs VR
-------------

The experiment supports both standard monitor presentation and Meta
Quest (VR) presentation via ``use_vr=True``.

**VR mode is preferred** for several reasons:

- Each eye sees the checkerboard independently via per-eye HMD buffers,
  so monocular blocks need no manual eye closure and have no light
  leakage from the unstimulated eye.
- The OpenXR / LibOVR compositor supplies a per-frame predicted photon
  time (``app_motion_to_photon_latency_s``), which is logged to the
  ``_timing.csv`` sidecar and applied trial-by-trial in the analysis to
  cancel most of the output-side display latency (render queue,
  compositor buffering, scan-out, HMD persistence).
- A keyboard (spacebar) or Quest controller trigger is required to advance
  past block instructions and to exit the experiment. The presented
  stimulus is otherwise hands-free during the block — there is no manual
  eye covering required.

A residual unmeasured chain delay (Quest Link transport + panel scan-out
+ LCD response + Cyton RF) of ~25 ± 15 ms remains, and is treated as a
fixed ``link_panel_lag`` constant in the analysis. Differential
biomarkers (IOLD, check-size slope difference, amplitude ratios,
Δ-asymmetry contrasts) are robust to this residual because both eyes
share the same chain — the residual cancels in any L−R contrast.
*Absolute* P100 latency (vs clinical norms) does *not* survive this
residual and is not reported as a clinical number.

A separate stimulus calibration caveat applies to VR: the Quest panel is
not photometrically calibrated to a specific cd/m² or contrast ratio, so
absolute P100 *amplitude* will diverge from clinical norms by an unknown
constant factor. As with the latency offset, differential / within-
subject biomarkers remain interpretable; absolute amplitudes are not
interchangeable with clinical norms.

In monitor mode the software flip is the only timing source, so any
fixed display-pipeline latency must be characterised separately if
absolute latency is needed. The most accurate option is a photodiode
taped over a sync patch on the screen with its analogue output routed
into a spare Cyton input (auxiliary channel or a free EEG channel) —
the diode fires when actual photons arrive at the screen, so epoching
off the diode trace aligns trials to true stimulus onset with sub-frame
precision and removes the entire display-chain residual.


Electrode Placement
-------------------

The P100 is generated in occipital cortex. The default cap montage in
``00x__pattern_reversal_run_experiment.py`` is an 8-channel OpenBCI
Cyton layout:

.. code-block:: python

   ch_names = ["Fz", "Pz", "P7", "P8", "O1", "O2", "Oz", "M2"]
   # Reference: M1 (Cyton SRB / hardware reference)
   # Ground:    A2

Channel roles in the analysis pipeline:

1. **Oz** — the primary ISCEV electrode; highest-amplitude P100. All
   single-channel biomarkers (IOLD, check-size slope, amplitude ratio,
   bootstrap CI) are computed at Oz.
2. **O1, O2** — lateral occipital. Used for the hemispheric-asymmetry
   biomarker, but interpretation is confounded by paradoxical
   lateralization (V1 dipoles in the calcarine fold project across the
   midline). The inter-ocular Δ-asymmetry contrast separates eye-
   dependent skew from stationary anatomical / electrode-stationary
   asymmetries.
3. **P7, P8** — lateral parieto-occipital electrodes. Because they largely
   bypass the "paradoxical lateralization" seen at O1/O2, they provide a
   cleaner, more direct readout of each hemisphere independently. They are
   essential for side-localizing cortical asymmetry and drive three biomarkers:
   lateral propagation latency (7), P7/P8 hemispheric asymmetry (8), and the
   lateral-hemisphere composite (9).
4. **Pz** — parietal midline. Used in the topology QC check to confirm
   the expected ``Oz > O1/O2 > P7/P8/Pz`` posterior gradient.
5. **Fz** — frontal midline. Doubles as (a) the reference electrode for
   the ISCEV Oz−Fz scheme, and (b) when the linked-mastoid scheme is
   active, the Halliday polarity-inversion check: a genuine V1-generated
   P100 produces an *inverted* (negative) deflection at Fz at the same
   latency, confirming a posterior dipole.
6. **M1 (reference) + M2** — mastoids. M1 is the Cyton SRB hardware
   reference (zero by construction). M2 is recorded; together they form
   the linked-mastoid (M1+M2)/2 reference scheme used alongside Oz−Fz.

For the alternative ``mark-iv`` montage (Thinkpulse dry actives at 4×
gain), edit ``ch_names`` in ``00x__pattern_reversal_run_experiment.py``
to the Thinkpulse channel set.


Reference Schemes
-----------------

The analysis pipeline runs every biomarker under two reference schemes
in sequence and prints both:

- **Oz − Fz (ISCEV standard)** — directly comparable with clinical
  PR-VEP norms. Sensitive to Fz contact quality.
- **Linked mastoid (M1+M2)/2** — more stable when Fz contact is weak,
  unaffected by frontal EMG, and the only scheme under which the Fz
  Halliday polarity-inversion check is meaningful (Fz is zero by
  construction under the Oz−Fz reference).

Each scheme produces its own set of plots and biomarkers; agreement
between the two is itself a quality indicator.


Biomarkers
----------

The analysis script ``01r__pattern_reversal_viz.py`` reports several differential biomarkers, in addition to the per-condition P100 latency and amplitude at Oz. These are grouped by their primary clinical utility:

**Remyelination & Longitudinal Tracking (Demyelination Indicators)**
These metrics are the highest value for tracking repair because they isolate pathway delays and cancel out daily systemic noise or anatomical distortions.

1. **Inter-ocular latency difference (IOLD)** — signed ``L − R`` P100 latency at Oz. ``> 8 ms`` is the most-cited clinical threshold for unilateral demyelination; robust to the ``link_panel_lag`` residual because both eyes share the same chain. The gold standard for longitudinal tracking of optic nerve repair.
2. **Inter-ocular check-size slope difference** — demyelinated (and repairing) optic nerves preferentially delay the small-check response. The ``L_eye − R_eye`` difference of latency-vs-check-size slopes amplifies asymmetric demyelination and is an excellent leading indicator of repair.
5. **Inter-ocular Δ-asymmetry contrast** — ``(O1−O2)|L_eye − (O1−O2)|R_eye``. By subtracting the spatial asymmetry between eyes, it cancels stationary anatomical distortions (e.g., asymmetrical cortical folding). The remaining Δ strictly represents pathway-driven changes over time.
6. **Bootstrap confidence intervals on P100 / IOLD** — 1000 trial-resamples per eye recompute the trimmed-mean P100 location, giving a 95% CI. The IOLD CI is flagged separately for excluding zero (statistically separable) and excluding ±8 ms (clinically meaningful). Tracking this CI width proves remyelination is statistically real.
7. **Lateral extrastriate P100 (P7 / P8) & Propagation** — per-channel P100 detection plus the Oz → lateral propagation latency (``P7−Oz``, ``P8−Oz``). While IOLD tracks *optic nerve* myelination, this tracks *intracortical* white matter myelination. A severe delay here indicates a cortical white matter issue (e.g., TBI, concussion, or cortical demyelination), not an optic nerve problem.

**Axonal & Compressive Indicators**
While latency measures myelin, amplitude measures the surviving nerve fibers (axons).

3. **Inter-ocular amplitude ratios** — ``L / R`` P100 amplitude at Oz. Tracks axonal loss (e.g., after severe optic neuritis) or compressive lesions (e.g., tumor blocking the signal). Less specific than latency but critical for differentiating slow recovery from permanent damage.
- **N75–P100–N145 Peak-to-Peak Amplitude** — measures the total "energy" of the visual response, which is more stable than absolute voltage against baseline drift.

**Morphological & Cortical Indicators**
Focus on waveform shape and direct hemispheric comparisons.

4. **Hemispheric asymmetry (O1 vs O2)** — per-eye P100 latency/amplitude at O1 and O2. Due to paradoxical lateralization, a deficit at O1 (left scalp) suggests *right* hemisphere involvement.
8. **Lateral extrastriate asymmetry (P7 vs P8)** — direct hemispheric read without the paradoxical lateralization of V1. Flags stationary asymmetry vs. eye-dependent asymmetry.
9. **Combined lateral hemisphere composites** — ``(O1+P7)/2`` vs ``(O2+P8)/2`` traces. Provides ~√2 better SNR than O1/O2 alone.
- **W-Peak (Bifurcated P100) Analysis** — detects if the P100 splits into two peaks (a "W" shape), which often indicates a central scotoma (macular dysfunction) or severe uncorrected refractive error causing spatial desynchronization.

**Quality Control**

10. **Topology QC** (linked-mastoid only) — confirms (a) the ``Oz > Pz > 0`` posterior-gradient at the Oz P100 latency, and (b) the Halliday frontal polarity inversion (Fz negative at the Oz P100 latency) which strongly confirms a true V1 generator.

All biomarkers are computed twice (once per reference scheme) and printed alongside their expected normal ranges, so a session's clinical interpretability is visible at a glance.


Latency Resolution
------------------

The precision of a P100 latency estimate depends on three factors:

1. **Display refresh rate** — determines the worst-case stimulus timing
   jitter. At 120 Hz this is ~4.2 ms per frame; at 60 Hz, ~16.7 ms.
2. **EEG sampling rate** — the Cyton samples at 250 Hz, giving 4 ms
   between samples. Without interpolation, the peak latency is locked
   to the nearest sample and cannot resolve shifts smaller than 4 ms.
3. **Number of trials** — averaging more reversals reduces noise in the
   ERP waveform, tightening the bootstrap CI around the peak estimate.
   The default 8-block design yields ~400 reversals per eye (split
   across the two check sizes).

To achieve sub-sample precision, ``vep_utils.get_pr_vep_latencies``
fits a parabola through the peak sample and its two neighbours and
takes the vertex as the true peak — bringing effective resolution to
~0.5 ms at 250 Hz, well below the sample interval.

For studies that require detecting latency shifts of 1–2 ms (e.g.
within-subject longitudinal comparisons), the combination of 120 Hz
display, parabolic interpolation, and the default 8-block design is
recommended. The trimmed-mean evoked estimator (used throughout the
analysis) further reduces sensitivity to occasional blink-contaminated
trials that survive amplitude rejection.


Longitudinal Tracking
---------------------

To monitor P100 latency over time — for example during nerve recovery
or longitudinal intervention tracking — record multiple sessions using
the same subject and session numbering scheme.

The analysis pipeline is split into two scripts so this is fast:

- ``01r__pattern_reversal_viz.py`` runs the per-session analysis. In
  addition to the figures and stdout output, it writes a
  ``biomarkers.json`` file into the recording directory containing
  every biomarker under both reference schemes plus session metadata
  (subject, session, device, site, trial counts, PC-side and
  ``link_panel_lag`` timing values).
- ``02r__pattern_reversal_longitudinal.py`` reads every session's
  ``biomarkers.json`` for a given subject, builds a flattened
  ``pandas.DataFrame``, and plots IOLD, per-eye P100 latency,
  check-size slope difference, O1/O2 and P7/P8 Δ-asymmetry, and
  amplitude ratio as a function of session. Bootstrap 95% CI bands are
  drawn on the IOLD and per-eye P100 panels so a real shift is visually
  separable from session noise.

This split means new longitudinal points are added in seconds (read
JSON, plot) rather than minutes (re-load EEG, filter, epoch,
bootstrap), and individual sessions can be re-analysed without
invalidating the rest of the trend series.

Before attributing a latency change to an intervention (like remyelination), establish a **baseline**: record at least 3–5 sessions over 1–2 weeks under the same conditions. Latency shifts can be caused by many non-pathological factors, including subject fatigue, alertness, caffeine intake, body temperature changes, or ocular differences (like pupil size or slight changes in focus/fixation). The longitudinal script supports a ``BASELINE_LAST_SESSION_NB`` cutoff that limits the summary statistics to those initial sessions, capturing the natural day-to-day variability for your setup and participant. Subsequent sessions need to fall significantly outside this baseline variability (e.g., beyond the ``mean ± std``) to be considered a true physiological or structural shift.


Timing Notes
------------

Two sidecar files are written alongside each recording to let you check
timing after the fact:

- ``{save_fn}_timing.csv`` — per-trial software and compositor
  timestamps and their delta, including the per-trial
  ``app_motion_to_photon_latency_s`` used by the analysis.
- ``{save_fn}_frame_stats.json`` — per-frame intervals and dropped-
  frame count (150%-of-refresh threshold).




References
----------

.. [Odom2016] Odom JV, Bach M, Brigell M, Holder GE, McCulloch DL, Mizota A,
   Tormene AP; International Society for Clinical Electrophysiology of Vision.
   **ISCEV standard for clinical visual evoked potentials: (2016 update).**
   *Documenta Ophthalmologica* 133(1):1-9. doi:10.1007/s10633-016-9553-y
