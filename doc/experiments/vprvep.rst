********************************
_
*********************************

Visual Pattern Reversal VEP
============================

The pattern reversal visual evoked potential (PRVEP) is a well-established clinical and research technique used to assess the function of the visual pathways. This experiment presents alternating checkerboard patterns to each eye separately, generating characteristic evoked potentials that reflect the integrity of the visual system from retina to visual cortex.

The PRVEP is particularly valuable for:

* **Clinical diagnostics**: Detection of optic nerve disorders, multiple sclerosis, and other neurological conditions affecting vision
* **Monocular assessment**: Independent evaluation of each eye's visual pathway
* **Research applications**: Investigation of visual processing, binocular rivalry, and cortical plasticity
* **Myelination studies**: Precise measurement of conduction velocity changes for remyelination research
* **VR compatibility**: This implementation supports both traditional monitor-based and VR headset presentations

The experiment alternates between presenting checkerboard patterns to the left and right eyes across multiple blocks, with the checkerboard pattern reversing at a configurable rate to elicit robust VEP responses.

**Optimizing Display Refresh Rate and Reversal Frequency**

**Finding Your Monitor's Maximum Refresh Rate**

To achieve optimal timing precision for VEP measurements, first determine your display's capabilities:

**Windows:**
1. Right-click desktop → Display Settings → Advanced Display Settings
2. Note the refresh rate (e.g., 60Hz, 120Hz, 165Hz)
3. Or use: ``dxdiag`` → Display tab → Current Display Mode

**Linux:**
```bash
xrandr | grep '*'  # Shows current refresh rate
xrandr --verbose   # Shows all available rates
```

**Experimental Setup**

**Standard Monitor Setup**
For traditional monitor-based recordings, participants manually occlude one eye per block while fixating on a central red dot. The checkerboard stimulus covers the central visual field with 1-degree check sizes, optimized for clear VEP generation.

**VR Setup** 
When using VR headsets (tested with Meta Quest 2/3s), the experiment provides true stereoscopic presentation with independent stimulation of each eye. This eliminates the need for manual eye occlusion and provides more precise control over monocular stimulation.

**Technical Parameters**
- **Check size**: 1 degree of visual angle (0.5 cycles per degree)
- **Reversal rate**: 2Hz (500ms stimulus-onset asynchrony)
- **Block duration**: 50 seconds (100 trials per block)  
- **Total blocks**: 4 (alternating left/right eye)
- **Display requirements**: 60Hz or higher refresh rate

**Typical Durations**
Pattern reversal experiments typically run for 50 seconds per block, with blocks alternating between eyes. With 4 blocks (2 per eye), this results in approximately 100 seconds of stimulation per eye, plus any inter-block breaks or setup time. Total experiment time is around 5-10 minutes depending on participant instructions and rests.

**Electrode Placement**

The PRVEP is optimally recorded from occipital electrodes, particularly:

* **Oz**: Primary recording site over the visual cortex
* **O1, O2**: Additional occipital sites for lateralization studies
* **POz**: Parieto-occipital midline electrode

For EEG devices with limited electrode coverage, place the primary electrode at Oz (back of the head, approximately 10% of the head circumference above the inion).

Extra Electrode

While PRVEPs can be measured with the Cyton, the Muse 2016/2 supports an additional electrode via the device’s microUSB port which can be placed on Oz or POz.

For instructions on building and using an extra electrode with Muse, see: :doc:`../misc/using_an_extra_electrode_muse`
For this experiment, the extra electrode can be placed at Oz or POz and secured with a band or cap.

**Pattern Reversal VEP Experiment Notebook Examples**

.. include:: ../auto_examples/visual_block_pattern_reversal/index.rst
