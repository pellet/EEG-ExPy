
*********************************
Pattern Reversal VEP
*********************************

Visual Pattern Reversal VEP
============================

The pattern reversal visual evoked potential (PR-VEP) is a well-established clinical and research technique used to assess the function of the visual pathways. This experiment presents alternating checkerboard patterns to each eye separately, generating characteristic evoked potentials that reflect the integrity of the visual system from retina to visual cortex.

The PR-VEP is particularly valuable for:

* **Clinical diagnostics**: Detection of optic nerve disorders, multiple sclerosis, and other neurological conditions affecting vision
* **Monocular assessment**: Independent evaluation of each eye's visual pathway
* **Research applications**: Investigation of visual processing, binocular rivalry, and cortical plasticity
* **VR compatibility**: This implementation supports both traditional monitor-based and VR headset presentations

The experiment alternates between presenting checkerboard patterns to the left and right eyes across multiple blocks, with the checkerboard pattern reversing at 2Hz (every 500ms) to elicit robust VEP responses.

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

**Electrode Placement**

The PR-VEP is optimally recorded from occipital electrodes, particularly:

* **Oz**: Primary recording site over the visual cortex
* **O1, O2**: Additional occipital sites for lateralization studies
* **POz**: Parieto-occipital midline electrode

For EEG devices with limited electrode coverage, place the primary electrode at Oz (back of the head, approximately 10% of the head circumference above the inion).

**Pattern Reversal VEP Experiment Notebook Examples**

.. include:: ../auto_examples/visual_pattern_reversal_vep/index.rst
