"""EEG-ExPy/eegnb/experiments/visual_vep/pattern_reversal_vep.py

Pattern Reversal Visual Evoked Potential (PR-VEP) Experiment
=============================================================

This module implements a Pattern Reversal Visual Evoked Potential (PR-VEP) experiment,
which is used to measure the P100 component of the visual evoked potential.

The P100 is a positive deflection in the EEG signal that occurs approximately 100ms after
a visual stimulus. It is most prominent in the occipital region and is elicited by presenting
a checkerboard pattern that reverses (swaps black and white squares) at regular intervals.

This implementation supports both:
- Standard monitor-based presentation with monocular viewing (one eye at a time)
- VR-based presentation with stereoscopic viewing (Meta Quest)

The experiment runs in blocks, alternating between left and right eye stimulation to allow
for monocular visual evoked potential measurements.

Clinical and Research Applications:
- Assessment of visual pathway function
- Detection of optic nerve lesions
- Multiple sclerosis diagnosis
- Visual system maturation studies

Standard Parameters:
- Check size: 1 degree (0.5 cycles per degree)
- Reversal rate: 2 Hz (pattern reverses twice per second)
- Recording duration: Typically 50 seconds per block
- Electrodes: Focus on occipital channels (O1, Oz, O2, PO3, PO4)
"""

from time import time
import numpy as np

from psychopy import visual
from typing import Optional, Dict, Any
from eegnb.devices.eeg import EEG
from eegnb.experiments.BlockExperiment import BlockExperiment
from stimupy.stimuli.checkerboards import contrast_contrast

# Pixels per degree for Oculus/Meta Quest with PC link
QUEST_PPD = 20


class VisualPatternReversalVEP(BlockExperiment):
    """Pattern Reversal Visual Evoked Potential (PR-VEP) experiment class.

    This experiment presents a checkerboard pattern that reverses contrast at a fixed rate
    to elicit the P100 visual evoked potential component. The experiment alternates between
    left and right eye stimulation across blocks to allow for monocular VEP measurements.

    The checkerboard consists of black and white squares that reverse at 2 Hz (twice per second),
    with the participant fixating on a central red dot throughout the presentation.

    Attributes:
        display_refresh_rate (int): Expected display refresh rate in Hz (must be divisible by reversal rate)
        black_background (visual.Rect): Black rectangle covering the non-stimulated eye or periphery
        grey_background (visual.Rect): Grey background for VR to maintain luminance
        parameter (np.ndarray): Array indicating which eye to stimulate for each trial (0=left, 1=right)

    Example:
        >>> from eegnb.devices.eeg import EEG
        >>> eeg = EEG(device='muse')
        >>> experiment = VisualPatternReversalVEP(display_refresh_rate=60, eeg=eeg)
        >>> experiment.run()
    """

    def __init__(self, display_refresh_rate: int, eeg: Optional[EEG] = None, save_fn=None,
                 block_duration_seconds=50, block_trial_size: int=100, n_blocks: int=4,
                 use_vr=False, use_fullscr=True):
        """Initialize the Pattern Reversal VEP experiment.

        Args:
            display_refresh_rate (int): Expected refresh rate of the display in Hz.
                Must be divisible by the stimulus reversal rate (2 Hz). Common values: 60, 120.
            eeg (Optional[EEG]): EEG device object for recording. If None, runs without EEG recording.
            save_fn (Optional[str]): Filename for saving recorded data. If None, generates default filename.
            block_duration_seconds (int): Duration of each block in seconds. Default: 50.
            block_trial_size (int): Number of trials (pattern reversals) per block. Default: 100.
            n_blocks (int): Total number of blocks to run. Default: 4 (alternates between eyes).
            use_vr (bool): If True, uses VR headset (Meta Quest) for stereoscopic presentation.
                If False, uses standard monitor with monocular viewing. Default: False.
            use_fullscr (bool): If True, runs in fullscreen mode. Default: True.

        Note:
            - The stimulus reverses at 2 Hz (SOA = 0.5 seconds)
            - Blocks alternate between left eye (even blocks) and right eye (odd blocks)
            - For monitor-based presentation, participants manually close one eye
            - For VR presentation, the non-stimulated eye sees a black screen
        """
        self.display_refresh_rate = display_refresh_rate

        # Timing parameters for pattern reversal VEP
        soa = 0.5  # Stimulus Onset Asynchrony: 0.5s = 2 Hz reversal rate (standard for P100)
        iti = 0    # No inter-trial interval (continuous presentation)
        jitter = 0 # No jitter (precise timing critical for VEP)

        # Initialize parent BlockExperiment class with stereoscopic support
        super().__init__("Visual Pattern Reversal VEP", block_duration_seconds, eeg, save_fn,
                        block_trial_size, n_blocks, iti, soa, jitter, use_vr, use_fullscr,
                        stereoscopic=True)

        # Instruction text shown at the start of the experiment
        self.instruction_text = f"""Welcome to the Visual Pattern Reversal VEP experiment!

        This experiment will run for {n_blocks} blocks of {block_duration_seconds} seconds each.

        Press spacebar or controller to continue.
        """

        # Setting up the trial and parameter list
        # 0 = left eye, 1 = right eye
        left_eye = 0
        right_eye = 1

        # Alternate between left and right eye blocks
        # Even blocks (0, 2, 4...) = left eye
        # Odd blocks (1, 3, 5...) = right eye
        block_eyes = []
        for block_num in range(n_blocks):
            eye = left_eye if block_num % 2 == 0 else right_eye
            block_eyes.extend([eye] * block_trial_size)
        self.parameter = np.array(block_eyes)

    @staticmethod
    def create_monitor_checkerboard(intensity_checks):
        """Create a checkerboard stimulus for standard monitor presentation.

        Uses standard clinical parameters for pattern reversal VEP:
        - Check size: 1 degree of visual angle (0.5 cycles per degree)
        - Field size: 16 x 16 degrees
        - Viewing distance and pixels per degree calibrated for typical monitor setup

        Args:
            intensity_checks (tuple): Tuple of two values specifying the intensity pattern
                for the checkerboard squares. Typically (1, -1) or (-1, 1) for full contrast
                black and white reversal.

        Returns:
            dict: Dictionary containing the checkerboard image and metadata from stimupy.
                The 'img' key contains the numpy array of the checkerboard pattern.

        Note:
            The standard 1-degree check size is optimal for eliciting the P100 response.
            Smaller or larger check sizes may produce different VEP components.
        """
        # Standard parameters for monitor-based pattern reversal VEP
        # Using standard 1 degree check size at 72 pixels per degree
        return contrast_contrast(
            visual_size=(16, 16),  # Visual field size in degrees
            ppd=72,  # Pixels per degree (assumes ~60cm viewing distance on typical monitor)
            frequency=(0.5, 0.5),  # Spatial frequency: 0.5 cpd = 1 degree check size (standard)
            intensity_checks=intensity_checks,  # Contrast pattern for checkerboard squares
            target_shape=(0, 0),  # No target (full field checkerboard)
            alpha=0,  # No transparency
            tau=0     # No temporal modulation
        )

    @staticmethod
    def create_vr_checkerboard(intensity_checks):
        """Create a checkerboard stimulus optimized for VR presentation (Meta Quest).

        Parameters are adjusted for the Meta Quest's display characteristics:
        - Resolution: ~20 pixels per degree
        - Field of view: ~90 degrees
        - Check size: 1 degree (standard for VEP)

        Args:
            intensity_checks (tuple): Tuple of two values specifying the intensity pattern
                for the checkerboard squares. Typically (1, -1) or (-1, 1) for full contrast
                black and white reversal.

        Returns:
            dict: Dictionary containing the checkerboard image and metadata from stimupy.
                The 'img' key contains the numpy array of the checkerboard pattern.

        Note:
            VR presentation offers advantages including:
            - Better control of monocular presentation (no need to manually close one eye)
            - Larger field of view
            - More immersive experience reducing distractions
        """
        # Optimized parameters for Oculus/Meta Quest with PC link
        # Meta Quest has approximately 20 pixels per degree and a ~90° FOV
        # Using standard 1 degree check size (0.5 cpd)
        return contrast_contrast(
            visual_size=(20, 20),  # Size in degrees - covers central field while fitting in FOV
            ppd=QUEST_PPD,  # Pixels per degree for Meta Quest
            frequency=(0.5, 0.5),  # Spatial frequency: 0.5 cpd = 1 degree check size (standard)
            intensity_checks=intensity_checks,  # Contrast pattern for checkerboard squares
            target_shape=(0, 0),  # No target (full field checkerboard)
            alpha=0,  # No transparency
            tau=0     # No temporal modulation
        )

    def load_stimulus(self) -> Dict[str, Any]:
        """Load and prepare all visual stimuli for the experiment.

        This method creates the checkerboard patterns, fixation points, and instruction text
        needed for the experiment. For VR mode, it creates separate stimuli for left and right
        eyes to enable stereoscopic presentation. For monitor mode, it creates a single set
        of stimuli.

        The checkerboard reversals are pre-computed in two phases (black/white and white/black)
        to ensure precise timing during presentation.

        Returns:
            Dict[str, Any]: Dictionary containing all stimulus objects organized by eye/mode:

                For VR mode:
                    {
                        'left': {
                            'checkerboards': [phase1_stim, phase2_stim],
                            'fixation': fixation_stim,
                            'vr_block_instructions': instruction_text_stim
                        },
                        'right': {
                            'checkerboards': [phase1_stim, phase2_stim],
                            'fixation': fixation_stim,
                            'vr_block_instructions': instruction_text_stim
                        }
                    }

                For monitor mode:
                    {
                        'monoscopic': {
                            'checkerboards': [phase1_stim, phase2_stim],
                            'fixation': fixation_stim,
                            'vr_block_instructions': instruction_text_stim
                        }
                    }

        Raises:
            AssertionError: If the display refresh rate doesn't match expected rate or
                          if refresh rate is not divisible by stimulus rate (SOA).

        Note:
            This method is called once at the start of the experiment. The loaded stimuli
            are then reused across all blocks for efficiency.
        """
        # Frame rate verification in Hz
        # GetActualFrameRate() crashes in psychxr due to 'EndFrame called before BeginFrame'
        actual_frame_rate = np.round(self.window.displayRefreshRate if self.use_vr else self.window.getActualFrameRate())

        # Ensure the expected frame rate matches and is divisible by the stimulus rate (soa)
        assert actual_frame_rate % self.soa == 0, \
            f"Expected frame rate divisible by stimulus rate: {self.soa}, but got {actual_frame_rate} Hz"
        assert self.display_refresh_rate == actual_frame_rate, \
            f"Expected frame rate {self.display_refresh_rate} Hz, but got {actual_frame_rate} Hz"

        # Select checkerboard creation function based on presentation mode
        if self.use_vr:
            # Create the VR checkerboard
            create_checkerboard = self.create_vr_checkerboard
            # The window is large over the eye, checkerboard should only cover the central vision
            size = self.window.size / 1.5
        else:
            # Create the Monitor checkerboard
            create_checkerboard = self.create_monitor_checkerboard
            # Use square aspect ratio based on window height
            size = (self.window_size[1], self.window_size[1])

        # The surrounding / periphery needs to be dark when not using VR.
        # Also used for covering eye which is not being stimulated.
        self.black_background = visual.Rect(self.window,
                                            width=self.window.size[0],
                                            height=self.window.size[1],
                                            fillColor='black')

        # A grey background behind the checkerboard must be used in VR to maintain luminance.
        # This prevents adaptation effects and maintains consistent visual conditions.
        self.grey_background = visual.Rect(self.window,
                                            width=self.window.size[0],
                                            height=self.window.size[1],
                                            fillColor=[-0.22, -0.22, -0.22])

        # Create checkerboard stimuli
        def create_checkerboard_stim(intensity_checks, pos):
            """Helper function to create a checkerboard stimulus at a specific position.

            Args:
                intensity_checks (tuple): Intensity pattern for the checkerboard
                pos (tuple): Position (x, y) in pixels for the stimulus center

            Returns:
                visual.ImageStim: PsychoPy image stimulus object containing the checkerboard
            """
            return visual.ImageStim(self.window,
                                    image=create_checkerboard(intensity_checks)['img'],
                                    units='pix', size=size, color='white', pos=pos)

        # Create fixation stimuli
        def create_fixation_stim(pos):
            """Helper function to create a red fixation dot at a specific position.

            The fixation point is critical for maintaining stable gaze during the experiment,
            which ensures consistent P100 responses.

            Args:
                pos (tuple): Position (x, y) for the fixation point

            Returns:
                visual.GratingStim: PsychoPy grating stimulus configured as a fixation dot
            """
            fixation = visual.GratingStim(
                win=self.window,
                pos=pos,
                sf=400 if self.use_vr else 0.2,  # High spatial frequency for VR
                color=[1, 0, 0]  # Red color
            )
            fixation.size = 0.02 if self.use_vr else 0.4
            return fixation

        # Create VR block instruction stimuli
        def create_vr_block_instruction(pos):
            """Helper function to create instruction text for VR blocks.

            Args:
                pos (tuple): Position (x, y) for the instruction text

            Returns:
                visual.TextStim: PsychoPy text stimulus with block instructions
            """
            return visual.TextStim(
                win=self.window,
                text="Focus on the red dot, and try not to blink whilst the squares are flashing, "
                     "press the spacebar or pull the controller trigger when ready to commence.",
                color=[-1, -1, -1],  # Black text
                pos=pos,
                height=0.1
            )

        # Create and position stimulus
        def create_eye_stimuli(eye_x_pos, pix_x_pos):
            """Helper function to create a complete set of stimuli for one eye.

            Args:
                eye_x_pos (float): X position in normalized coordinates (-1 to 1)
                pix_x_pos (float): X position in pixels

            Returns:
                dict: Dictionary containing checkerboards, fixation, and instructions for one eye
            """
            return {
                'checkerboards': [
                    create_checkerboard_stim((1, -1), pos=(pix_x_pos, 0)),  # Phase 1: white-black
                    create_checkerboard_stim((-1, 1), pos=(pix_x_pos, 0))   # Phase 2: black-white
                ],
                'fixation': create_fixation_stim([eye_x_pos, 0]),
                'vr_block_instructions': create_vr_block_instruction((eye_x_pos, 0))
            }

        # Structure all stimuli in organized dictionary
        if self.use_vr:
            # Calculate pixel positions for stereoscopic presentation
            # Each eye sees stimuli positioned appropriately for stereoscopic viewing
            window_width = self.window.size[0]
            left_pix_x_pos = self.left_eye_x_pos * (window_width / 2)
            right_pix_x_pos = self.right_eye_x_pos * (window_width / 2)

            return {
                'left': create_eye_stimuli(self.left_eye_x_pos, left_pix_x_pos),
                'right': create_eye_stimuli(self.right_eye_x_pos, right_pix_x_pos)
            }
        else:
            # Monoscopic presentation - centered stimuli
            return {
                'monoscopic': create_eye_stimuli(0, 0)
            }

    def _present_vr_block_instructions(self, open_eye, closed_eye):
        """Present block instructions for VR mode with one eye active and one blocked.

        This internal helper method sets up the VR buffers so that one eye sees the
        instructions and fixation point, while the other eye sees a black screen.

        Args:
            open_eye (str): Eye to show instructions to ('left' or 'right')
            closed_eye (str): Eye to show black screen to ('left' or 'right')

        Note:
            This method only draws to the buffers; it does not flip the window.
            The flip must be called by the parent method.
        """
        # Set buffer to the eye that should see the instructions
        self.window.setBuffer(open_eye)
        self.stim[open_eye]['vr_block_instructions'].draw()
        self.stim[open_eye]['fixation'].draw()

        # Set buffer to the eye that should be blocked
        self.window.setBuffer(closed_eye)
        self.black_background.draw()

    def present_block_instructions(self, current_block: int) -> None:
        """Present instructions at the beginning of each block.

        Instructions inform the participant which eye will be stimulated during the
        upcoming block. For VR mode, the instructions appear in the active eye while
        the other eye sees black. For monitor mode, text instructions ask the participant
        to close the appropriate eye.

        Args:
            current_block (int): The current block number (0-indexed).
                Even blocks (0, 2, 4...) stimulate the left eye.
                Odd blocks (1, 3, 5...) stimulate the right eye.

        Note:
            This method is called repeatedly in a loop until the participant presses
            spacebar or the controller trigger to begin the block. The actual flip()
            is handled by the parent class.
        """
        if self.use_vr:
            # VR mode: Show instructions to one eye, black to the other
            if current_block % 2 == 0:
                # Even block: left eye open
                self._present_vr_block_instructions(open_eye="left", closed_eye="right")
            else:
                # Odd block: right eye open
                self._present_vr_block_instructions(open_eye="right", closed_eye="left")
        else:
            # Monitor mode: Display text instructions for manual eye closure
            if current_block % 2 == 0:
                instruction_text = (
                    "Close your right eye, then focus on the red dot with your left eye. "
                    "Press spacebar or controller when ready."
                )
            else:
                instruction_text = (
                    "Close your left eye, then focus on the red dot with your right eye. "
                    "Press spacebar or controller when ready."
                )
            # Draw instruction text and fixation point
            text = visual.TextStim(win=self.window, text=instruction_text, color=[-1, -1, -1])
            text.draw()
            self.stim['monoscopic']['fixation'].draw()

        # Note: window.flip() is called by the parent class after this method returns

    def present_stimulus(self, idx: int):
        """Present a single pattern reversal stimulus.

        This method is called for each trial in the experiment. It determines which eye
        should be stimulated based on the current block, displays the appropriate phase
        of the checkerboard (alternating black/white pattern), and sends a marker to the
        EEG recording system.

        The checkerboard alternates between two phases (black/white and white/black) with
        each call, creating the pattern reversal effect at 2 Hz (when called with SOA=0.5s).

        Args:
            idx (int): Trial index within the current block (0 to block_trial_size-1).
                Used to determine which checkerboard phase to display (even/odd alternation).

        Note:
            The method pushes EEG markers to enable time-locking analysis:
            - Marker 1: Left eye stimulation
            - Marker 2: Right eye stimulation

            These markers are used during analysis to epoch the EEG data and compute
            averaged VEP waveforms.
        """
        # Get the label of the trial (which eye is being stimulated)
        trial_idx = self.current_block_index * self.block_trial_size + idx
        label = self.parameter[trial_idx]  # 0 = left eye, 1 = right eye

        # Determine which eye should see the stimulus and which should be blocked
        open_eye = 'left' if label == 0 else 'right'
        closed_eye = 'left' if label == 1 else 'right'

        # Draw checkerboard and fixation
        if self.use_vr:
            # VR mode: Draw to the active eye buffer
            self.window.setBuffer(open_eye)
            self.grey_background.draw()
            display = self.stim['left' if label == 0 else 'right']
        else:
            # Monitor mode: Draw to single buffer
            self.black_background.draw()
            display = self.stim['monoscopic']

        # Alternate between two checkerboard phases to create reversal effect
        # idx % 2 gives 0 or 1, selecting between the two pre-created checkerboard phases
        checkerboard_frame = idx % 2
        display['checkerboards'][checkerboard_frame].draw()
        display['fixation'].draw()

        if self.use_vr:
            # VR mode: Draw black screen to the blocked eye
            self.window.setBuffer(closed_eye)
            self.black_background.draw()

        # Flip window to display the stimulus
        self.window.flip()

        # Push EEG marker with timestamp
        # These markers are critical for time-locked averaging during analysis
        marker = self.markernames[label]  # Get marker name for this eye (1 or 2)
        self.eeg.push_sample(marker=marker, timestamp=time())

    def present_iti(self):
        """Present the inter-trial interval (ITI) screen.

        For this experiment, the ITI is set to 0 (continuous presentation), but this
        method is still implemented to maintain compatibility with the base class.
        It displays a black screen in VR mode or flips the window in monitor mode.

        Note:
            In pattern reversal VEP, continuous presentation is standard to maintain
            a steady state of visual stimulation and maximize the number of epochs
            for averaging.
        """
        if self.use_vr:
            # VR mode: Draw black to both eye buffers
            for eye in ['left', 'right']:
                self.window.setBuffer(eye)
                self.black_background.draw()
        # Flip window to display the ITI screen (black in both modes)
        self.window.flip()
