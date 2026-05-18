""" 
Initial run of the Experiment Class Refactor base class

Specific experiments are implemented as sub classes that overload a load_stimulus and present_stimulus method

Running each experiment:
obj = VisualP300({parameters})
obj.run()
"""

from abc import abstractmethod, ABC
from typing import Callable
from eegnb.devices.eeg import EEG
from eegnb.devices.vr import VR
from psychopy import prefs, visual, event, core

import gc
import logging
from time import time
import random
import csv

import numpy as np
from pandas import DataFrame

from eegnb import generate_save_fn
from eegnb.experiments import diagnostics
from eegnb.utils.display import snap_refresh_rate

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):

    def __init__(self, exp_name, duration, eeg, save_fn, n_trials: int, iti: float, soa: float, jitter: float,
                 use_vr=False, use_fullscr = True, screen_num=0, stereoscopic = False, devices=None):
        """ Initializer for the Base Experiment Class

        Args:
            exp_name (str): Name of the experiment
            duration (float): Duration of the experiment in seconds
            eeg: EEG device object for recording
            save_fn (str): Save filename function for data
            n_trials (int): Number of trials/stimulus
            iti (float): Inter-trial interval
            soa (float): Stimulus on arrival
            jitter (float): Random delay between stimulus
            use_vr (bool): Use VR for displaying stimulus
            use_fullscr (bool): Use fullscreen mode
            screen_num (int): Screen number (if multiple monitors present)
            stereoscopic (bool): Use stereoscopic rendering for VR
        """

        self.exp_name = exp_name
        self.instruction_text = """\nWelcome to the {} experiment!\nStay still, focus on the centre of the screen, and try not to blink. \nThis block will run for %s seconds.\n
        Press spacebar to continue. \n""".format(self.exp_name)
        self.duration = duration
        self.eeg: EEG = eeg
        self.devices = devices if devices is not None else []
        self.save_fn = save_fn
        self.n_trials = n_trials
        self.iti = iti
        self.soa = soa
        self.jitter = jitter
        self.use_vr = use_vr
        self.screen_num = screen_num
        self.stereoscopic = stereoscopic
        if use_vr:
            # VR interface accessible by specific experiment classes for customizing and using controllers.
            self.vr: VR = VR(monoscopic=not stereoscopic, headLocked=True)

        # Shift the display so it aligns perfectly with the center of each eye.
        if use_vr and stereoscopic:
            self.left_eye_x_pos, self.right_eye_x_pos = self.vr.compute_optical_axis_offsets()
        else:
            self.left_eye_x_pos = 0
            self.right_eye_x_pos = 0

        self.use_fullscr = use_fullscr
        self.window_size = [1600,800]

        # Diagnostic results — populated by run()/setup(), read by show_diagnostics()
        self.signal_check = None
        self.display_check = None

        # Marker observers: callables (trial_idx, timestamp) invoked on every
        # push_marker(). Used by integrations that want timing context but
        # don't emit a hardware/software marker themselves (e.g. VR compositor
        # telemetry, eyetracker fixation logs, photodiode metadata sidecars).
        # Hardware/software *emitters* (Cyton, XID, kernelflow, etc.) live in
        # self.devices and are dispatched via push_sample, not this list.
        self.marker_listeners: list = []
        self.monitor_timing_data: list = []

        if not self.use_vr:
            def _record_monitor_timing(trial_idx, timestamp, flip_time=None):
                # timestamp is software_time (when marker is pushed)
                # flip_time is when the frame actually appeared
                self.monitor_timing_data.append([trial_idx, timestamp, flip_time])
            self.marker_listeners.append(_record_monitor_timing)

        # Initializing the marker names
        self.markernames = [1, 2]

        # Setting up the trial and parameter list
        self.parameter = np.random.binomial(1, 0.5, self.n_trials)
        self.trials = DataFrame(dict(parameter=self.parameter, timestamp=np.zeros(self.n_trials)))


    @abstractmethod
    def load_stimulus(self):
        """ 
        Method that loads the stimulus for the specific experiment, overwritten by the specific experiment
        Returns the stimulus object in the form of [{stim1},{stim2},...]
        Throws error if not overwritten in the specific experiment
        """
        raise NotImplementedError

    @abstractmethod
    def present_stimulus(self, idx : int):
        """
        Method that presents the stimulus for the specific experiment, overwritten by the specific experiment
        Displays the stimulus on the screen
        Pushes EEG Sample if EEG is enabled
        Throws error if not overwritten in the specific experiment

        idx : Trial index for the current trial
        """
        raise NotImplementedError

    def present_iti(self):
        """
        Method that presents the inter-trial interval display for the specific experiment.

        This method defines what is shown on the screen during the period between stimuli.
        It could be a blank screen, a fixation cross, or any other appropriate display.

        This is an optional method - the default implementation simply flips the window with no additional content.
        Subclasses can override this method to provide custom ITI graphics.
        """
        self.window.flip()

    def present_soa(self, idx: int):
        """
        Method called each frame during the SOA wait (stimulus-on period between trial transitions).

        Recommended for VR: override this to redraw the stimulus for trial `idx`. VR compositors
        prefer a freshly drawn frame each submission; submitting only a flip leads
        the compositor to treat frames as stale, which can drop to half-rate
        reprojection and increase dropped/late frames. Overriding gives smoother
        presentation and more accurate frame timing.

        idx : Trial index of the most recently presented stimulus — same value that was
              passed to the preceding present_stimulus call.
        """
        raise NotImplementedError

    def _draw_blank_frame(self):
        """Draw a blank frame and flip — used for display rate measurement."""
        self._draw(self.window.flip)

    def setup(self, instructions=True):
        # Setting up Graphics
        if self.use_vr:
            self.window = self.vr
            self.display_check = self.vr.validate_frame_rate(self._draw_blank_frame)
            # Capture per-marker compositor stats alongside each EEG trigger.
            self.marker_listeners.append(self.vr.log_telemetry)
        else:
            self.window = visual.Window(self.window_size,
                                        monitor="testMonitor",
                                        units="deg",
                                        screen=self.screen_num,
                                        fullscr=self.use_fullscr)
            actual_hz = self.window.getActualFrameRate()
            self.display_check = {
                'target_hz': round(actual_hz, 1) if actual_hz else None,
                'actual_hz': round(actual_hz, 1) if actual_hz else None,
                'deviation_pct': 0.0,
                'ok': actual_hz is not None,
            }
            self.window.mouseVisible = False

        # Snap the target rate to the nearest standard panel rate so
        # downstream stimulus code can rely on a clean integer Hz.
        target = self.display_check.get('target_hz')
        self.refresh_rate = snap_refresh_rate(target) if target else None

        # Loading the stimulus from the specific experiment, throws an error if not overwritten in the specific experiment
        self.stim = self.load_stimulus()

        # Show diagnostics screen first if anything's wrong, then instructions.
        if instructions:
            if not self.show_diagnostics():
                return False
            if not self.show_instructions():
                return False

        # Checking for EEG to setup the EEG stream
        if self.eeg:
            # If no save_fn passed, and data is being streamed, generate a new unnamed save file
            if self.save_fn is None and self.eeg.backend not in ['serialport', 'kernelflow']:
                # Generating a random int for the filename
                random_id = random.randint(1000,10000)
                # Generating save function
                experiment_directory = self.name.replace(' ', '_')
                self.save_fn = generate_save_fn(self.eeg.device_name, experiment_directory, random_id, random_id, data_dir="unnamed")

                print(
                    f"No path for a save file was passed to the experiment. Saving data to {self.save_fn}"
                )
        return True

    def show_instructions(self):
        """ 
        Method that shows the instructions for the specific Experiment
        In the usual case it is not overwritten, the instruction text can be overwritten by the specific experiment
        No parameters accepted, can be skipped through passing a False while running the Experiment
        """

        # Splitting instruction text into lines
        if '%s' in self.instruction_text:
            self.instruction_text = self.instruction_text % self.duration

        # clear/reset any old key/controller events
        self._clear_user_input()

        # Waiting for the user to press the spacebar or controller button or trigger to start the experiment
        while not self._user_input('start'):
            # Displaying the instructions on the screen
            text = visual.TextStim(win=self.window, text=self.instruction_text, color=[-1, -1, -1])
            self._draw(lambda: self.__draw_instructions(text))

            if self._user_input('cancel'):
                return False
        return True

    def show_diagnostics(self):
        """Display a pre-experiment diagnostics screen — only when flagged.

        Shows synthetic-device warning, degraded-framerate warning, and
        noisey electrode warning. If none are flagged the screen is skipped
        entirely so the experimenter goes straight to the instructions.
        Returns True to continue, False if the user cancels.
        """
        warnings = diagnostics.format_diagnostic_warnings(
            device_name=self.eeg.device_name if self.eeg else None,
            display=self.display_check,
            signal_check=self.signal_check,
        )
        if not warnings:
            return True

        body = "\n\n".join(warnings)
        body += "\n\nFix the issues above, or press spacebar / index trigger to continue anyway."

        self._clear_user_input()

        while not self._user_input('start'):
            text = visual.TextStim(
                win=self.window, text=body,
                color=[1, 1, 1], font='Courier New',
                height=0.04, wrapWidth=1.8, units='norm',
                anchorHoriz='center', anchorVert='center',
            )
            self._draw(lambda: self.__draw_instructions(text))
            if self._user_input('cancel'):
                return False
        return True

    def show_results(self, text):
        """Display a results / summary screen after the experiment.

        Mirrors ``show_instructions``: loops the render loop until user input. Uses a
        monospaced font so tabular output (e.g. recording quality reports)
        aligns correctly.

        Args:
            text (str): Text to display. Long lines are wrapped automatically.
        """
        self._clear_user_input()

        stim = visual.TextStim(
            win=self.window, text=text,
            color=[1, 1, 1],          # white on black background
            font='Courier New',
            height=0.04,
            wrapWidth=1.8,
            units='norm',
            anchorHoriz='center', anchorVert='center',
        )

        while not self._user_input('start'):
            self._draw(lambda: self.__draw_results(stim))
            if self._user_input('cancel'):
                break

    def __draw_results(self, stim):
        if self.use_vr and self.stereoscopic:
            for eye, x_pos in [("left", self.left_eye_x_pos),
                                ("right", self.right_eye_x_pos)]:
                self.window.setBuffer(eye)
                stim.pos = (x_pos, 0)
                stim.draw()
        else:
            stim.draw()
        self.window.flip()

    def post_run(self):
        """Called after the trial loop ends, before the window closes.

        Default: display a recording quality report so the experimenter can
        decide whether to re-record before removing the electrodes. Override in a
        subclass or reassign on the instance to replace this behaviour.
        """
        if not self.save_fn:
            return
        try:
            text = diagnostics.post_session_report(self.save_fn)
            text += "\n\nPress spacebar or index trigger to close."
            self.show_results(text)
        except Exception as e:
            print(f"[post_run] Could not display quality report: {e}")

    def _user_input(self, input_type):
        if input_type == 'start':
            key_input = 'spacebar'
            vr_inputs = [
                ('RightTouch', 'A', True),
                ('LeftTouch', 'X', True),
                ('Xbox', 'A', None)
            ]

        elif input_type == 'cancel':
            key_input = 'escape'
            vr_inputs = [
                ('RightTouch', 'B', False),
                ('LeftTouch', 'Y', False),
                ('Xbox', 'B', None)
            ]

        else:
            raise Exception(f'Invalid input_type: {input_type}')

        if len(event.getKeys(keyList=key_input)) > 0:
            return True

        if self.use_vr:
            for controller, button, trigger in vr_inputs:
                if self.get_vr_input(controller, button, trigger):
                    return True

        return False

    def get_vr_input(self, vr_controller, button=None, trigger=False):
        """
        Method that returns True if the user presses the corresponding vr controller button or trigger
        Args:
            vr_controller: 'Xbox', 'LeftTouch' or 'RightTouch'
            button: None, 'A', 'B', 'X' or 'Y'
            trigger (bool): Set to True for trigger

        Returns:

        """
        trigger_squeezed = False
        if trigger:
            for x in self.vr.getIndexTriggerValues(vr_controller):
                if x > 0.0:
                    trigger_squeezed = True

        button_pressed = False
        if button is not None:
            button_pressed, tsec = self.vr.getButtons([button], vr_controller, 'released')

        if trigger_squeezed or button_pressed:
            return True

        return False

    def __draw_instructions(self, text):
        if self.use_vr and self.stereoscopic:
            for eye, x_pos in [("left", self.left_eye_x_pos), ("right", self.right_eye_x_pos)]:
                self.window.setBuffer(eye)
                text.pos = (x_pos, 0)
                text.draw()
        else:
            text.draw()
        self.window.flip()

    def _draw(self, present_stimulus: Callable):
        """
        Set the current eye position and projection for all given stimulus,
        then draw all stimulus and flip the window/buffer
         """
        if self.use_vr:
            tracking_state = self.window.getTrackingState()
            self.window.calcEyePoses(tracking_state.headPose.thePose)
            self.window.setDefaultView()

        present_stimulus()

    def _clear_user_input(self):
        event.getKeys()
        self.clear_vr_input()

    def clear_vr_input(self):
        """
        Clears/resets input events from vr controllers
        """
        if self.use_vr:
            self.vr.updateInputState()
        
    def _run_trial_loop(self, start_time, duration):
        """
        Run the trial presentation loop
        
        This method handles the common trial presentation logic.
        
        Args:
            start_time (float): Time when the trial loop started
            duration (float): Maximum duration of the trial loop in seconds

        """

        if self.use_vr and type(self).present_soa is BaseExperiment.present_soa:
            raise NotImplementedError(
                f"{type(self).__name__} uses VR but does not override present_soa(idx). "
                "psychxr does not honor setAutoDraw, and the VR compositor requires per-frame "
                "redraws during the SOA wait; the default flip-only implementation will blank "
                "the stimulus after one frame. Override present_soa(idx) to redraw your stimulus."
            )

        def iti_with_jitter():
            return self.iti + np.random.rand() * self.jitter

        # Initialize trial variables
        current_trial = trial_end_time = -1
        trial_start_time = None
        rendering_trial = -1
        has_soa_override = type(self).present_soa is not BaseExperiment.present_soa
        
        # Clear/reset user input buffer
        self._clear_user_input()
        
        # Run the trial loop
        while (time() - start_time) < duration:
            elapsed_time = time() - start_time
            
            # Do not present stimulus until current trial begins(Adhere to inter-trial interval).
            if elapsed_time > trial_end_time:
                current_trial += 1
                
                # Calculate timing for this trial
                trial_start_time = elapsed_time + iti_with_jitter()
                trial_end_time = trial_start_time + self.soa

            # Do not present stimulus after trial has ended(stimulus on arrival interval).
            if elapsed_time >= trial_start_time:
                # if current trial number changed present new stimulus.
                if current_trial > rendering_trial:
                    # Stimulus presentation overwritten by specific experiment
                    self._draw(lambda: self.present_stimulus(current_trial))
                    rendering_trial = current_trial
                elif has_soa_override:
                    # Keep submitting frames during SOA wait — VR compositor
                    # drops to lower framerate if we stall between reversals.
                    self._draw(lambda: self.present_soa(current_trial))
            else:
                self._draw(lambda: self.present_iti())

            if self._user_input('cancel'):
                return False

        return True

    def _enable_frame_tracking(self):
        """Enable per-frame interval recording for dropped frame diagnostics."""
        self.window.recordFrameIntervals = True
        rate = self.window.displayRefreshRate if self.use_vr else self.window.getActualFrameRate()
        self.display_refresh_rate = int(np.round(rate)) if rate else None
        # Threshold for counting a frame as "dropped" — 50% over expected duration
        expected_frame_dur = 1.0 / (rate or 60)
        self.window.refreshThreshold = expected_frame_dur * 1.5

    def _save_monitor_telemetry(self):
        """Saves memory-buffered monitor timing telemetry to a CSV sidecar."""
        if self.save_fn is None or not self.monitor_timing_data:
            return

        timing_path = self.save_fn.with_name(self.save_fn.stem + '_timing.csv')
        with open(timing_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial_idx', 'software_time', 'flip_time'])
            writer.writerows(self.monitor_timing_data)
        print(f"  Saved monitor timing telemetry to {timing_path}")

    def run(self, instructions=True):
        """ Run the experiment """
        self.signal_check = diagnostics.check_signal_quality(self.eeg)

        # Setup the experiment
        self.setup(instructions)

        # Start EEG Stream, wait for signal to settle, and then pull timestamp for start point
        if self.eeg:
            if self.eeg.backend not in ['serialport']:
                print("Wait for the EEG-stream to start...")
                self.eeg.start(self.save_fn, duration=self.duration + 5)
                print("EEG Stream started")

        self._enable_frame_tracking()

        # Record experiment until a key is pressed or duration has expired.
        record_start_time = time()

        core.rush(True)
        gc.disable()
        try:
            if self.use_vr:        
                self.vr.sync_vr_clock()
            self._run_trial_loop(record_start_time, self.duration)
        finally:
            gc.enable()
            core.rush(False)

        # Clearing the screen for the next trial
        event.clearEvents()

        # Closing the EEG stream
        if self.eeg:
            self.eeg.stop()

        if self.use_vr:
            self.vr.save_telemetry(self.save_fn)
            self.vr.save_frame_stats(self.save_fn)
        else:
            self._save_monitor_telemetry()

        # Post-run hook (e.g. show a summary / quality report screen)
        self.post_run()

        # Closing the window
        self.window.close()



    def push_marker(self, marker, trial_idx):
        """Push a trigger to the primary EEG and every additional device in
        self.devices, then notify any registered marker_listeners with
        (trial_idx, timestamp).

        Emitters (self.eeg, self.devices) record the marker value into their
        respective streams — Cyton's marker channel, XID's TTL output,
        muselsl's lsl marker stream, etc.
        Listeners (self.marker_listeners) receive only the timing context —
        they're observers that capture additional state at marker time
        (VR compositor telemetry, eyetracker fixation, photodiode metadata).
        """
        timestamp = time()
        if self.eeg:
            self.eeg.push_sample(marker=marker, timestamp=timestamp)
        for dev in self.devices:
            dev.push_sample(marker=marker, timestamp=timestamp)
        for listener in self.marker_listeners:
            try:
                listener(trial_idx, timestamp)
            except Exception:
                logger.exception("marker listener failed: %s", listener)

    @property
    def name(self) -> str:
        """ This experiment's name """
        return self.exp_name
