"""
Experiment base classes.

BaseDisplay   — PsychoPy window, VR rendering, and input handling.
                Subclass this when you need a window but not the EEG trial loop.

BaseExperiment(BaseDisplay) — adds EEG, timed ITI/SOA trial loop, and parameter arrays.
                Specific experiments subclass this and implement load_stimulus / present_stimulus.

Running an experiment:
    obj = VisualP300({parameters})
    obj.run()
"""

from abc import abstractmethod, ABC
from typing import Callable
from eegnb.devices.eeg import EEG
from psychopy import prefs
from psychopy.visual.rift import Rift

from time import time
import random

import numpy as np
from pandas import DataFrame
from psychopy import visual, event

from eegnb import generate_save_fn


class BaseDisplay:
    """
    PsychoPy window + VR rendering + input handling.

    Provides:
      - Window / Rift creation via setup_window()
      - Per-frame VR eye-pose setup via _draw()
      - Keyboard and VR controller input via _user_input()
      - Instruction screen via show_instructions()

    Does NOT include: EEG, trial timing, ITI/SOA, parameter arrays.
    """

    def __init__(self, exp_name, use_vr=False, use_fullscr=True,
                 screen_num=0, stereoscopic=False):
        self.exp_name = exp_name
        self.instruction_text = ""
        self.use_vr = use_vr
        self.screen_num = screen_num
        self.stereoscopic = stereoscopic
        self.use_fullscr = use_fullscr
        self.window_size = [1600, 800]

        if use_vr:
            self.rift: Rift = visual.Rift(monoscopic=not stereoscopic, headLocked=True)

        if stereoscopic:
            self.left_eye_x_pos = 0.2
            self.right_eye_x_pos = -0.2
        else:
            self.left_eye_x_pos = 0
            self.right_eye_x_pos = 0

    def setup_window(self):
        """Create the PsychoPy window (or bind the Rift for VR).
        Override in subclasses to change units, background colour, etc."""
        self.window = (
            self.rift if self.use_vr
            else visual.Window(
                self.window_size, monitor="testMonitor", units="deg",
                screen=self.screen_num, fullscr=self.use_fullscr,
            )
        )

    def show_instructions(self, text_color=None):
        """Display instruction screen and wait for spacebar / VR trigger.

        Args:
            text_color: PsychoPy colour for text. Defaults to black [-1, -1, -1].

        Returns:
            True if the user started, False if they cancelled.
        """
        if text_color is None:
            text_color = [-1, -1, -1]

        self.window.mouseVisible = False
        self._clear_user_input()

        while not self._user_input('start'):
            text = visual.TextStim(win=self.window, text=self.instruction_text,
                                   color=text_color)
            self._draw(lambda: self._draw_instructions(text))
            self.window.mouseVisible = True
            if self._user_input('cancel'):
                return False
        return True

    def _draw(self, fn: Callable):
        """Apply VR eye-pose setup (if needed) then call fn().

        fn must draw all stimuli AND call window.flip().
        """
        if self.use_vr:
            tracking_state = self.window.getTrackingState()
            self.window.calcEyePoses(tracking_state.headPose.thePose)
            self.window.setDefaultView()
        fn()

    def _draw_instructions(self, text):
        if self.use_vr and self.stereoscopic:
            for eye, x_pos in [("left", self.left_eye_x_pos),
                                ("right", self.right_eye_x_pos)]:
                self.window.setBuffer(eye)
                text.pos = (x_pos, 0)
                text.draw()
        else:
            text.draw()
        self.window.flip()

    def _user_input(self, input_type):
        if input_type == 'start':
            key_input = 'spacebar'
            vr_inputs = [
                ('RightTouch', 'A', True),
                ('LeftTouch', 'X', True),
                ('Xbox', 'A', None),
            ]
        elif input_type == 'cancel':
            key_input = 'escape'
            vr_inputs = [
                ('RightTouch', 'B', False),
                ('LeftTouch', 'Y', False),
                ('Xbox', 'B', None),
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
        """Return True if the given VR controller button or trigger is activated."""
        trigger_squeezed = False
        if trigger:
            for x in self.rift.getIndexTriggerValues(vr_controller):
                if x > 0.0:
                    trigger_squeezed = True

        button_pressed = False
        if button is not None:
            button_pressed, tsec = self.rift.getButtons([button], vr_controller, 'released')

        return trigger_squeezed or button_pressed

    def _clear_user_input(self):
        event.getKeys()
        self.clear_vr_input()

    def clear_vr_input(self):
        """Clear pending VR controller input events."""
        if self.use_vr:
            self.rift.updateInputState()

    @property
    def name(self) -> str:
        return self.exp_name


class BaseExperiment(BaseDisplay, ABC):
    """
    Adds EEG recording, timed ITI/SOA trial loop, and parameter arrays to BaseDisplay.

    Subclasses must implement:
        load_stimulus()       — build and return stimulus objects
        present_stimulus(idx) — draw the stimulus for trial idx and flip the window
    """

    def __init__(self, exp_name, duration, eeg, save_fn, n_trials: int,
                 iti: float, soa: float, jitter: float,
                 use_vr=False, use_fullscr=True, screen_num=0,
                 stereoscopic=False, devices=list):
        super().__init__(exp_name, use_vr, use_fullscr, screen_num, stereoscopic)

        self.instruction_text = (
            "\nWelcome to the {} experiment!\n"
            "Stay still, focus on the centre of the screen, and try not to blink. \n"
            "This block will run for %s seconds.\n\n"
            "        Press spacebar to continue. \n"
        ).format(self.exp_name)

        self.duration = duration
        self.eeg: EEG = eeg
        self.devices = devices
        self.save_fn = save_fn
        self.n_trials = n_trials
        self.iti = iti
        self.soa = soa
        self.jitter = jitter

        self.markernames = [1, 2]

        self.parameter = np.random.binomial(1, 0.5, self.n_trials)
        self.trials = DataFrame(dict(parameter=self.parameter,
                                     timestamp=np.zeros(self.n_trials)))

    @abstractmethod
    def load_stimulus(self):
        """Build stimulus objects. Called from setup(); return value stored as self.stim."""
        raise NotImplementedError

    @abstractmethod
    def present_stimulus(self, idx: int):
        """Draw the stimulus for trial idx and flip the window."""
        raise NotImplementedError

    def present_iti(self):
        """Draw the inter-trial interval display. Override for custom ITI graphics."""
        self.window.flip()

    def setup(self, instructions=True):
        """Create window, load stimulus, show instructions, and initialise EEG."""
        self.setup_window()
        self.stim = self.load_stimulus()

        if instructions:
            # Format duration into instruction text before display
            self.instruction_text = self.instruction_text % self.duration
            if not self.show_instructions():
                return False

        if self.eeg:
            if self.save_fn is None and self.eeg.backend not in ['serialport', 'kernelflow']:
                random_id = random.randint(1000, 10000)
                experiment_directory = self.name.replace(' ', '_')
                self.save_fn = generate_save_fn(
                    self.eeg.device_name, experiment_directory,
                    random_id, random_id, data_dir="unnamed",
                )
                print(f"No path for a save file was passed to the experiment. "
                      f"Saving data to {self.save_fn}")

        return True

    def _run_trial_loop(self, start_time, duration):
        """Run the trial presentation loop.

        Args:
            start_time: time() value when the loop started
            duration:   maximum wall-clock seconds to run
        """
        def iti_with_jitter():
            return self.iti + np.random.rand() * self.jitter

        current_trial = trial_end_time = -1
        trial_start_time = None
        rendering_trial = -1

        self._clear_user_input()

        while (time() - start_time) < duration:
            elapsed_time = time() - start_time

            # Advance to next trial once the current SOA window has elapsed
            if elapsed_time > trial_end_time:
                current_trial += 1
                trial_start_time = elapsed_time + iti_with_jitter()
                trial_end_time = trial_start_time + self.soa

            if elapsed_time >= trial_start_time:
                if current_trial > rendering_trial:
                    self._draw(lambda: self.present_stimulus(current_trial))
                    rendering_trial = current_trial
            else:
                self._draw(lambda: self.present_iti())

            if self._user_input('cancel'):
                return False

        return True

    def run(self, instructions=True):
        """Run the full experiment: setup → EEG start → trial loop → EEG stop → close."""
        self.setup(instructions)

        if self.eeg:
            if self.eeg.backend not in ['serialport']:
                print("Wait for the EEG-stream to start...")
                self.eeg.start(self.save_fn, duration=self.duration + 5)
                print("EEG Stream started")

        record_start_time = time()
        self._run_trial_loop(record_start_time, self.duration)

        event.clearEvents()

        if self.eeg:
            self.eeg.stop()

        self.window.close()

    def send_triggers(self, marker):
        """Send timing triggers to all registered recording devices."""
        for dev in self.devices:
            timestamp = time()
            dev.push_sample(marker=marker, timestamp=timestamp)
