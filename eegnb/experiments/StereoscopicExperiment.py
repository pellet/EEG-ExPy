"""
StereoscopicExperiment - A wrapper class for BaseExperiment that provides proper VR stereoscopic rendering.

This class wraps any BaseExperiment instance and overrides the drawing method to render
separately for left and right eyes, enabling proper stereoscopic VR display.

The wrapper maintains all the functionality of the wrapped BaseExperiment while adding
stereoscopic rendering capabilities.

Usage:
    from eegnb.experiments import BaseExperiment, StereoscopicExperiment
    
    # Create your base experiment
    base_exp = SomeExperiment(...)
    
    # Wrap it for stereoscopic rendering
    stereo_exp = StereoscopicExperiment(base_exp)
    
    # Run with stereoscopic VR rendering
    stereo_exp.run()
"""

from typing import Callable
from .Experiment import BaseExperiment


class StereoscopicExperiment:
    """
    A wrapper class that adds stereoscopic VR rendering to any BaseExperiment.
    
    This class acts as a transparent wrapper around BaseExperiment instances,
    delegating all method calls and attribute access to the wrapped experiment
    while overriding the drawing method to provide proper left/right eye rendering.
    """
    
    def __init__(self, base_experiment: BaseExperiment):
        """
        Initialize the stereoscopic wrapper.
        
        Args:
            base_experiment: The BaseExperiment instance to wrap
        """
        self._base_experiment = base_experiment
        
        # Ensure the base experiment is configured for VR
        if not base_experiment.use_vr:
            raise ValueError("Base experiment must be configured with use_vr=True for stereoscopic rendering")
    
    def __getattr__(self, name):
        """
        Delegate all attribute access to the wrapped base experiment.
        
        This allows the wrapper to be used transparently as if it were
        the original BaseExperiment instance.
        """
        return getattr(self._base_experiment, name)
    
    def _BaseExperiment__draw(self, present_stimulus: Callable):
        """
        Override the private __draw method to provide stereoscopic rendering.
        
        This method renders the stimulus separately for left and right eyes,
        providing proper stereoscopic VR display.
        
        Args:
            present_stimulus: Callable that presents the stimulus
        """
        # Get the VR window (rift instance)
        window = self._base_experiment.window
        
        # Get tracking state for head position
        tracking_state = window.getTrackingState()
        
        # Calculate eye poses based on head pose
        window.calcEyePoses(tracking_state.headPose.thePose)
        
        # Render for left eye
        window.setBuffer('left')
        window.setProjectionMatrix(window.projectionMatrix[0])
        window.setViewMatrix(window.viewMatrix[0])
        present_stimulus()
        
        # Render for right eye  
        window.setBuffer('right')
        window.setProjectionMatrix(window.projectionMatrix[1])
        window.setViewMatrix(window.viewMatrix[1])
        present_stimulus()
        
        # Submit the frame to the VR system
        window.flip()
    
    def run(self, instructions=True):
        """
        Run the experiment with stereoscopic rendering.
        
        This method uses the BaseExperiment's trial loop with the
        stereoscopic drawing method, eliminating code duplication.
        
        Args:
            instructions: Whether to show instructions before starting
        """
        # Use the base experiment's trial loop with our stereoscopic drawing method
        self._base_experiment._run_trial_loop(instructions, self._BaseExperiment__draw)
