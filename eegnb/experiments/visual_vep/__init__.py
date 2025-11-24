"""Visual Evoked Potential (VEP) experiments module.

This module contains experiments for measuring visual evoked potentials,
including pattern reversal VEP for assessing the P100 component.
"""

from .vep import VisualVEP
from .pattern_reversal_vep import VisualPatternReversalVEP

__all__ = ['VisualVEP', 'VisualPatternReversalVEP']
