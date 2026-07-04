"""VIOLA-SO101: object-centric imitation policy for the SO-101 arm."""

from .config import VIOLAConfig
from .policy import VIOLAPolicy

__all__ = ["VIOLAConfig", "VIOLAPolicy"]
__version__ = "0.1.0"
