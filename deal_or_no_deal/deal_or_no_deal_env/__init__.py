"""Deal or No Deal Gym-compatible environment.

This package provides a flexible RL environment for the Deal or No Deal game.
"""

from .version import __version__
from .registration import register_deal_or_no_deal

__all__ = ["__version__", "register_deal_or_no_deal"]


