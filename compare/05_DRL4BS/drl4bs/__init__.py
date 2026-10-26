"""Non-official, paper-first DRL4BS adaptation for nonnegative tabular spectra."""

from .selector import DRL4BSSelector, SelectionResult, band_selection_tabular

__all__ = ["DRL4BSSelector", "SelectionResult", "band_selection_tabular"]
__version__ = "0.1.0"
