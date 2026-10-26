"""Paper-first PyTorch adaptation of BS-Net-FC for tabular spectra."""

from .selector import BSNetFCSelector, SelectionResult, band_selection_tabular

__all__ = ["BSNetFCSelector", "SelectionResult", "band_selection_tabular"]
__version__ = "0.1.0"
