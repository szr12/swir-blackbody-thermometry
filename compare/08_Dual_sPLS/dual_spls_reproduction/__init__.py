"""Dual-sPLS pseudo-lasso 的论文与官方 R 源码对齐实现。"""

from .selector import DualSPLSModel, DualSPLSSelector, band_selection_regression

__all__ = ["DualSPLSModel", "DualSPLSSelector", "band_selection_regression"]

