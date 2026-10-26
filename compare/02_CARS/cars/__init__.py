"""Full CARS-PLS reproduction for continuous-temperature spectra."""

from .core import (
    EDFSchedule,
    adaptive_reweighted_sample,
    enforce_edf,
    exponentially_decreasing_schedule,
    make_cv_splits,
    normalized_coefficient_weights,
    pls_cross_validated_rmse,
    validate_disjoint_group_splits,
)
from .selector import (
    METHOD_NAME,
    CARSResult,
    CARSSelector,
    band_selection_regression,
)
from .data import (
    BlackbodyTable,
    EnviCubeRecord,
    inspect_blackbody_directory,
    load_blackbody_table,
    open_envi_cube,
    parse_blackbody_record,
    scan_blackbody_directory,
)

__all__ = [
    "METHOD_NAME",
    "CARSResult",
    "CARSSelector",
    "band_selection_regression",
    "EDFSchedule",
    "adaptive_reweighted_sample",
    "enforce_edf",
    "exponentially_decreasing_schedule",
    "make_cv_splits",
    "normalized_coefficient_weights",
    "pls_cross_validated_rmse",
    "validate_disjoint_group_splits",
    "BlackbodyTable",
    "EnviCubeRecord",
    "inspect_blackbody_directory",
    "load_blackbody_table",
    "open_envi_cube",
    "parse_blackbody_record",
    "scan_blackbody_directory",
]
