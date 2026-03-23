import numpy as np
from numpy import ndarray
from pandas import Series


def feature_weight(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: w.sum() for k, (v, w) in features.items()}, dtype=int)


def average(
    features: dict[str, tuple[ndarray, ndarray]], fill_empty: float = 0.0
) -> Series:
    """Weighted average; ``fill_empty`` is returned for labels with no observations."""
    weighted_average = {}
    for k, (v, w) in features.items():
        weighted_average[k] = np.average(v, axis=0, weights=w).sum() if w.sum() > 0 else fill_empty
    return Series(weighted_average, dtype=float)


def time_average(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    """Weighted average; returns NaN (not 0) for labels with no observations."""
    return average(features, fill_empty=np.nan)


def average2d(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    """2-D weighted average; omits zero-weight labels."""
    return Series(
        {
            k: np.average(v, axis=0, weights=w).sum().sum()
            for k, (v, w) in features.items()
            if w.sum() > 0
        },
        dtype=float,
    )
