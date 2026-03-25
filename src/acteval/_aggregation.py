"""Aggregation operations and multi-tier post-processing for evaluation results.

Provides two groups of functions:

1. **Per-feature aggregation** (formerly ``ops.py``):
   ``feature_weight``, ``average``, ``time_average``, ``average2d`` — compute
   weighted statistics over a ``{key: (values, weights)}`` feature dict.

2. **Multi-tier collapse** (formerly ``post_process.py``):
   Collapse raw per-segment rows upward through the three output tiers:
   - ``descriptions_to_group_level`` / ``distances_to_group_level``:
     ``(domain, feature, segment, ...)`` → ``(domain, feature, ...)``
   - ``descriptions_to_domain_level`` / ``distances_to_domain_level``:
     ``(domain, feature, ...)`` → ``(domain, ...)``

   The optional ``extra`` parameter appends additional index levels to the
   grouper — use ``extra=["label"]`` for split-stratified aggregation.
"""
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

# ---------------------------------------------------------------------------
# Hardcoded drop-lists for feasibility sub-features
# ---------------------------------------------------------------------------

_REMOVE_FEATURES = [
    ("feasibility", "not home based", "starts"),
    ("feasibility", "not home based", "ends"),
    ("feasibility", "consecutive", "home"),
    ("feasibility", "consecutive", "work"),
    ("feasibility", "consecutive", "education"),
]

_REMOVE_GROUPS = [
    ("feasibility", "not home based"),
    ("feasibility", "consecutive"),
]


def _drop_features(df: DataFrame, features: list[tuple]) -> DataFrame:
    sorted_df = df.sort_index()
    if not features:
        return df
    n = len(features[0])
    feature_set = set(features)
    to_drop = [idx for idx in sorted_df.index if idx[:n] in feature_set]
    if not to_drop:
        return df
    return sorted_df.drop(to_drop, axis=0)


# ---------------------------------------------------------------------------
# Weighted aggregation helpers
# ---------------------------------------------------------------------------


def weighted_average(report: DataFrame, suffix: str = "__weight") -> Series:
    """Weighted average of dataframe using weights in the weight column."""
    cols = [c for c in report.columns if not c.endswith(suffix)]
    scores = DataFrame()
    for c in cols:
        weights = report[f"{c}{suffix}"]
        total = weights.sum()
        scores[c] = report[c] * weights / total
    return scores.sum()


def distance_weighted_average(
    report: DataFrame,
    base_col: str = "observed__weight",
    suffix: str = "__weight",
) -> Series:
    """Weighted average using both model weights and base weights.

    Averaging base and model weights handles cases where models have different
    feature coverage — features present in only one side get half-weight.
    """
    cols = [c for c in report.columns if not c.endswith(suffix)]
    base_weights = report[base_col]
    scores = DataFrame()
    for c in cols:
        weights = (report[f"{c}{suffix}"] + base_weights) / 2
        total = weights.sum()
        scores[c] = report[c] * weights / total
    return scores.sum()


# ---------------------------------------------------------------------------
# Per-feature ops (formerly ops.py)
# ---------------------------------------------------------------------------


def feature_weight(features: dict[str, tuple[ndarray, ndarray]]) -> Series:
    return Series({k: w.sum() for k, (v, w) in features.items()}, dtype=int)


def average(
    features: dict[str, tuple[ndarray, ndarray]], fill_empty: float = 0.0
) -> Series:
    """Weighted average; ``fill_empty`` is returned for labels with no observations."""
    weighted_avg = {}
    for k, (v, w) in features.items():
        weighted_avg[k] = np.average(v, axis=0, weights=w).sum() if w.sum() > 0 else fill_empty
    return Series(weighted_avg, dtype=float)


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


# ---------------------------------------------------------------------------
# Multi-tier collapse (formerly post_process.py)
# ---------------------------------------------------------------------------


def descriptions_to_group_level(
    descriptions: DataFrame | Series, extra: list[str] = []
) -> DataFrame | Series:
    """Aggregate feature-level descriptions to group level (domain, feature)."""
    grouper = ["domain", "feature"] + extra
    desc = _drop_features(descriptions.drop("unit", axis=1), _REMOVE_FEATURES)
    group_desc = desc.groupby(grouper).apply(weighted_average)
    group_desc["unit"] = descriptions["unit"].groupby(grouper).first()
    return group_desc


def distances_to_group_level(
    distances: DataFrame | Series, extra: list[str] = []
) -> DataFrame | Series:
    """Aggregate feature-level distances to group level (domain, feature)."""
    grouper = ["domain", "feature"] + extra
    dist = _drop_features(distances.drop("unit", axis=1), _REMOVE_FEATURES)
    group_dist = dist.groupby(grouper).apply(distance_weighted_average)
    group_dist["unit"] = distances["unit"].groupby(grouper).first()
    return group_dist


def descriptions_to_domain_level(
    group_descriptions: DataFrame | Series, extra: list[str] = []
) -> DataFrame | Series:
    """Aggregate group-level descriptions to domain level."""
    grouper = ["domain"] + extra
    domain_desc = _drop_features(group_descriptions.drop("unit", axis=1), _REMOVE_GROUPS)
    return domain_desc.groupby(grouper).mean()


def distances_to_domain_level(
    group_distances: DataFrame | Series, extra: list[str] = []
) -> DataFrame | Series:
    """Aggregate group-level distances to domain level."""
    grouper = ["domain"] + extra
    domain_dist = _drop_features(group_distances.drop("unit", axis=1), _REMOVE_GROUPS)
    return domain_dist.groupby(grouper).mean()
