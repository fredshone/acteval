"""Aggregation operations and multi-tier post-processing for evaluation results.

Provides two groups of functions:

1. **Per-feature aggregation** (formerly ``ops.py``):
   ``feature_weight``, ``average``, ``time_average``, ``average2d`` — compute
   weighted statistics over a ``{key: (values, weights)}`` feature dict.

2. **Multi-tier collapse** (formerly ``post_process.py``):
   Collapse raw per-segment rows upward through the three output tiers using
   ``ResultFrame`` for clean value/weight/unit separation:

   - ``descriptions_to_group_level`` / ``distances_to_group_level``:
     ``(domain, feature, segment, ...)`` → ``(domain, feature, ...)``
   - ``descriptions_to_domain_level`` / ``distances_to_domain_level``:
     ``(domain, feature, ...)`` → ``(domain, ...)``

   The optional ``extra`` parameter appends additional index levels to the
   grouper — use ``extra=["label"]`` for split-stratified aggregation.

   Pass ``drop=None`` (or ``drop=[]``) to skip the feature-drop step; supply
   your own list of ``(domain, feature, segment, ...)`` tuples to override the
   defaults.
"""

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series

from acteval._result_frame import ResultFrame

# ---------------------------------------------------------------------------
# Default drop-lists for feasibility sub-features
# ---------------------------------------------------------------------------
# These are the *defaults* for the tier-collapse functions below.  Pass a
# different list (or ``None``) to override per call.

DEFAULT_REMOVE_FEATURES: list[tuple] = [
    ("feasibility", "not home based", "starts"),
    ("feasibility", "not home based", "ends"),
    ("feasibility", "consecutive", "home"),
    ("feasibility", "consecutive", "work"),
    ("feasibility", "consecutive", "education"),
]

DEFAULT_REMOVE_GROUPS: list[tuple] = [
    ("feasibility", "not home based"),
    ("feasibility", "consecutive"),
]

# Backward-compatible aliases (keep old names importable).
_REMOVE_FEATURES = DEFAULT_REMOVE_FEATURES
_REMOVE_GROUPS = DEFAULT_REMOVE_GROUPS


def _drop_features(df: DataFrame, features: list[tuple]) -> DataFrame:
    """Drop rows from *df* whose index prefix matches any entry in *features*.

    .. deprecated::
        Use ``ResultFrame.drop_rows()`` instead.  This function is kept for
        any callers that still operate on plain DataFrames.
    """
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
# Weighted aggregation helpers (kept for backward compatibility)
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
        weighted_avg[k] = (
            np.average(v, axis=0, weights=w).sum() if w.sum() > 0 else fill_empty
        )
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
# Multi-tier collapse
# ---------------------------------------------------------------------------


def descriptions_to_group_level(
    descriptions: DataFrame | ResultFrame,
    extra: list[str] = [],
    drop: list[tuple] | None = DEFAULT_REMOVE_FEATURES,
) -> DataFrame:
    """Aggregate feature-level descriptions to group level (domain, feature).

    Args:
        descriptions: Wide-format DataFrame or ``ResultFrame`` at the segment level.
        extra: Additional index levels to preserve (e.g. ``["label"]``).
        drop: Index prefix tuples to exclude before aggregating.
              Defaults to ``DEFAULT_REMOVE_FEATURES``.  Pass ``None`` or ``[]``
              to skip filtering.
    """
    grouper = ["domain", "feature"] + extra
    rf = (
        descriptions
        if isinstance(descriptions, ResultFrame)
        else ResultFrame.from_wide(descriptions)
    )
    if drop:
        rf = rf.drop_rows(drop)
    group_rf = rf.aggregate(grouper)
    out = group_rf.values.copy()
    if group_rf.units is not None:
        out["unit"] = group_rf.units
    return out


def distances_to_group_level(
    distances: DataFrame | ResultFrame,
    extra: list[str] = [],
    drop: list[tuple] | None = DEFAULT_REMOVE_FEATURES,
) -> DataFrame:
    """Aggregate feature-level distances to group level (domain, feature).

    Args:
        distances: Wide-format DataFrame or ``ResultFrame`` at the segment level.
        extra: Additional index levels to preserve (e.g. ``["label"]``).
        drop: Index prefix tuples to exclude before aggregating.
              Defaults to ``DEFAULT_REMOVE_FEATURES``.  Pass ``None`` or ``[]``
              to skip filtering.
    """
    grouper = ["domain", "feature"] + extra
    rf = (
        distances
        if isinstance(distances, ResultFrame)
        else ResultFrame.from_wide(distances)
    )
    if drop:
        rf = rf.drop_rows(drop)
    group_rf = rf.aggregate_distances(grouper)
    out = group_rf.values.copy()
    if group_rf.units is not None:
        out["unit"] = group_rf.units
    return out


def descriptions_to_domain_level(
    group_descriptions: DataFrame | ResultFrame,
    extra: list[str] = [],
    drop: list[tuple] | None = DEFAULT_REMOVE_GROUPS,
) -> DataFrame:
    """Aggregate group-level descriptions to domain level.

    Uses an unweighted mean so that each feature group contributes equally.

    Args:
        group_descriptions: Wide-format DataFrame or ``ResultFrame`` at group level.
        extra: Additional index levels to preserve (e.g. ``["label"]``).
        drop: Index prefix tuples to exclude before aggregating.
              Defaults to ``DEFAULT_REMOVE_GROUPS``.  Pass ``None`` or ``[]``
              to skip filtering.
    """
    grouper = ["domain"] + extra
    rf = (
        group_descriptions
        if isinstance(group_descriptions, ResultFrame)
        else ResultFrame.from_wide(group_descriptions)
    )
    if drop:
        rf = rf.drop_rows(drop)
    domain_rf = rf.mean(grouper)
    # Domain level has no unit column — return values only (matching existing behaviour).
    return domain_rf.values


def distances_to_domain_level(
    group_distances: DataFrame | ResultFrame,
    extra: list[str] = [],
    drop: list[tuple] | None = DEFAULT_REMOVE_GROUPS,
) -> DataFrame:
    """Aggregate group-level distances to domain level.

    Uses an unweighted mean so that each feature group contributes equally.

    Args:
        group_distances: Wide-format DataFrame or ``ResultFrame`` at group level.
        extra: Additional index levels to preserve (e.g. ``["label"]``).
        drop: Index prefix tuples to exclude before aggregating.
              Defaults to ``DEFAULT_REMOVE_GROUPS``.  Pass ``None`` or ``[]``
              to skip filtering.
    """
    grouper = ["domain"] + extra
    rf = (
        group_distances
        if isinstance(group_distances, ResultFrame)
        else ResultFrame.from_wide(group_distances)
    )
    if drop:
        rf = rf.drop_rows(drop)
    domain_rf = rf.mean(grouper)
    return domain_rf.values
