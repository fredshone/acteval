"""Aggregation operations and multi-tier post-processing for evaluation results.

Provides two groups of functions:

1. **Per-feature aggregation** (formerly ``ops.py``):
   ``feature_weight``, ``average``, ``time_average``, ``average2d`` — compute
   weighted statistics over a ``{key: (values, weights)}`` feature dict.
   These are the ``describe_fn`` implementations plugged into ``JobSpec``.

2. **Multi-tier collapse** (formerly ``post_process.py``):
   Collapse raw per-segment ``ResultFrame``s upward through the three output
   tiers:

   - ``descriptions_to_group_level`` / ``distances_to_group_level``:
     ``(domain, feature, segment, ...)`` → ``(domain, feature, ...)``
   - ``descriptions_to_domain_level`` / ``distances_to_domain_level``:
     ``(domain, feature, ...)`` → ``(domain, ...)``

   Every function takes and returns a ``ResultFrame`` — callers extract
   ``.values``/``.units`` for the public-facing DataFrame once they've reached
   whichever tier they need (see ``evaluate.py``'s ``ScheduleView``).

   The optional ``extra`` parameter appends additional index levels to the
   grouper — use ``extra=["label"]`` for split-stratified aggregation.

   Pass ``drop=None`` (or ``drop=[]``) to skip the feature-drop step; supply
   your own list of ``(domain, feature, segment, ...)`` tuples to override the
   defaults.
"""

import numpy as np
from numpy import ndarray
from pandas import Series

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
    descriptions: ResultFrame,
    extra: list[str] = [],
    drop: list[tuple] | None = DEFAULT_REMOVE_FEATURES,
) -> ResultFrame:
    """Aggregate feature-level descriptions to group level (domain, feature).

    Args:
        descriptions: ``ResultFrame`` at the segment level.
        extra: Additional index levels to preserve (e.g. ``["label"]``).
        drop: Index prefix tuples to exclude before aggregating.
              Defaults to ``DEFAULT_REMOVE_FEATURES``.  Pass ``None`` or ``[]``
              to skip filtering.
    """
    grouper = ["domain", "feature"] + extra
    rf = descriptions.drop_rows(drop) if drop else descriptions
    return rf.aggregate(grouper)


def distances_to_group_level(
    distances: ResultFrame,
    target_weights: Series,
    extra: list[str] = [],
    drop: list[tuple] | None = DEFAULT_REMOVE_FEATURES,
) -> ResultFrame:
    """Aggregate feature-level distances to group level (domain, feature).

    Args:
        distances: ``ResultFrame`` at the segment level.
        target_weights: Raw per-row target weight, for blending (see
            ``ResultFrame.aggregate_distances``).
        extra: Additional index levels to preserve (e.g. ``["label"]``).
        drop: Index prefix tuples to exclude before aggregating.
              Defaults to ``DEFAULT_REMOVE_FEATURES``.  Pass ``None`` or ``[]``
              to skip filtering.
    """
    grouper = ["domain", "feature"] + extra
    rf = distances.drop_rows(drop) if drop else distances
    return rf.aggregate_distances(grouper, target_weights=target_weights)


def descriptions_to_domain_level(
    group_descriptions: ResultFrame,
    extra: list[str] = [],
    drop: list[tuple] | None = DEFAULT_REMOVE_GROUPS,
) -> ResultFrame:
    """Aggregate group-level descriptions to domain level.

    Uses an unweighted mean so that each feature group contributes equally.

    Args:
        group_descriptions: ``ResultFrame`` at group level.
        extra: Additional index levels to preserve (e.g. ``["label"]``).
        drop: Index prefix tuples to exclude before aggregating.
              Defaults to ``DEFAULT_REMOVE_GROUPS``.  Pass ``None`` or ``[]``
              to skip filtering.
    """
    grouper = ["domain"] + extra
    rf = group_descriptions.drop_rows(drop) if drop else group_descriptions
    return rf.mean(grouper)


def distances_to_domain_level(
    group_distances: ResultFrame,
    extra: list[str] = [],
    drop: list[tuple] | None = DEFAULT_REMOVE_GROUPS,
) -> ResultFrame:
    """Aggregate group-level distances to domain level.

    Uses an unweighted mean so that each feature group contributes equally.

    Args:
        group_distances: ``ResultFrame`` at group level.
        extra: Additional index levels to preserve (e.g. ``["label"]``).
        drop: Index prefix tuples to exclude before aggregating.
              Defaults to ``DEFAULT_REMOVE_GROUPS``.  Pass ``None`` or ``[]``
              to skip filtering.
    """
    grouper = ["domain"] + extra
    rf = group_distances.drop_rows(drop) if drop else group_distances
    return rf.mean(grouper)
