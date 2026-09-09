"""Evaluation pipeline: orchestrates job execution and aggregates results.

This module is the core evaluation engine. ``evaluate.py`` is a thin public API
that builds ``Population`` objects, calls the low-level helpers here to compute
per-feature rows, then aggregates them via ``_aggregation.py``.

## Three-tier aggregation

Raw per-segment rows are collapsed upward in three steps:

1. ``_aggregate_features``: raw rows → one row per ``(domain, feature, segment)``
2. ``_aggregate_groups``: → one row per ``(domain, feature)``, dropping entries
   in ``DEFAULT_REMOVE_FEATURES``
3. ``_aggregate_domains``: → one row per ``domain``, dropping entries in
   ``DEFAULT_REMOVE_GROUPS``

``DEFAULT_REMOVE_FEATURES`` and ``DEFAULT_REMOVE_GROUPS`` (in ``_aggregation.py``)
are hardcoded lists; update them if feature or group names change.

## Output structure

``descriptions`` and ``distances`` are each a ``ResultFrame`` (see
``_result_frame.py``): parallel ``values``/``weights`` DataFrames sharing the
same MultiIndex ``(domain, feature, segment)`` and column set, plus a
``units`` Series. ``descriptions`` has one column per model plus a ``"target"``
column; ``distances`` has one column per model only (there's no such thing as
the target's distance to itself) — the target's distance-side weight, used
only to blend into each model's own weight before aggregating, is tracked
separately (see ``Evaluator._target_distance_weights`` / ``EvalResult``).

## ``missing_distance``

Each ``JobSpec`` carries a ``missing_distance`` value:
- ``1.0`` (timing features): maximum-penalty distance when an activity is absent
  from synthetic schedules.
- ``None`` (participation, transitions): EMD is computed on whatever data exists.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from pandas import DataFrame, MultiIndex, Series, concat

from acteval._jobs import CreativityConfig, JobSpec, StructuralConfig
from acteval._result_frame import ResultFrame
from acteval.features import creativity, structural
from acteval.population import Population


def add_stats(data: DataFrame, columns: dict[str, DataFrame]):
    data["mean"] = data[columns].mean(axis=1)
    data["std"] = data[columns].std(axis=1)


def _aggregate_features(
    descriptions: ResultFrame,
    distances: ResultFrame,
    target_weights: Series,
    extra: list[str] = [],
) -> tuple[DataFrame, DataFrame]:
    """Tier 1: collapse per-segment rows into one row per (domain, feature, segment[, ...]).

    Args:
        target_weights: Raw target distance weights, for blending into each
            model's weight before aggregating distances.
        extra: Additional index levels to preserve after aggregation,
            e.g. ``["label"]`` or ``["label", "cat"]`` for split-aware output.
    """
    grouper = ["domain", "feature", "segment"] + extra

    feat_desc_rf = descriptions.aggregate(grouper)
    feat_desc = feat_desc_rf.values.copy()
    if feat_desc_rf.units is not None:
        feat_desc["unit"] = feat_desc_rf.units

    feat_dist_rf = distances.aggregate_distances(grouper, target_weights=target_weights)
    feat_dist = feat_dist_rf.values.copy()
    # unit comes from descriptions (distances have no target description values)
    feat_dist["unit"] = descriptions.units.groupby(level=grouper).first()

    return feat_desc, feat_dist


_PARALLEL_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Low-level helpers: density features
# ---------------------------------------------------------------------------


def _observed_base(
    spec: JobSpec, observed_features: dict
) -> tuple[Series, Series, tuple]:
    """Build the target-only base rows for a feature spec.

    Returns (value, weight, default): parallel Series with a flat segment
    index, sorted by weight then value descending. The MultiIndex is NOT set
    here; the caller sets it after stacking specs.
    """
    default = _make_default(observed_features)
    weight = spec.size_fn(observed_features)
    value = spec.describe_fn(observed_features)
    order = (
        DataFrame({"weight": weight, "value": value})
        .sort_values(ascending=False, by=["weight", "value"])
        .index
    )
    return value.reindex(order), weight.reindex(order), default


def _model_contribution(
    model: str,
    spec: JobSpec,
    obs_features: dict,
    synth_features: dict,
    default: tuple,
) -> tuple[Series, Series, Series]:
    """Compute weight, description, and distance columns for one model × one spec."""
    synth_weight = spec.size_fn(synth_features)
    desc = _describe_feature(model, synth_features, spec.describe_fn)
    dist = _score_features(
        model,
        obs_features,
        synth_features,
        spec.distance_fn,
        default,
        spec.missing_distance,
    )
    return synth_weight, desc, dist


# ---------------------------------------------------------------------------
# Low-level helpers: creativity
# ---------------------------------------------------------------------------


def _observed_base_creativity(
    target_schedules: DataFrame,
    observed_hash: set,
    config: CreativityConfig,
) -> tuple[Series, Series, Series, Series, Series]:
    """Build target base rows for enabled creativity metrics.

    Args:
        target_schedules: Observed schedule DataFrame.
        observed_hash: Pre-computed population hash set for this split/cat.
        config: Controls which creativity rows to produce.

    Returns:
        (desc_value, desc_weight, desc_unit, dist_weight, dist_unit) — all
        Series sharing the same (domain, feature, segment) index. There is no
        ``dist_value``: the target has no distance to itself.
    """
    obs_diversity = creativity.diversity(target_schedules, observed_hash)
    n = target_schedules.pid.nunique()
    names = ["domain", "feature", "segment"]
    desc_idx, desc_weight, desc_val, desc_unit = [], [], [], []
    dist_idx, dist_weight, dist_unit = [], [], []

    if config.diversity:
        desc_idx.append(("creativity", "diversity", "all"))
        desc_weight.append(n)
        desc_val.append(obs_diversity)
        desc_unit.append("prob. unique")
        dist_idx.append(("creativity", "homogeneity", "all"))
        dist_weight.append(n)
        dist_unit.append("prob. not unique")
    if config.novelty:
        desc_idx.append(("creativity", "novelty", "all"))
        desc_weight.append(n)
        desc_val.append(1)
        desc_unit.append("prob. novel")
        dist_idx.append(("creativity", "conservatism", "all"))
        dist_weight.append(n)
        dist_unit.append("prob. conservative")

    desc_index = MultiIndex.from_tuples(desc_idx, names=names)
    dist_index = MultiIndex.from_tuples(dist_idx, names=names)
    return (
        Series(desc_val, index=desc_index),
        Series(desc_weight, index=desc_index),
        Series(desc_unit, index=desc_index),
        Series(dist_weight, index=dist_index),
        Series(dist_unit, index=dist_index),
    )


def _model_cols_creativity(
    model: str,
    pid_hashes: dict,
    sample_pids,
    observed_hash: set,
    config: CreativityConfig,
) -> tuple[Series, Series, Series, Series]:
    """Compute creativity columns for one model using pre-computed per-pid hashes.

    Args:
        model: Model name.
        pid_hashes: ``{pid: hash_str}`` for the full synthetic population.
        sample_pids: Pid values for this (split, cat) subset.
        observed_hash: Pre-cached hash set for this (split, cat) target subset.
        config: Controls which creativity rows to produce.

    Returns:
        (desc_value, desc_weight, dist_value, dist_weight) — parallel Series.
    """
    y_hash = {pid_hashes[p] for p in sample_pids if p in pid_hashes}
    y_count = len(sample_pids)
    y_diversity = len(y_hash) / y_count if y_count > 0 else 0
    names = ["domain", "feature", "segment"]
    desc_idx, desc_weight, desc_val = [], [], []
    dist_idx, dist_weight, dist_val = [], [], []

    if config.diversity:
        desc_idx.append(("creativity", "diversity", "all"))
        desc_weight.append(y_count)
        desc_val.append(y_diversity)
        dist_idx.append(("creativity", "homogeneity", "all"))
        dist_weight.append(y_count)
        dist_val.append(1 - y_diversity)
    if config.novelty:
        y_novelty = creativity.novelty(observed_hash, y_hash)
        desc_idx.append(("creativity", "novelty", "all"))
        desc_weight.append(y_count)
        desc_val.append(y_novelty)
        dist_idx.append(("creativity", "conservatism", "all"))
        dist_weight.append(y_count)
        dist_val.append(1 - y_novelty)

    desc_index = MultiIndex.from_tuples(desc_idx, names=names)
    dist_index = MultiIndex.from_tuples(dist_idx, names=names)
    return (
        Series(desc_val, index=desc_index, name=model),
        Series(desc_weight, index=desc_index, name=model),
        Series(dist_val, index=dist_index, name=model),
        Series(dist_weight, index=dist_index, name=model),
    )


# ---------------------------------------------------------------------------
# Low-level helpers: structural / feasibility
# ---------------------------------------------------------------------------


def _observed_base_structural(
    target_schedules: DataFrame, config: StructuralConfig
) -> tuple[Series, Series, Series]:
    """Build target base rows for enabled structural (feasibility) metrics.

    Novel-scoped rows use ``value = 0`` (by definition the observed population
    has no novel schedules).

    Returns:
        (value, weight, unit) — parallel Series.
    """
    value_parts, weight_parts, unit_parts = [], [], []
    if config.home_based or config.consecutive:
        w, m = structural.feasibility_eval(
            Population(target_schedules),
            name="target",
            home_based=config.home_based,
            consecutive=config.consecutive,
        )
        value_parts.append(m)
        weight_parts.append(w)
        unit_parts.append(Series("prob. infeasible", index=m.index))
    if config.home_based_novel or config.consecutive_novel:
        idx = structural.feasibility_index(
            home_based=config.home_based_novel,
            consecutive=config.consecutive_novel,
            suffix=" (novel)",
        )
        n = target_schedules.pid.nunique()
        value_parts.append(Series([0.0] * len(idx), index=idx))
        weight_parts.append(Series([n] * len(idx), index=idx))
        unit_parts.append(Series(["prob. infeasible"] * len(idx), index=idx))
    return concat(value_parts), concat(weight_parts), concat(unit_parts)


def _model_cols_structural(
    model: str,
    per_pid_flags: dict,
    synth_dense_pids,
    novel_dense_pids,
    config: StructuralConfig,
) -> tuple[Series, Series]:
    """Compute structural columns for one model using pre-computed per-pid flags.

    Args:
        model: Model name.
        per_pid_flags: Output of ``structural.feasibility`` for the full
            synthetic population, pre-computed once before the split loop.
        synth_dense_pids: Dense pid indices for all persons in this split/cat.
        novel_dense_pids: Dense pid indices for novel persons only (may be None
            if ``config.needs_novel_pids`` is False).
        config: Controls which structural rows to produce.

    Returns:
        (value, weight) — parallel Series (empty if nothing is enabled).
    """
    value_parts, weight_parts = [], []
    if config.home_based or config.consecutive:
        w, m = structural.feasibility_aggregate(
            per_pid_flags,
            synth_dense_pids,
            model,
            home_based=config.home_based,
            consecutive=config.consecutive,
        )
        value_parts.append(m)
        weight_parts.append(w)
    if config.home_based_novel or config.consecutive_novel:
        w, m = structural.feasibility_aggregate(
            per_pid_flags,
            novel_dense_pids,
            model,
            home_based=config.home_based_novel,
            consecutive=config.consecutive_novel,
            suffix=" (novel)",
        )
        value_parts.append(m)
        weight_parts.append(w)
    if not value_parts:
        return Series(dtype=float), Series(dtype=float)
    return concat(value_parts), concat(weight_parts)


def _describe_feature(
    model: str,
    features: dict[str, tuple[np.array, np.array]],
    describe: Callable,
):
    feature_description = describe(features)
    feature_description.name = model
    return feature_description


def _score_features(
    model: str,
    a: dict[str, tuple[np.array, np.array]],
    b: dict[str, tuple[np.array, np.array]],
    distance: Callable,
    default: tuple[np.array, np.array],
    missing_distance=None,
):
    index = list(set(a.keys()) | set(b.keys()))

    def _compute(k):
        if missing_distance is not None and not (
            _feature_present(a, k) and _feature_present(b, k)
        ):
            return missing_distance
        return distance(_get_or_default(a, k, default), _get_or_default(b, k, default))

    if len(index) > _PARALLEL_THRESHOLD:
        with ThreadPoolExecutor() as executor:
            values = list(executor.map(_compute, index))
        metrics = Series(dict(zip(index, values)), name=model)
    else:
        metrics = Series({k: _compute(k) for k in index}, name=model)
    metrics = metrics.fillna(0)
    return metrics


def _feature_present(features, key):
    f = features.get(key)
    return f is not None and len(f[0]) > 0


def _get_or_default(
    features: dict[str, tuple[np.array, np.array]],
    key: str,
    default: tuple[np.array, np.array],
):
    feature = features.get(key)
    if feature is None:
        return default
    support, _ = feature
    if len(support) == 0:
        return default
    return feature


def _make_default(features: dict[str, tuple[np.array, np.array]]):
    default_shape = _infer_feature_shape(features)
    default_support = np.zeros(default_shape)
    return (default_support, np.array([1]))


def _infer_feature_shape(features: dict[str, tuple[np.array, np.array]]) -> np.array:
    for values, _ in iter(features.values()):
        if len(values) > 0:
            default_shape = list(values.shape)
            default_shape[0] = 1
            return default_shape
    return np.array([1])
