"""Evaluation pipeline: orchestrates job execution and aggregates results.

This module is the core evaluation engine. ``evaluate.py`` is a thin public API
that builds ``Population`` objects and calls into here.

## Entry points

- ``describe()``: applies three-tier aggregation and returns the six output
  DataFrames that ``compare()`` exposes.
- ``describe_labels()``: same but keyed by label (used for split evaluation).

## Three-tier aggregation

Raw per-segment rows are collapsed upward in three steps:

1. ``_aggregate_features``: raw rows → one row per ``(domain, feature, segment)``
2. ``_aggregate_groups``: → one row per ``(domain, feature)``, dropping entries
   in ``_REMOVE_FEATURES``
3. ``_aggregate_domains``: → one row per ``domain``, dropping entries in
   ``_REMOVE_GROUPS``

``_REMOVE_FEATURES`` and ``_REMOVE_GROUPS`` are hardcoded lists; update them if
feature or group names change.

## Output DataFrame structure

``descriptions`` and ``distances`` share the same MultiIndex
``(domain, feature, segment)``. Columns are ``observed__weight``, ``observed``,
then one column per model name plus a ``{model}__weight`` column for each.
``unit`` is a string column carried alongside.

## ``missing_distance``

Each ``JobSpec`` carries a ``missing_distance`` value:
- ``1.0`` (timing features): maximum-penalty distance when an activity is absent
  from synthetic schedules.
- ``None`` (participation, transitions): EMD is computed on whatever data exists.
"""

import warnings
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
    descriptions: DataFrame,
    distances: DataFrame,
    extra: list[str] = [],
) -> tuple[DataFrame, DataFrame]:
    """Tier 1: collapse per-segment rows into one row per (domain, feature, segment[, ...]).

    Uses ``ResultFrame`` internally to avoid fragile suffix-based column
    filtering; the returned DataFrames preserve the existing schema
    (values + unit, no weight columns).

    Args:
        extra: Additional index levels to preserve after aggregation,
            e.g. ``["label"]`` or ``["label", "cat"]`` for split-aware output.
    """
    grouper = ["domain", "feature", "segment"] + extra

    desc_rf = ResultFrame.from_wide(descriptions)
    feat_desc_rf = desc_rf.aggregate(grouper)
    feat_desc = feat_desc_rf.values.copy()
    if feat_desc_rf.units is not None:
        feat_desc["unit"] = feat_desc_rf.units

    dist_rf = ResultFrame.from_wide(distances)
    feat_dist_rf = dist_rf.aggregate_distances(grouper)
    feat_dist = feat_dist_rf.values.copy()
    # unit comes from descriptions (distances have no observed description values)
    feat_dist["unit"] = descriptions["unit"].groupby(level=grouper).first()

    return feat_desc, feat_dist


def describe(descriptions: DataFrame, distances: DataFrame) -> dict[str, DataFrame]:
    from acteval._aggregation import (
        descriptions_to_domain_level,
        descriptions_to_group_level,
        distances_to_domain_level,
        distances_to_group_level,
    )

    feat_desc, feat_dist = _aggregate_features(descriptions, distances)
    group_desc = descriptions_to_group_level(descriptions)
    group_dist = distances_to_group_level(distances)
    domain_desc = descriptions_to_domain_level(group_desc)
    domain_dist = distances_to_domain_level(group_dist)
    return {
        "descriptions": feat_desc,
        "distances": feat_dist,
        "group_descriptions": group_desc,
        "group_distances": group_dist,
        "domain_descriptions": domain_desc,
        "domain_distances": domain_dist,
    }


_PARALLEL_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Low-level helpers: density features
# ---------------------------------------------------------------------------


def _observed_base(spec: JobSpec, observed_features: dict) -> tuple[DataFrame, tuple]:
    """Build the observed-only base rows for a feature spec.

    Returns (base_df, default) where base_df has columns {observed__weight,
    observed} with a flat segment index sorted by weight descending.
    The MultiIndex is NOT set here; the caller sets it after stacking specs.
    """
    default = _make_default(observed_features)
    observed_weight = spec.size_fn(observed_features)
    observed_weight.name = "observed__weight"
    description_observed = spec.describe_fn(observed_features)
    base = DataFrame(
        {"observed__weight": observed_weight, "observed": description_observed}
    )
    base = base.sort_values(ascending=False, by=["observed__weight", "observed"])
    return base, default


def _model_contribution(
    model: str,
    spec: JobSpec,
    obs_features: dict,
    synth_features: dict,
    default: tuple,
) -> tuple[Series, Series, Series]:
    """Compute weight, description, and distance columns for one model × one spec."""
    synth_weight = spec.size_fn(synth_features)
    synth_weight.name = f"{model}__weight"
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
) -> tuple[DataFrame, DataFrame]:
    """Build observed base rows for enabled creativity metrics.

    Args:
        target_schedules: Observed schedule DataFrame.
        observed_hash: Pre-computed population hash set for this split/cat.
        config: Controls which creativity rows to produce.
    """
    obs_diversity = creativity.diversity(target_schedules, observed_hash)
    n = target_schedules.pid.nunique()
    names = ["domain", "feature", "segment"]
    desc_idx, desc_weight, desc_val, desc_unit = [], [], [], []
    dist_idx, dist_weight, dist_val, dist_unit = [], [], [], []

    if config.diversity:
        desc_idx.append(("creativity", "diversity", "all"))
        desc_weight.append(n); desc_val.append(obs_diversity); desc_unit.append("prob. unique")
        dist_idx.append(("creativity", "homogeneity", "all"))
        dist_weight.append(n); dist_val.append(1 - obs_diversity); dist_unit.append("prob. not unique")
    if config.novelty:
        desc_idx.append(("creativity", "novelty", "all"))
        desc_weight.append(n); desc_val.append(1); desc_unit.append("prob. novel")
        dist_idx.append(("creativity", "conservatism", "all"))
        dist_weight.append(n); dist_val.append(0); dist_unit.append("prob. conservative")

    base_desc = DataFrame(
        {"observed__weight": desc_weight, "observed": desc_val, "unit": desc_unit},
        index=MultiIndex.from_tuples(desc_idx, names=names),
    )
    base_dist = DataFrame(
        {"observed__weight": dist_weight, "observed": dist_val, "unit": dist_unit},
        index=MultiIndex.from_tuples(dist_idx, names=names),
    )
    return base_desc, base_dist


def _model_cols_creativity(
    model: str,
    pid_hashes: dict,
    sample_pids,
    observed_hash: set,
    config: CreativityConfig,
) -> tuple[DataFrame, DataFrame]:
    """Compute creativity columns for one model using pre-computed per-pid hashes.

    Args:
        model: Model name.
        pid_hashes: ``{pid: hash_str}`` for the full synthetic population.
        sample_pids: Pid values for this (split, cat) subset.
        observed_hash: Pre-cached hash set for this (split, cat) target subset.
        config: Controls which creativity rows to produce.
    """
    y_hash = {pid_hashes[p] for p in sample_pids if p in pid_hashes}
    y_count = len(sample_pids)
    y_diversity = len(y_hash) / y_count if y_count > 0 else 0
    names = ["domain", "feature", "segment"]
    desc_idx, desc_weight, desc_val = [], [], []
    dist_idx, dist_weight, dist_val = [], [], []

    if config.diversity:
        desc_idx.append(("creativity", "diversity", "all"))
        desc_weight.append(y_count); desc_val.append(y_diversity)
        dist_idx.append(("creativity", "homogeneity", "all"))
        dist_weight.append(y_count); dist_val.append(1 - y_diversity)
    if config.novelty:
        y_novelty = creativity.novelty(observed_hash, y_hash)
        desc_idx.append(("creativity", "novelty", "all"))
        desc_weight.append(y_count); desc_val.append(y_novelty)
        dist_idx.append(("creativity", "conservatism", "all"))
        dist_weight.append(y_count); dist_val.append(1 - y_novelty)

    desc = DataFrame(
        {f"{model}__weight": desc_weight, model: desc_val},
        index=MultiIndex.from_tuples(desc_idx, names=names),
    )
    dist = DataFrame(
        {f"{model}__weight": dist_weight, model: dist_val},
        index=MultiIndex.from_tuples(dist_idx, names=names),
    )
    return desc, dist


# ---------------------------------------------------------------------------
# Low-level helpers: structural / feasibility
# ---------------------------------------------------------------------------


def _observed_base_structural(
    target_schedules: DataFrame, config: StructuralConfig
) -> DataFrame:
    """Build observed base rows for enabled structural (feasibility) metrics.

    Novel-scoped rows use ``observed = 0`` (by definition the observed population
    has no novel schedules).
    """
    parts = []
    if config.home_based or config.consecutive:
        w, m = structural.feasibility_eval(
            Population(target_schedules),
            name="observed",
            home_based=config.home_based,
            consecutive=config.consecutive,
        )
        base = concat([w, m], axis=1)
        base["unit"] = "prob. infeasible"
        parts.append(base)
    if config.home_based_novel or config.consecutive_novel:
        idx = structural.feasibility_index(
            home_based=config.home_based_novel,
            consecutive=config.consecutive_novel,
            suffix=" (novel)",
        )
        n = target_schedules.pid.nunique()
        novel_base = DataFrame(
            {
                "observed__weight": [n] * len(idx),
                "observed": [0.0] * len(idx),
                "unit": "prob. infeasible",
            },
            index=idx,
        )
        parts.append(novel_base)
    return concat(parts)


def _model_cols_structural(
    model: str,
    per_pid_flags: dict,
    synth_dense_pids,
    novel_dense_pids,
    config: StructuralConfig,
) -> DataFrame:
    """Compute structural columns for one model using pre-computed per-pid flags.

    Args:
        model: Model name.
        per_pid_flags: Output of ``structural.feasibility`` for the full
            synthetic population, pre-computed once before the split loop.
        synth_dense_pids: Dense pid indices for all persons in this split/cat.
        novel_dense_pids: Dense pid indices for novel persons only (may be None
            if ``config.needs_novel_pids`` is False).
        config: Controls which structural rows to produce.
    """
    parts = []
    if config.home_based or config.consecutive:
        w, m = structural.feasibility_aggregate(
            per_pid_flags,
            synth_dense_pids,
            model,
            home_based=config.home_based,
            consecutive=config.consecutive,
        )
        parts.append(concat([w, m], axis=1))
    if config.home_based_novel or config.consecutive_novel:
        w, m = structural.feasibility_aggregate(
            per_pid_flags,
            novel_dense_pids,
            model,
            home_based=config.home_based_novel,
            consecutive=config.consecutive_novel,
            suffix=" (novel)",
        )
        parts.append(concat([w, m], axis=1))
    return concat(parts) if parts else DataFrame()


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


def evaluate(
    synthetic_schedules: dict[str, DataFrame],
    target_schedules: DataFrame,
    report_stats: bool = True,
    verbose: bool = False,
):
    """Deprecated: use ``compare`` instead."""
    warnings.warn(
        "evaluate is deprecated, use compare instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from acteval.evaluate import compare

    return compare(target_schedules, synthetic_schedules, report_stats=report_stats)


def subsample_and_evaluate(
    synthetic_schedules: dict[str, DataFrame],
    synthetic_attributes: dict[str, DataFrame],
    target_schedules: DataFrame,
    target_attributes: DataFrame,
    split_on: list[str],
    report_stats: bool = True,
    verbose: bool = False,
):
    """Deprecated: use ``compare_splits`` instead for better performance."""
    warnings.warn(
        "subsample_and_evaluate is deprecated, use compare_splits instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from acteval.evaluate import compare_splits

    return compare_splits(
        observed=target_schedules,
        synthetic_schedules=synthetic_schedules,
        synthetic_attributes=synthetic_attributes,
        target_attributes=target_attributes,
        split_on=split_on,
        report_stats=report_stats,
        verbose=verbose,
    )
