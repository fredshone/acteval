import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from pandas import DataFrame, MultiIndex, Series, concat

from acteval.creativity.features import creativity
from acteval.filters import filter_novel
from acteval.jobs import JobSpec, get_jobs
from acteval.population import Population
from acteval.structural.features import structural

# ---------------------------------------------------------------------------
# Aggregation helpers (formerly _aggregate.py)
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
    to_drop = [f for f in features if f in sorted_df.index]
    if not to_drop:
        return df
    return sorted_df.drop(to_drop, axis=0)


def weighted_av(report: DataFrame, suffix: str = "__weight") -> Series:
    """Weighted average of dataframe using weights in the weight column."""
    cols = [c for c in report.columns if not c.endswith(suffix)]
    scores = DataFrame()
    for c in cols:
        weights = report[f"{c}{suffix}"]
        total = weights.sum()
        scores[c] = report[c] * weights / total
    return scores.sum()


def distance_weighted_av(
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


def add_stats(data: DataFrame, columns: dict[str, DataFrame]):
    data["mean"] = data[columns].mean(axis=1)
    data["std"] = data[columns].std(axis=1)


def _aggregate_features(
    descriptions: DataFrame, distances: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Tier 1: collapse per-segment rows into one row per (domain, feature, segment)."""
    grouper = ["domain", "feature", "segment"]
    feat_desc = descriptions.drop("unit", axis=1).groupby(grouper).apply(weighted_av)
    feat_desc["unit"] = descriptions["unit"].groupby(grouper).first()
    feat_dist = distances.drop("unit", axis=1).groupby(grouper).apply(distance_weighted_av)
    feat_dist["unit"] = descriptions["unit"].groupby(grouper).first()
    return feat_desc, feat_dist


def _aggregate_groups(
    descriptions: DataFrame, distances: DataFrame, grouper: list[str]
) -> tuple[DataFrame, DataFrame]:
    """Tier 2: aggregate features into groups, dropping noisy sub-features."""
    desc = _drop_features(descriptions.drop("unit", axis=1), _REMOVE_FEATURES)
    group_desc = desc.groupby(grouper).apply(weighted_av)
    group_desc["unit"] = descriptions["unit"].groupby(grouper).first()
    dist = _drop_features(distances.drop("unit", axis=1), _REMOVE_FEATURES)
    group_dist = dist.groupby(grouper).apply(distance_weighted_av)
    group_dist["unit"] = descriptions["unit"].groupby(grouper).first()
    return group_desc, group_dist


def _aggregate_domains(
    group_desc: DataFrame, group_dist: DataFrame, grouper: list[str]
) -> tuple[DataFrame, DataFrame]:
    """Tier 3: aggregate groups into domains, dropping non-representative groups."""
    domain_desc = _drop_features(group_desc.drop("unit", axis=1), _REMOVE_GROUPS)
    domain_dist = _drop_features(group_dist.drop("unit", axis=1), _REMOVE_GROUPS)
    return domain_desc.groupby(grouper).mean(), domain_dist.groupby(grouper).mean()


def describe(descriptions: DataFrame, distances: DataFrame) -> dict[str, DataFrame]:
    feat_desc, feat_dist = _aggregate_features(descriptions, distances)
    group_desc, group_dist = _aggregate_groups(descriptions, distances, ["domain", "feature"])
    domain_desc, domain_dist = _aggregate_domains(group_desc, group_dist, ["domain"])
    return {
        "descriptions": feat_desc,
        "distances": feat_dist,
        "group_descriptions": group_desc,
        "group_distances": group_dist,
        "domain_descriptions": domain_desc,
        "domain_distances": domain_dist,
    }


def describe_labels(
    descriptions: DataFrame, distances: DataFrame
) -> dict[str, DataFrame]:
    group_desc, group_dist = _aggregate_groups(
        descriptions, distances, ["domain", "feature", "label"]
    )
    domain_desc, domain_dist = _aggregate_domains(group_desc, group_dist, ["domain", "label"])
    return {
        "label_descriptions": descriptions,
        "label_distances": distances,
        "label_group_descriptions": group_desc,
        "label_group_distances": group_dist,
        "label_domain_descriptions": domain_desc,
        "label_domain_distances": domain_dist,
    }

_PARALLEL_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Low-level helpers: density features
# ---------------------------------------------------------------------------


def _observed_base(
    spec: JobSpec, observed_features: dict
) -> tuple[DataFrame, tuple]:
    """Build the observed-only base rows for a feature spec.

    Returns (base_df, default) where base_df has columns {observed__weight,
    observed} with a flat segment index sorted by weight descending.
    The MultiIndex is NOT set here; the caller sets it after stacking specs.
    """
    default = extract_default(observed_features)
    observed_weight = spec.size_fn(observed_features)
    observed_weight.name = "observed__weight"
    description_observed = spec.describe_fn(observed_features)
    base = DataFrame({"observed__weight": observed_weight, "observed": description_observed})
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
    desc = describe_feature(model, synth_features, spec.describe_fn)
    dist = score_features(
        model, obs_features, synth_features, spec.distance_fn, default, spec.missing_distance
    )
    return synth_weight, desc, dist


# ---------------------------------------------------------------------------
# Low-level helpers: creativity
# ---------------------------------------------------------------------------


def _observed_base_creativity(
    target_schedules: DataFrame,
) -> tuple[DataFrame, DataFrame, object]:
    """Build observed base rows for creativity metrics.

    Returns (base_desc, base_dist, observed_hash).
    """
    observed_hash = creativity.hash_population(target_schedules)
    observed_diversity = creativity.diversity(target_schedules, observed_hash)
    feature_count = target_schedules.pid.nunique()
    mi_desc = MultiIndex.from_tuples(
        [("creativity", "diversity", "all"), ("creativity", "novelty", "all")],
        names=["domain", "feature", "segment"],
    )
    mi_dist = MultiIndex.from_tuples(
        [("creativity", "homogeneity", "all"), ("creativity", "conservatism", "all")],
        names=["domain", "feature", "segment"],
    )
    base_desc = DataFrame(
        {
            "observed__weight": [feature_count, feature_count],
            "observed": [observed_diversity, 1],
            "unit": ["prob. unique", "prob. novel"],
        },
        index=mi_desc,
    )
    base_dist = DataFrame(
        {
            "observed__weight": [feature_count, feature_count],
            "observed": [1 - observed_diversity, 0],
            "unit": ["prob. not unique", "prob. conservative"],
        },
        index=mi_dist,
    )
    return base_desc, base_dist, observed_hash


def _model_cols_creativity(
    model: str, y: DataFrame, observed_hash: object
) -> tuple[DataFrame, DataFrame]:
    """Compute creativity columns for one model."""
    y_hash = creativity.hash_population(y)
    y_diversity = creativity.diversity(y, y_hash)
    y_count = y.pid.nunique()
    mi_desc = MultiIndex.from_tuples(
        [("creativity", "diversity", "all"), ("creativity", "novelty", "all")],
        names=["domain", "feature", "segment"],
    )
    mi_dist = MultiIndex.from_tuples(
        [("creativity", "homogeneity", "all"), ("creativity", "conservatism", "all")],
        names=["domain", "feature", "segment"],
    )
    desc = DataFrame(
        {
            f"{model}__weight": [y_count, y_count],
            model: [y_diversity, creativity.novelty(observed_hash, y_hash)],
        },
        index=mi_desc,
    )
    dist = DataFrame(
        {
            f"{model}__weight": [y_count, y_count],
            model: [1 - y_diversity, creativity.conservatism(observed_hash, y_hash)],
        },
        index=mi_dist,
    )
    return desc, dist


# ---------------------------------------------------------------------------
# Low-level helpers: structural / feasibility
# ---------------------------------------------------------------------------


def _observed_base_structural(target_schedules: DataFrame) -> DataFrame:
    """Build observed base rows for structural (feasibility) metrics."""
    observed_weights, observed_metrics = structural.feasibility_eval(
        Population(target_schedules), name="observed"
    )
    base = concat([observed_weights, observed_metrics], axis=1)
    base["unit"] = "prob. infeasible"
    return base


def _model_cols_structural(
    model: str, y: DataFrame, target_schedules: DataFrame
) -> DataFrame:
    """Compute structural columns for one model."""
    y_df = filter_novel(y, target_schedules)
    weights, metrics = structural.feasibility_eval(Population(y_df), name=model)
    return concat([weights, metrics], axis=1)


# ---------------------------------------------------------------------------
# process_metrics: populations → features
# ---------------------------------------------------------------------------


def process_metrics(
    synthetic_schedules: dict[str, DataFrame],
    target_schedules: DataFrame,
    verbose: bool = False,
    config_path=None,
    target_pop: Population | None = None,
    synthetic_pops: dict[str, Population] | None = None,
    cached_features: dict[tuple, dict] | None = None,
    cached_synthetic_features: dict[str, dict[tuple, dict]] | None = None,
) -> tuple[DataFrame, DataFrame]:
    density_jobs, run_creativity, run_structural = get_jobs(config_path)

    if target_pop is None:
        target_pop = Population(target_schedules)
    if synthetic_pops is None:
        synthetic_pops = {m: Population(y) for m, y in synthetic_schedules.items()}

    # --- observed features for all density jobs (computed once) ---
    obs_features: dict[tuple, dict] = {}
    defaults: dict[tuple, tuple] = {}
    for spec in density_jobs:
        key = (spec.domain, spec.name)
        obs_feat = (cached_features or {}).get(key) or spec.feature_fn(target_pop).aggregate()
        obs_features[key] = obs_feat
        defaults[key] = extract_default(obs_feat)

    # --- observed base rows ---
    base_desc_parts: list[DataFrame] = []
    base_dist_parts: list[DataFrame] = []

    observed_hash = None
    if run_creativity:
        bd, bi, observed_hash = _observed_base_creativity(target_schedules)
        base_desc_parts.append(bd)
        base_dist_parts.append(bi.drop("observed", axis=1))

    if run_structural:
        base_struct = _observed_base_structural(target_schedules)
        base_desc_parts.append(base_struct)
        base_dist_parts.append(base_struct.drop("observed", axis=1))

    for spec in density_jobs:
        key = (spec.domain, spec.name)
        base, _ = _observed_base(spec, obs_features[key])
        mi = MultiIndex.from_tuples(
            [(spec.domain, spec.name, f) for f in base.index],
            names=["domain", "feature", "segment"],
        )
        base.index = mi
        base_desc_parts.append(base.assign(unit=spec.description_name))
        base_dist_parts.append(base[["observed__weight"]].assign(unit=spec.distance_name))

    base_desc = concat(base_desc_parts)
    base_dist = concat(base_dist_parts)

    # --- per-population columns (populations → features) ---
    model_desc_cols: list[DataFrame] = []
    model_dist_cols: list[DataFrame] = []
    timings: dict[str, float] = {}

    for model, pop in synthetic_pops.items():
        if verbose:
            print(f">>> Evaluating {model}")
        t0 = time.perf_counter()

        m_desc_parts: list[DataFrame] = []
        m_dist_parts: list[DataFrame] = []

        if run_creativity:
            c_desc, c_dist = _model_cols_creativity(model, synthetic_schedules[model], observed_hash)
            m_desc_parts.append(c_desc)
            m_dist_parts.append(c_dist)

        if run_structural:
            s_cols = _model_cols_structural(model, synthetic_schedules[model], target_schedules)
            m_desc_parts.append(s_cols)
            m_dist_parts.append(s_cols)

        for spec in density_jobs:
            key = (spec.domain, spec.name)
            synth_feat = None
            if cached_synthetic_features:
                model_cache = cached_synthetic_features.get(model)
                if model_cache is not None and key in model_cache:
                    synth_feat = model_cache[key]
            if synth_feat is None:
                synth_feat = spec.feature_fn(pop).aggregate()
            w, d, s = _model_contribution(model, spec, obs_features[key], synth_feat, defaults[key])
            desc_part = DataFrame({f"{model}__weight": w, model: d})
            dist_part = DataFrame({f"{model}__weight": w, model: s})
            desc_part.index = MultiIndex.from_tuples(
                [(spec.domain, spec.name, f) for f in desc_part.index],
                names=["domain", "feature", "segment"],
            )
            dist_part.index = MultiIndex.from_tuples(
                [(spec.domain, spec.name, f) for f in dist_part.index],
                names=["domain", "feature", "segment"],
            )
            m_desc_parts.append(desc_part)
            m_dist_parts.append(dist_part)

        model_desc_cols.append(concat(m_desc_parts))
        model_dist_cols.append(concat(m_dist_parts))
        timings[model] = time.perf_counter() - t0

    if verbose:
        print("\n--- Model timings ---")
        for model_name, elapsed in sorted(timings.items(), key=lambda x: -x[1]):
            print(f"  {model_name:40s} {elapsed:.3f}s")
        print(f"  {'TOTAL':40s} {sum(timings.values()):.3f}s")

    descriptions = concat([base_desc] + model_desc_cols, axis=1)
    distances = concat([base_dist] + model_dist_cols, axis=1)

    return descriptions, distances


def describe_feature(
    model: str,
    features: dict[str, tuple[np.array, np.array]],
    describe: Callable,
):
    feature_description = describe(features)
    feature_description.name = model
    return feature_description


def score_features(
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
        return distance(defaulting_get(a, k, default), defaulting_get(b, k, default))

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


def defaulting_get(
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


def extract_default(features: dict[str, tuple[np.array, np.array]]):
    default_shape = extract_default_shape(features)
    default_support = np.zeros(default_shape)
    return (default_support, np.array([1]))


def extract_default_shape(features: dict[str, tuple[np.array, np.array]]) -> np.array:
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
    descriptions, distances = process_metrics(
        synthetic_schedules, target_schedules, verbose=verbose
    )
    frames = describe(descriptions, distances)

    if report_stats:
        columns = list(synthetic_schedules.keys())
        for frame in frames.values():
            add_stats(data=frame, columns=columns)

    return frames


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
