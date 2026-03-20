import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from pandas import DataFrame, MultiIndex, Series, concat

from acteval.creativity.features import creativity
from acteval.density.features.pid_features import PidFeatures
from acteval.filters import filter_novel
from acteval.jobs import get_jobs
from acteval.population import Population
from acteval.structural.features import structural

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


def _summarize(
    descriptions: DataFrame,
    distances: DataFrame,
    prefix: str = "",
    feature_grouper: list[str] | None = None,
    group_grouper: list[str] | None = None,
    domain_grouper: list[str] | None = None,
) -> dict[str, DataFrame]:
    """Unified 3-tier aggregation for describe() and describe_labels()."""
    frames = {}

    if feature_grouper is not None:
        # Feature tier
        feat_desc = (
            descriptions.drop("unit", axis=1)
            .groupby(feature_grouper)
            .apply(weighted_av)
        )
        feat_desc["unit"] = descriptions["unit"].groupby(feature_grouper).first()
        feat_dist = (
            distances.drop("unit", axis=1)
            .groupby(feature_grouper)
            .apply(distance_weighted_av)
        )
        feat_dist["unit"] = descriptions["unit"].groupby(feature_grouper).first()
        frames[f"{prefix}descriptions"] = feat_desc
        frames[f"{prefix}distances"] = feat_dist
    else:
        # No feature tier — raw data goes in as descriptions/distances
        frames[f"{prefix}descriptions"] = descriptions
        frames[f"{prefix}distances"] = distances

    # Group tier
    group_desc = descriptions.drop("unit", axis=1)
    group_desc = _drop_features(group_desc, _REMOVE_FEATURES)
    group_desc = group_desc.groupby(group_grouper).apply(weighted_av)
    group_desc["unit"] = descriptions["unit"].groupby(group_grouper).first()

    group_dist = distances.drop("unit", axis=1)
    group_dist = _drop_features(group_dist, _REMOVE_FEATURES)
    group_dist = group_dist.groupby(group_grouper).apply(distance_weighted_av)
    group_dist["unit"] = descriptions["unit"].groupby(group_grouper).first()

    frames[f"{prefix}group_descriptions"] = group_desc
    frames[f"{prefix}group_distances"] = group_dist

    # Domain tier
    domain_desc = group_desc.drop("unit", axis=1)
    domain_desc = _drop_features(domain_desc, _REMOVE_GROUPS)
    domain_desc = domain_desc.groupby(domain_grouper).mean()

    domain_dist = group_dist.drop("unit", axis=1)
    domain_dist = _drop_features(domain_dist, _REMOVE_GROUPS)
    domain_dist = domain_dist.groupby(domain_grouper).mean()

    frames[f"{prefix}domain_descriptions"] = domain_desc
    frames[f"{prefix}domain_distances"] = domain_dist

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
    return compare_splits(
        observed=target_schedules,
        synthetic_schedules=synthetic_schedules,
        synthetic_attributes=synthetic_attributes,
        target_attributes=target_attributes,
        split_on=split_on,
        report_stats=report_stats,
        verbose=verbose,
    )


def evaluate(
    synthetic_schedules: dict[str, DataFrame],
    target_schedules: DataFrame,
    report_stats: bool = True,
    verbose: bool = False,
):
    descriptions, distances = process_metrics(
        synthetic_schedules, target_schedules, verbose=verbose
    )
    frames = describe(descriptions, distances)

    if report_stats:
        columns = list(synthetic_schedules.keys())
        for frame in frames.values():
            add_stats(data=frame, columns=columns)

    return frames


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

    pairs: list[tuple[DataFrame, DataFrame]] = []
    timings = {}

    if run_creativity:
        if verbose:
            print(">>> Evaluating creativity")
        t0 = time.perf_counter()
        pairs.append(
            eval_creativity(
                synthetic_schedules=synthetic_schedules,
                target_schedules=target_schedules,
            )
        )
        timings["creativity"] = time.perf_counter() - t0

    if run_structural:
        if verbose:
            print(">>> Evaluating sample quality")
        t0 = time.perf_counter()
        sample_quality = eval_sample_quality(
            synthetic_schedules=synthetic_schedules,
            target_schedules=target_schedules,
        )
        timings["sample_quality"] = time.perf_counter() - t0
        pairs.append((sample_quality, sample_quality))

    if target_pop is None:
        target_pop = Population(target_schedules)
    if synthetic_pops is None:
        synthetic_pops = {m: Population(y) for m, y in synthetic_schedules.items()}

    for domain, jobs in density_jobs:
        for feature, size, description_job, distance_job in jobs:
            feature_name, feature_fn = feature[0], feature[1]
            if verbose:
                print(f">>> Evaluating {domain} {feature_name}")
            t0 = time.perf_counter()
            # use cached target features if available
            observed_features = None
            if cached_features is not None:
                observed_features = cached_features.get((domain, feature_name))
            if observed_features is None:
                observed_features = feature_fn(target_pop)
            synth_features_for_job = None
            if cached_synthetic_features is not None:
                key = (domain, feature_name)
                synth_features_for_job = {
                    model: feat_dict[key]
                    for model, feat_dict in cached_synthetic_features.items()
                    if key in feat_dict
                }
                if not synth_features_for_job:
                    synth_features_for_job = None
            pairs.append(
                eval_jobs(
                    synthetic_schedules=synthetic_pops,
                    target_schedules=target_pop,
                    domain=domain,
                    feature=(feature_name, feature_fn),
                    size=size,
                    description_job=description_job,
                    distance_job=distance_job,
                    observed_features=observed_features,
                    synthetic_features=synth_features_for_job,
                )
            )
            timings[f"{domain}/{feature_name}"] = time.perf_counter() - t0

    descriptions = concat([d.fillna(0.0) for d, _ in pairs if not d.empty], axis=0)
    distances = concat([d.fillna(0.0) for _, d in pairs if not d.empty], axis=0)

    if verbose:
        print("\n--- Job timings ---")
        for job_name, elapsed in sorted(timings.items(), key=lambda x: -x[1]):
            print(f"  {job_name:40s} {elapsed:.3f}s")
        print(f"  {'TOTAL':40s} {sum(timings.values()):.3f}s")

    return descriptions, distances


def describe(descriptions: DataFrame, distances: DataFrame) -> dict[str, DataFrame]:
    return _summarize(
        descriptions,
        distances,
        prefix="",
        feature_grouper=["domain", "feature", "segment"],
        group_grouper=["domain", "feature"],
        domain_grouper=["domain"],
    )


def describe_labels(
    descriptions: DataFrame, distances: DataFrame
) -> dict[str, DataFrame]:
    return _summarize(
        descriptions,
        distances,
        prefix="label_",
        feature_grouper=None,
        group_grouper=["domain", "feature", "label"],
        domain_grouper=["domain", "label"],
    )


def eval_creativity(
    synthetic_schedules: dict[str, DataFrame], target_schedules: DataFrame
) -> tuple[DataFrame, DataFrame]:
    observed_hash = creativity.hash_population(target_schedules)
    observed_diversity = creativity.diversity(target_schedules, observed_hash)
    feature_count = target_schedules.pid.nunique()

    desc_cols = {
        "observed__weight": [feature_count] * 2,
        "observed": [observed_diversity, 1],
    }
    dist_cols = {
        "observed__weight": [feature_count] * 2,
        "observed": [1 - observed_diversity, 0],
    }

    for model, y in synthetic_schedules.items():
        y_hash = creativity.hash_population(y)
        y_diversity = creativity.diversity(y, y_hash)
        y_count = y.pid.nunique()
        desc_cols[f"{model}__weight"] = [y_count, y_count]
        desc_cols[model] = [y_diversity, creativity.novelty(observed_hash, y_hash)]
        dist_cols[f"{model}__weight"] = [y_count, y_count]
        dist_cols[model] = [
            1 - y_diversity,
            creativity.conservatism(observed_hash, y_hash),
        ]

    desc_cols["unit"] = ["prob. unique", "prob. novel"]
    dist_cols["unit"] = ["prob. not unique", "prob. conservative"]

    descriptions = DataFrame(desc_cols)
    dist = DataFrame(dist_cols)

    descriptions.index = MultiIndex.from_tuples(
        [("creativity", "diversity", "all"), ("creativity", "novelty", "all")],
        names=["domain", "feature", "segment"],
    )
    dist.index = MultiIndex.from_tuples(
        [
            ("creativity", "homogeneity", "all"),
            ("creativity", "conservatism", "all"),
        ],
        names=["domain", "feature", "segment"],
    )
    return descriptions, dist


def eval_sample_quality(
    synthetic_schedules: dict[str, DataFrame], target_schedules: DataFrame
) -> tuple[DataFrame, DataFrame]:
    observed_weights, observed_metrics = structural.feasibility_eval(
        Population(target_schedules), name="observed"
    )
    results = [observed_weights, observed_metrics]
    for model, y in synthetic_schedules.items():
        y_df = filter_novel(y, target_schedules)
        weights, metrics = structural.feasibility_eval(Population(y_df), name=model)
        results.append(weights)
        results.append(metrics)
    results = concat(results, axis=1)
    results["unit"] = "prob. infeasible"
    return results


def eval_jobs(
    synthetic_schedules: dict[str, DataFrame],
    target_schedules: DataFrame,
    domain: str,
    feature: tuple[str, Callable],
    size: Callable,
    description_job: tuple[str, Callable],
    distance_job: tuple[str, Callable],
    observed_features=None,
    synthetic_features: dict[str, dict] | None = None,
) -> tuple[DataFrame, DataFrame]:
    # unpack tuples
    feature_name, feature_fn = feature
    description_name, describe = description_job
    distance_name, distance_metric = distance_job

    # build observed features (use cached if provided)
    if observed_features is None:
        observed_features = feature_fn(target_schedules)

    # need to create a default feature for missing sampled features
    default = extract_default(observed_features)

    # create an observed feature count and description
    observed_weight = size(observed_features)
    observed_weight.name = "observed__weight"
    description_observed = describe(observed_features)
    base = DataFrame(
        {"observed__weight": observed_weight, "observed": description_observed}
    )

    # sort by count and description
    base = base.sort_values(ascending=False, by=["observed__weight", "observed"])

    # collect per-model results as tuples, concat once after the loop (avoids O(M²) concat)
    model_results = []
    for model, y in synthetic_schedules.items():
        synth_features = (
            synthetic_features.get(model) if synthetic_features is not None else None
        )
        if synth_features is None:
            synth_features = feature_fn(y)
        synth_weight = size(synth_features)
        synth_weight.name = f"{model}__weight"
        model_results.append((
            synth_weight,
            describe_feature(model, synth_features, describe),
            score_features(model, observed_features, synth_features, distance_metric, default),
        ))

    desc_parts = [base] + [x for w, d, _ in model_results for x in (w, d)]
    dist_parts = [base] + [x for w, _, s in model_results for x in (w, s)]
    feature_descriptions = concat(desc_parts, axis=1)
    feature_distances = concat(dist_parts, axis=1)

    # add domain and feature name to index
    feature_descriptions["unit"] = description_name
    feature_distances["unit"] = distance_name
    feature_descriptions.index = MultiIndex.from_tuples(
        [(domain, feature_name, f) for f in feature_descriptions.index],
        name=["domain", "feature", "segment"],
    )
    feature_distances.index = MultiIndex.from_tuples(
        [(domain, feature_name, f) for f in feature_distances.index],
        name=["domain", "feature", "segment"],
    )

    return feature_descriptions, feature_distances


def rank(data: DataFrame) -> DataFrame:
    # feature rank
    rank = data.drop(["observed", "unit"], axis=1, errors="ignore").rank(
        axis=1, method="min"
    )
    col_ranks = rank.sum(axis=0)
    ranked = [i for _, i in sorted(zip(col_ranks, col_ranks.index))]
    return rank[ranked]


def _report_impl(
    frames: dict[str, DataFrame],
    prefix: str,
    head_grouper: list[str],
    log_dir: Path | None = None,
    head: int | None = None,
    verbose: bool = True,
    suffix: str = "",
    ranking: bool = False,
):
    if head is not None:
        frames[f"{prefix}descriptions_short"] = (
            frames[f"{prefix}descriptions"].groupby(head_grouper).head(head)
        )
        frames[f"{prefix}distances_short"] = (
            frames[f"{prefix}distances"].groupby(head_grouper).head(head)
        )
    else:
        frames[f"{prefix}descriptions_short"] = frames[f"{prefix}descriptions"]
        frames[f"{prefix}distances_short"] = frames[f"{prefix}distances"]

    if log_dir is not None:
        for name, frame in frames.items():
            frame.to_csv(Path(log_dir, f"{name}{suffix}.csv"))

    if verbose:
        print("\nDescriptions:")
        print_markdown(frames[f"{prefix}descriptions_short"])
        print("\nEvalutions (Distance):")
        print_markdown(frames[f"{prefix}distances_short"])

    print("\nGroup Descriptions:")
    print_markdown(frames[f"{prefix}group_descriptions"])
    print("\nGroup Evaluations (Distance):")
    print_markdown(frames[f"{prefix}group_distances"])
    if ranking:
        print("\nGroup Evaluations (Ranked):")
        print_markdown(rank(frames[f"{prefix}group_distances"]))

    print("\nDomain Descriptions:")
    print_markdown(frames[f"{prefix}domain_descriptions"])
    print("\nDomain Evaluations (Distance):")
    print_markdown(frames[f"{prefix}domain_distances"])
    if ranking:
        print("\nDomain Evaluations (Ranked):")
        print_markdown(rank(frames[f"{prefix}domain_distances"]))


def report(
    frames: dict[str, DataFrame],
    log_dir: Path | None = None,
    head: int | None = None,
    verbose: bool = True,
    suffix: str = "",
    ranking: bool = False,
):
    _report_impl(
        frames, "", ["domain", "feature"], log_dir, head, verbose, suffix, ranking
    )


def report_splits(
    frames: dict[str, DataFrame],
    log_dir: Path | None = None,
    head: int | None = None,
    verbose: bool = True,
    suffix: str = "",
    ranking: bool = False,
):
    _report_impl(
        frames,
        "label_",
        ["domain", "feature", "label"],
        log_dir,
        head,
        verbose,
        suffix,
        ranking,
    )


def add_stats(data: DataFrame, columns: dict[str, DataFrame]):
    data["mean"] = data[columns].mean(axis=1)
    data["std"] = data[columns].std(axis=1)


def print_markdown(data: DataFrame):
    print(data.to_markdown(tablefmt="fancy_grid", floatfmt=".3f"))


def describe_feature(
    model: str,
    features: dict[str, tuple[np.array, np.array]],
    describe: Callable,
):
    feature_description = describe(features)
    feature_description.name = model
    return feature_description


_PARALLEL_THRESHOLD = 50


def score_features(
    model: str,
    a: dict[str, tuple[np.array, np.array]],
    b: dict[str, tuple[np.array, np.array]],
    distance: Callable,
    default: tuple[np.array, np.array],
):
    index = list(set(a.keys()) | set(b.keys()))

    if len(index) > _PARALLEL_THRESHOLD:
        # POT's C extensions release the GIL — threads give real parallelism
        def _compute(k):
            return distance(
                defaulting_get(a, k, default), defaulting_get(b, k, default)
            )

        with ThreadPoolExecutor() as executor:
            values = list(executor.map(_compute, index))
        metrics = Series(dict(zip(index, values)), name=model)
    else:
        metrics = Series(
            {
                k: distance(
                    defaulting_get(a, k, default),
                    defaulting_get(b, k, default),
                )
                for k in index
            },
            name=model,
        )
    metrics = metrics.fillna(0)
    return metrics


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
    # we use a single feature of zeros as required
    # look for a size
    default_shape = extract_default_shape(features)
    default_support = np.zeros(default_shape)
    return (default_support, np.array([1]))


def extract_default_shape(features: dict[str, tuple[np.array, np.array]]) -> np.array:
    for k, _ in iter(features.values()):
        if len(k) > 0:
            default_shape = list(k.shape)
            default_shape[0] = 1
            return default_shape
    return np.array([1])


def weighted_av(report: DataFrame, suffix: str = "__weight") -> Series:
    """Weighted average of dataframe using weights in the weight column."""
    cols = list(report.columns)
    cols = [c for c in cols if not c.endswith(suffix)]
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
    """Weighted average of dataframe using weights in the weight column and a base column.
    This deals with cases where models have different features.
    """
    cols = list(report.columns)
    cols = [c for c in cols if not c.endswith(suffix)]
    base_weights = report[base_col]
    scores = DataFrame()
    for c in cols:
        weights = report[f"{c}{suffix}"]
        weights = (weights + base_weights) / 2
        total = weights.sum()
        scores[c] = report[c] * weights / total
    return scores.sum()


def _all_feature_jobs(config_path=None):
    """Yield (domain, feature_tuple, size, desc_job, dist_job) for all active jobs."""
    density_jobs, _, _ = get_jobs(config_path)
    for domain, jobs in density_jobs:
        for feature, size, description_job, distance_job in jobs:
            yield domain, feature, size, description_job, distance_job


def _build_orig_to_dense(pops: dict[str, "Population"]) -> dict[str, dict]:
    """Build {model: {orig_pid: dense_pid}} maps from each Population."""
    return {
        model: {orig: dense for dense, orig in enumerate(pop.unique_pids_original)}
        for model, pop in pops.items()
    }


def _precompute_pid_features(
    pops: dict[str, "Population"], config_path=None
) -> dict[str, dict[tuple, "PidFeatures"]]:
    """Compute PidFeatures once per population. Returns {model: {(domain, feat_name): PidFeatures}}."""
    result: dict[str, dict[tuple, PidFeatures]] = {model: {} for model in pops}
    for domain, feature, _, _, _ in _all_feature_jobs(config_path):
        per_pid_fn = feature[2] if len(feature) > 2 else None
        if per_pid_fn is None:
            continue
        key = (domain, feature[0])
        for model, pop in pops.items():
            result[model][key] = per_pid_fn(pop)
    return result


_TRAILING_DIGITS = re.compile(r"\d+$")


def _key_activities(key: str) -> frozenset[str] | None:
    """Return activity names referenced by a feature key, or None for non-activity keys.

    Key formats:
    - n-gram transitions: "act1>act2>act3" (split by ">")
    - joint participation: "act1+act2" (split by "+")
    - timing / single-act: "actN" or "act" (strip trailing digits)
    - non-activity keys like "sequence lengths" contain spaces → return None
    """
    if ">" in key:
        return frozenset(key.split(">"))
    if "+" in key:
        return frozenset(key.split("+"))
    stripped = _TRAILING_DIGITS.sub("", key)
    if not stripped or " " in stripped:
        return None
    return frozenset([stripped])


def _subset_pid_features(
    pid_features: dict[str, dict[tuple, "PidFeatures"]],
    dense_pids: dict[str, np.ndarray],
    subset_acts: dict[str, frozenset[str]] | None = None,
) -> dict[str, dict[tuple, dict]]:
    """Subset each model's PidFeatures to dense_pids and aggregate.

    Empty entries and entries for activities absent from the subset are dropped
    so the result matches what ``feature_fn(Population(sub_df))`` would return.

    Args:
        subset_acts: Optional ``{model: frozenset_of_activity_names}`` used to
            filter out keys referencing activities not in the subset.
    """
    result: dict[str, dict[tuple, dict]] = {}
    for model, model_features in pid_features.items():
        acts = subset_acts[model] if subset_acts is not None else None
        model_result: dict[tuple, dict] = {}
        for key, pf in model_features.items():
            aggregated = pf.subset(dense_pids[model]).aggregate()
            filtered: dict = {}
            for k, v in aggregated.items():
                if len(v[0]) == 0:
                    continue
                if acts is not None:
                    key_acts = _key_activities(k)
                    if key_acts is not None and not key_acts.issubset(acts):
                        continue
                filtered[k] = v
            model_result[key] = filtered
        result[model] = model_result
    return result


class Evaluator:
    """Pre-computes target features once; compare multiple synthetic populations."""

    def __init__(self, target: DataFrame, config_path=None):
        self._target = target
        self._target_pop = Population(target)
        self._config_path = config_path
        self._target_features: dict[tuple, dict] = {}
        self._target_pid_features: dict[tuple, PidFeatures] = {}
        self._precompute()

    def _precompute(self) -> None:
        for domain, feature, size, desc_job, dist_job in _all_feature_jobs(
            self._config_path
        ):
            feature_name = feature[0]
            feature_fn = feature[1]
            per_pid_fn = feature[2] if len(feature) > 2 else None
            key = (domain, feature_name)
            self._target_features[key] = feature_fn(self._target_pop)
            if per_pid_fn is not None:
                self._target_pid_features[key] = per_pid_fn(self._target_pop)

    def compare(
        self,
        synthetic: dict[str, DataFrame],
        report_stats: bool = True,
    ) -> dict[str, DataFrame]:
        """Compare synthetic populations against pre-computed target features."""
        synthetic_pops = {m: Population(y) for m, y in synthetic.items()}

        descriptions, distances = process_metrics(
            synthetic_schedules=synthetic,
            target_schedules=self._target,
            config_path=self._config_path,
            target_pop=self._target_pop,
            synthetic_pops=synthetic_pops,
            cached_features=self._target_features,
        )

        frames = describe(descriptions, distances)

        if report_stats:
            columns = list(synthetic.keys())
            for frame in frames.values():
                add_stats(data=frame, columns=columns)

        return frames

    def compare_splits(
        self,
        synthetic_schedules: dict[str, DataFrame],
        synthetic_attributes: dict[str, DataFrame],
        target_attributes: DataFrame,
        split_on: list[str],
        report_stats: bool = True,
        verbose: bool = False,
    ) -> dict[str, DataFrame]:
        """Compare synthetic populations against target, split by attribute categories.

        Target density features are computed once (at init) and subset per split
        using per-pid features. Creativity and structural features are recomputed
        per split.

        Args:
            synthetic_schedules: ``{model_name: schedules_df}``.
            synthetic_attributes: ``{model_name: attributes_df}`` with ``pid`` column.
            target_attributes: Target attributes DataFrame with ``pid`` column.
            split_on: Attribute columns to split on.
            report_stats: Whether to append mean/std columns.
            verbose: Print progress.

        Returns:
            Dict of result DataFrames with per-split and per-label summaries.
        """
        # Build mapping from original target pids to dense Population pids
        orig_to_dense = {
            orig: dense
            for dense, orig in enumerate(self._target_pop.unique_pids_original)
        }

        # Precompute synthetic populations and their per-pid features once
        synthetic_pops = {m: Population(y) for m, y in synthetic_schedules.items()}
        synth_orig_to_dense = _build_orig_to_dense(synthetic_pops)
        synth_pid_features = _precompute_pid_features(synthetic_pops, self._config_path)

        density_jobs, run_creativity, run_structural = get_jobs(self._config_path)

        pairs: list[tuple[DataFrame, DataFrame]] = []

        for split in split_on:
            target_cats = target_attributes[split].unique()
            for cat in target_cats:
                # --- target pid subset ---
                target_orig_pids = target_attributes[
                    target_attributes[split] == cat
                ].pid.values
                target_dense_pids = np.array(
                    [orig_to_dense[p] for p in target_orig_pids if p in orig_to_dense],
                    dtype=np.int64,
                )

                sub_target = self._target[self._target.pid.isin(target_orig_pids)]

                # --- synthetic subsets ---
                sub_schedules: dict[str, DataFrame] = {}
                synth_dense_pids_for_cat: dict[str, np.ndarray] = {}
                for model, attributes in synthetic_attributes.items():
                    sample_pids = attributes[attributes[split] == cat].pid.values
                    if verbose:
                        print(
                            f">>> Subsampled {model} {split}={cat} with {len(sample_pids)}"
                        )
                    sample_schedules = synthetic_schedules[model]
                    sub_schedules[model] = sample_schedules[
                        sample_schedules.pid.isin(sample_pids)
                    ]
                    o2d = synth_orig_to_dense[model]
                    synth_dense_pids_for_cat[model] = np.array(
                        [o2d[p] for p in sample_pids if p in o2d], dtype=np.int64
                    )

                # --- build cached target features for this subset ---
                cached_subset: dict[tuple, dict] = {}
                for key, pid_feat in self._target_pid_features.items():
                    cached_subset[key] = pid_feat.subset(target_dense_pids).aggregate()

                # --- build cached synthetic features for this subset ---
                synth_sub_acts = {
                    model: frozenset(sub_schedules[model]["act"].unique())
                    for model in sub_schedules
                }
                cached_synth = _subset_pid_features(
                    synth_pid_features, synth_dense_pids_for_cat, synth_sub_acts
                )

                # --- run process_metrics with both cached target and synthetic ---
                desc, dist = process_metrics(
                    synthetic_schedules=sub_schedules,
                    target_schedules=sub_target,
                    verbose=verbose,
                    config_path=self._config_path,
                    cached_features=cached_subset,
                    cached_synthetic_features=cached_synth,
                )
                for r in (desc, dist):
                    r.index = MultiIndex.from_tuples(
                        [(*i, split, cat) for i in r.index],
                        names=list(r.index.names) + ["label", "cat"],
                    )
                pairs.append((desc, dist))

        descriptions = concat([d for d, _ in pairs], axis=0)
        distances = concat([d for _, d in pairs], axis=0)

        frames = describe(descriptions, distances)
        frames.update(describe_labels(descriptions, distances))

        if report_stats:
            columns = list(synthetic_schedules.keys())
            for frame in frames.values():
                add_stats(data=frame, columns=columns)

        return frames


def compare(
    observed: DataFrame,
    synthetic,
    report_stats: bool = True,
) -> dict[str, DataFrame]:
    """Compare observed and synthetic activity schedule populations.

    Args:
        observed: Observed schedules with columns pid, act, start, end, duration.
        synthetic: Single synthetic DataFrame or dict mapping model names to DataFrames.
        report_stats: Whether to append mean/std columns.

    Returns:
        Dict of result DataFrames (descriptions, distances, grouped variants).
    """
    if isinstance(synthetic, DataFrame):
        synthetic = {"synthetic": synthetic}
    return Evaluator(observed).compare(synthetic, report_stats=report_stats)


def compare_splits(
    observed: DataFrame,
    synthetic_schedules: dict[str, DataFrame],
    synthetic_attributes: dict[str, DataFrame],
    target_attributes: DataFrame,
    split_on: list[str],
    report_stats: bool = True,
    verbose: bool = False,
) -> dict[str, DataFrame]:
    """Compare observed and synthetic populations, split by attribute categories.

    Convenience wrapper around ``Evaluator.compare_splits``.

    Args:
        observed: Observed schedules with columns pid, act, start, end, duration.
        synthetic_schedules: ``{model_name: schedules_df}``.
        synthetic_attributes: ``{model_name: attributes_df}`` with ``pid`` column.
        target_attributes: Target attributes DataFrame with ``pid`` column.
        split_on: Attribute columns to split on.
        report_stats: Whether to append mean/std columns.
        verbose: Print progress.

    Returns:
        Dict of result DataFrames with per-split and per-label summaries.
    """
    evaluator = Evaluator(observed)
    return evaluator.compare_splits(
        synthetic_schedules=synthetic_schedules,
        synthetic_attributes=synthetic_attributes,
        target_attributes=target_attributes,
        split_on=split_on,
        report_stats=report_stats,
        verbose=verbose,
    )
