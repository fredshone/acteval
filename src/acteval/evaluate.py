import numpy as np
from pandas import DataFrame, MultiIndex, concat

from acteval._pipeline import (
    _model_cols_creativity,
    _model_cols_structural,
    _model_contribution,
    _observed_base,
    _observed_base_creativity,
    _observed_base_structural,
    add_stats,
    describe,
    describe_labels,
    evaluate,
    extract_default,
    process_metrics,
    score_features,
    subsample_and_evaluate,
)
from acteval._splits import (
    _all_feature_jobs,
    _key_activities,
    _precompute_pid_features,
    _subset_pid_features,
)
from acteval.jobs import get_jobs
from acteval.population import Population


class Evaluator:
    """Pre-computes target features once; compare multiple synthetic populations."""

    def __init__(self, target: DataFrame, config_path=None):
        self._target = target
        self._target_pop = Population(target)
        self._config_path = config_path
        self._target_features: dict[tuple, dict] = {}
        self._target_pid_features = {}
        self._precompute()

    def _precompute(self) -> None:
        for spec in _all_feature_jobs(self._config_path):
            key = (spec.domain, spec.name)
            pid_feat = spec.feature_fn(self._target_pop)
            self._target_features[key] = pid_feat.aggregate()
            self._target_pid_features[key] = pid_feat

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

        Loops populations → splits → features: for each model, all (split, cat)
        combinations are evaluated and its result columns are assembled in one pass.

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
        density_jobs, run_creativity, run_structural = get_jobs(self._config_path)

        synthetic_pops = {m: Population(y) for m, y in synthetic_schedules.items()}
        synth_pid_features = _precompute_pid_features(synthetic_pops, self._config_path)

        # --- precompute per-(split, cat) target data ---
        split_cat_info: list[tuple] = []
        for split in split_on:
            for cat in target_attributes[split].unique():
                target_orig_pids = target_attributes[
                    target_attributes[split] == cat
                ].pid.values
                target_dense_pids = self._target_pop.dense_pids_from_original(
                    target_orig_pids
                )
                sub_target = self._target[self._target.pid.isin(target_orig_pids)]
                cached_subset = {
                    key: pf.subset(target_dense_pids).aggregate()
                    for key, pf in self._target_pid_features.items()
                }
                split_cat_info.append((split, cat, sub_target, cached_subset))

        # --- observed base rows for all (split, cat) × features ---
        base_desc_parts: list[DataFrame] = []
        base_dist_parts: list[DataFrame] = []
        obs_hashes: dict[tuple, object] = {}  # {(split, cat): hash} — creativity only

        for split, cat, sub_target, cached_subset in split_cat_info:
            if run_creativity:
                bd, bi, obs_hash = _observed_base_creativity(sub_target)
                obs_hashes[(split, cat)] = obs_hash
                for df in (bd, bi):
                    df.index = MultiIndex.from_tuples(
                        [(*i, split, cat) for i in df.index],
                        names=list(df.index.names) + ["label", "cat"],
                    )
                base_desc_parts.append(bd)
                base_dist_parts.append(bi.drop("observed", axis=1))

            if run_structural:
                base_struct = _observed_base_structural(sub_target)
                base_struct.index = MultiIndex.from_tuples(
                    [(*i, split, cat) for i in base_struct.index],
                    names=list(base_struct.index.names) + ["label", "cat"],
                )
                base_desc_parts.append(base_struct)
                base_dist_parts.append(base_struct.drop("observed", axis=1))

            for spec in density_jobs:
                key = (spec.domain, spec.name)
                obs_feat = cached_subset[key]
                base, _ = _observed_base(spec, obs_feat)
                base.index = MultiIndex.from_tuples(
                    [(spec.domain, spec.name, f, split, cat) for f in base.index],
                    names=["domain", "feature", "segment", "label", "cat"],
                )
                base_desc_parts.append(base.assign(unit=spec.description_name))
                base_dist_parts.append(base[["observed__weight"]].assign(unit=spec.distance_name))

        base_desc = concat(base_desc_parts)
        base_dist = concat(base_dist_parts)

        # --- per-population columns: populations → splits/cats → features ---
        model_desc_cols: list[DataFrame] = []
        model_dist_cols: list[DataFrame] = []

        for model, pop in synthetic_pops.items():
            m_desc_parts: list[DataFrame] = []
            m_dist_parts: list[DataFrame] = []

            for split, cat, sub_target, cached_subset in split_cat_info:
                sample_pids = synthetic_attributes[model][
                    synthetic_attributes[model][split] == cat
                ].pid.values
                if verbose:
                    print(f">>> Subsampled {model} {split}={cat} with {len(sample_pids)}")
                sub_schedule = synthetic_schedules[model][
                    synthetic_schedules[model].pid.isin(sample_pids)
                ]
                synth_dense_pids = synthetic_pops[model].dense_pids_from_original(
                    sample_pids
                )
                synth_sub_acts = frozenset(sub_schedule["act"].unique())

                if run_creativity:
                    c_desc, c_dist = _model_cols_creativity(
                        model, sub_schedule, obs_hashes[(split, cat)]
                    )
                    for df in (c_desc, c_dist):
                        df.index = MultiIndex.from_tuples(
                            [(*i, split, cat) for i in df.index],
                            names=list(df.index.names) + ["label", "cat"],
                        )
                    m_desc_parts.append(c_desc)
                    m_dist_parts.append(c_dist)

                if run_structural:
                    s_cols = _model_cols_structural(model, sub_schedule, sub_target)
                    for label, parts in (("desc", m_desc_parts), ("dist", m_dist_parts)):
                        tagged = s_cols.copy()
                        tagged.index = MultiIndex.from_tuples(
                            [(*i, split, cat) for i in tagged.index],
                            names=list(tagged.index.names) + ["label", "cat"],
                        )
                        parts.append(tagged)

                for spec in density_jobs:
                    key = (spec.domain, spec.name)
                    obs_feat = cached_subset[key]
                    default = extract_default(obs_feat)

                    raw_synth = synth_pid_features[model][key].subset(synth_dense_pids).aggregate()
                    synth_feat = {
                        k: v for k, v in raw_synth.items()
                        if len(v[0]) > 0 and (
                            _key_activities(k) is None
                            or _key_activities(k).issubset(synth_sub_acts)
                        )
                    }

                    w, d, s = _model_contribution(model, spec, obs_feat, synth_feat, default)
                    desc_part = DataFrame({f"{model}__weight": w, model: d})
                    dist_part = DataFrame({f"{model}__weight": w, model: s})
                    desc_part.index = MultiIndex.from_tuples(
                        [(spec.domain, spec.name, f, split, cat) for f in desc_part.index],
                        names=["domain", "feature", "segment", "label", "cat"],
                    )
                    dist_part.index = MultiIndex.from_tuples(
                        [(spec.domain, spec.name, f, split, cat) for f in dist_part.index],
                        names=["domain", "feature", "segment", "label", "cat"],
                    )
                    m_desc_parts.append(desc_part)
                    m_dist_parts.append(dist_part)

            model_desc_cols.append(concat(m_desc_parts))
            model_dist_cols.append(concat(m_dist_parts))

        descriptions = concat([base_desc] + model_desc_cols, axis=1)
        distances = concat([base_dist] + model_dist_cols, axis=1)

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
