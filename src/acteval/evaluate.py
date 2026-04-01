import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, concat

from acteval._pipeline import (
    _aggregate_features,
    _make_default,
    _model_cols_creativity,
    _model_cols_structural,
    _model_contribution,
    _observed_base,
    _observed_base_creativity,
    _observed_base_structural,
)
from acteval._splits import (
    _get_density_jobs,
    _key_activities,
)
from acteval._aggregation import DEFAULT_REMOVE_FEATURES, DEFAULT_REMOVE_GROUPS
from acteval._result_frame import ResultFrame
from acteval.features import creativity, structural
from acteval._compat import _coerce_to_pandas, _is_dataframe
from acteval._jobs import get_jobs
from acteval.population import Population


class EvalResult:
    """Stores raw segment-level data; computes three-tier aggregation on demand.

    Use ``aggregate()`` to get the full set of output DataFrames, or access
    named properties (``descriptions``, ``domain_distances``, etc.) which call
    ``aggregate()`` with default parameters.

    Use ``save(path)`` to write all frames to CSV files.
    """

    def __init__(self, raw_desc: DataFrame, raw_dist: DataFrame):
        self._raw_desc = raw_desc   # (domain, feature, segment, label, cat) wide
        self._raw_dist = raw_dist

    # --- raw access ---

    @property
    def raw(self) -> dict[str, ResultFrame]:
        """Pre-aggregation data as ``ResultFrame`` objects (desc + dist)."""
        return {
            "descriptions": ResultFrame.from_wide(self._raw_desc),
            "distances": ResultFrame.from_wide(self._raw_dist),
        }

    # --- on-demand aggregation ---

    def aggregate(
        self,
        drop_features: list[tuple] | None = DEFAULT_REMOVE_FEATURES,
        drop_groups: list[tuple] | None = DEFAULT_REMOVE_GROUPS,
    ) -> dict[str, DataFrame]:
        """Compute all output tiers from raw segment-level data.

        Args:
            drop_features: Segment-level rows to drop before group aggregation.
                Defaults to ``DEFAULT_REMOVE_FEATURES``.  Pass ``None`` or ``[]``
                to skip filtering.
            drop_groups: Group-level rows to drop before domain aggregation.
                Defaults to ``DEFAULT_REMOVE_GROUPS``.  Pass ``None`` or ``[]``
                to skip filtering.

        Returns:
            Dict with keys: ``descriptions``, ``distances``,
            ``group_descriptions``, ``group_distances``,
            ``domain_descriptions``, ``domain_distances``.
            When real attribute splits were used, also includes ``label_*``
            variants of each key.
        """
        from acteval._aggregation import (
            descriptions_to_domain_level,
            descriptions_to_group_level,
            distances_to_domain_level,
            distances_to_group_level,
        )

        feat_desc, feat_dist = _aggregate_features(self._raw_desc, self._raw_dist)
        group_desc = descriptions_to_group_level(self._raw_desc, drop=drop_features)
        group_dist = distances_to_group_level(self._raw_dist, drop=drop_features)
        domain_desc = descriptions_to_domain_level(group_desc, drop=drop_groups)
        domain_dist = distances_to_domain_level(group_dist, drop=drop_groups)

        frames: dict[str, DataFrame] = {
            "descriptions": feat_desc,
            "distances": feat_dist,
            "group_descriptions": group_desc,
            "group_distances": group_dist,
            "domain_descriptions": domain_desc,
            "domain_distances": domain_dist,
        }

        # Include label_* only when real splits exist
        has_labels = not (
            self._raw_desc.index.get_level_values("label").unique().tolist() == ["__split__"]
        )
        if has_labels:
            lg_desc = descriptions_to_group_level(self._raw_desc, extra=["label"], drop=drop_features)
            lg_dist = distances_to_group_level(self._raw_dist, extra=["label"], drop=drop_features)
            ld_desc = descriptions_to_domain_level(lg_desc, extra=["label"], drop=drop_groups)
            ld_dist = distances_to_domain_level(lg_dist, extra=["label"], drop=drop_groups)
            frames.update({
                "label_descriptions": self._raw_desc,
                "label_distances": self._raw_dist,
                "label_group_descriptions": lg_desc,
                "label_group_distances": lg_dist,
                "label_domain_descriptions": ld_desc,
                "label_domain_distances": ld_dist,
            })

        return frames

    # --- named properties (call aggregate() with defaults) ---

    @property
    def descriptions(self) -> DataFrame:
        return self.aggregate()["descriptions"]

    @property
    def distances(self) -> DataFrame:
        return self.aggregate()["distances"]

    @property
    def group_descriptions(self) -> DataFrame:
        return self.aggregate()["group_descriptions"]

    @property
    def group_distances(self) -> DataFrame:
        return self.aggregate()["group_distances"]

    @property
    def domain_descriptions(self) -> DataFrame:
        return self.aggregate()["domain_descriptions"]

    @property
    def domain_distances(self) -> DataFrame:
        return self.aggregate()["domain_distances"]

    @property
    def label_descriptions(self) -> DataFrame | None:
        return self.aggregate().get("label_descriptions")

    @property
    def label_distances(self) -> DataFrame | None:
        return self.aggregate().get("label_distances")

    @property
    def label_group_descriptions(self) -> DataFrame | None:
        return self.aggregate().get("label_group_descriptions")

    @property
    def label_group_distances(self) -> DataFrame | None:
        return self.aggregate().get("label_group_distances")

    @property
    def label_domain_descriptions(self) -> DataFrame | None:
        return self.aggregate().get("label_domain_descriptions")

    @property
    def label_domain_distances(self) -> DataFrame | None:
        return self.aggregate().get("label_domain_distances")

    # --- dict-like interface (backward compat) ---

    def __getitem__(self, key: str) -> DataFrame:
        return self.aggregate()[key]

    def __iter__(self):
        return iter(self.aggregate())

    def keys(self):
        return self.aggregate().keys()

    def values(self):
        return self.aggregate().values()

    def items(self):
        return self.aggregate().items()

    # --- persistence ---

    def save(self, path: str | Path) -> None:
        """Save all aggregated frames to CSV files in *path*.

        Creates the directory if it does not exist.  One file per frame:
        ``descriptions.csv``, ``distances.csv``, ``group_descriptions.csv``, etc.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in self.aggregate().items():
            frame.to_csv(out / f"{name}.csv")

    # --- model introspection ---

    @property
    def model_names(self) -> list[str]:
        """Model column names, excluding 'observed', 'mean', 'std', and weight columns."""
        skip = {"observed", "mean", "std"}
        return [
            c
            for c in self.domain_distances.columns
            if c not in skip and not c.endswith("__weight")
        ]

    def summary(self) -> DataFrame:
        """Domain-level distances for each model (no observed/mean/std columns)."""
        return self.domain_distances[self.model_names]

    def rank_models(self) -> Series:
        """Mean domain distance per model, sorted ascending (lower is better)."""
        return self.summary().mean().sort_values()

    @property
    def best_model(self) -> str:
        """Model name with the lowest mean domain distance."""
        return self.rank_models().index[0]

    def __repr__(self) -> str:
        models = self.model_names
        header = f"EvalResult — {len(models)} model(s): {', '.join(models)}"
        return f"{header}\n\n{self.summary().to_string()}"


class Evaluator:
    """Pre-computes target features once; compare multiple synthetic populations."""

    def __init__(
        self,
        target: DataFrame,
        target_attributes: DataFrame | None = None,
        split_on: list[str] | None = None,
        config_path=None,
    ):
        target = _coerce_to_pandas(target)
        if target_attributes is not None:
            target_attributes = _coerce_to_pandas(target_attributes)
        if (target_attributes is None) != (split_on is None):
            raise ValueError(
                "target_attributes and split_on must both be provided or both be None"
            )
        self._numeric_bins: dict[str, tuple[np.ndarray, list[str]]] = {}
        if target_attributes is not None:
            if "pid" not in target_attributes.columns:
                raise ValueError(
                    "target_attributes DataFrame is missing required column 'pid'"
                )
            missing = [c for c in split_on if c not in target_attributes.columns]
            if missing:
                raise ValueError(
                    f"split_on column(s) {missing} not found in target_attributes"
                )
            _ORDINAL_LABELS = ["lowest", "low", "mid", "high", "highest"]
            for col in split_on:
                series = target_attributes[col]
                is_float = pd.api.types.is_float_dtype(series)
                is_large_int = (
                    pd.api.types.is_integer_dtype(series) and series.nunique() > 10
                )
                if is_float or is_large_int:
                    try:
                        _, edges = pd.qcut(
                            series, q=5, retbins=True, duplicates="drop"
                        )
                    except ValueError:
                        _, edges = pd.cut(series, bins=5, retbins=True)
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    actual_n = len(edges) - 1
                    labels = _ORDINAL_LABELS[:actual_n]
                    self._numeric_bins[col] = (edges, labels)
                    warnings.warn(
                        f"split_on column '{col}' is numeric; binning into "
                        f"{actual_n} ordinal bins {labels}. Encode as categorical "
                        f"to suppress this warning.",
                        UserWarning,
                        stacklevel=2,
                    )
            target_attributes = self._apply_numeric_bins(target_attributes)
        self._target = target
        self._target_pop = Population(target)
        self._config_path = config_path
        if target_attributes is None:
            target_attributes = DataFrame(
                {"pid": target["pid"].unique(), "__split__": "all"}
            )
            split_on = ["__split__"]
        self._target_attributes = target_attributes
        self._split_on = split_on
        self._target_pid_features = {}
        self._precompute_target()

    def __repr__(self) -> str:
        pop = self._target_pop
        acts = ", ".join(sorted(pop.unique_acts))
        lines = [
            "Evaluator(",
            f"  persons   : {pop.n:,}",
            f"  activities: {acts}",
        ]
        for split in self._split_on:
            if split == "__split__":
                continue
            cats = sorted(self._target_attributes[split].unique())
            lines.append(f"  split     : {split} \u2192 {cats}")
        lines.append(")")
        return "\n".join(lines)

    def _apply_numeric_bins(self, attributes: DataFrame) -> DataFrame:
        """Replace numeric split columns with ordinal bin labels using stored edges."""
        if not self._numeric_bins:
            return attributes
        attributes = attributes.copy()
        for col, (edges, labels) in self._numeric_bins.items():
            if col in attributes.columns:
                attributes[col] = pd.cut(
                    attributes[col], bins=edges, labels=labels, include_lowest=True
                ).astype(str)
        return attributes

    def _precompute_target(self) -> None:
        # Phase 1: run every feature function over the full target population,
        # storing per-pid results keyed by (domain, name).  These are cheap to
        # subset later, so we compute them once here rather than once per split.
        for spec in _get_density_jobs(self._config_path):
            key = (spec.domain, spec.name)
            self._target_pid_features[key] = spec.feature_fn(self._target_pop)

        self._density_jobs, self._run_creativity, self._run_structural = get_jobs(
            self._config_path
        )

        # Phase 2: for each (split, category) combination, slice the target
        # population to only the relevant pids and aggregate their features.
        # The result — (split, cat, sub_schedule, cached_feature_dict) — is
        # stored in _split_cat_info so compare_population can iterate over it.
        self._split_cat_info: list[tuple] = []
        for split in self._split_on:
            for cat in self._target_attributes[split].unique():
                target_orig_pids = self._target_attributes[
                    self._target_attributes[split] == cat
                ].pid.values
                target_dense_pids = self._target_pop.dense_pids_from_original(
                    target_orig_pids
                )
                sub_target = self._target[self._target.pid.isin(target_orig_pids)]
                # Aggregate per-pid features down to population-level distributions
                # for each (domain, name) key — these become the "observed" side
                # of every distance calculation.
                cached_subset = {
                    key: pf.subset(target_dense_pids).aggregate()
                    for key, pf in self._target_pid_features.items()
                }
                self._split_cat_info.append((split, cat, sub_target, cached_subset))

        # Phase 3: build the "observed" base rows that will sit alongside the
        # per-model columns in the final concatenated DataFrames.  Also cache
        # sequence hashes for creativity scoring.
        base_desc_parts: list[DataFrame] = []
        base_dist_parts: list[DataFrame] = []
        self._obs_hashes: dict[tuple, object] = {}

        for split, cat, sub_target, cached_subset in self._split_cat_info:
            if self._run_creativity:
                bd, bi, obs_hash = _observed_base_creativity(sub_target)
                # obs_hash is a frozenset of sequence hashes; used in
                # compare_population to flag novel (unseen) sequences.
                self._obs_hashes[(split, cat)] = obs_hash
                for df in (bd, bi):
                    df.index = MultiIndex.from_tuples(
                        [(*i, split, cat) for i in df.index],
                        names=list(df.index.names) + ["label", "cat"],
                    )
                base_desc_parts.append(bd)
                base_dist_parts.append(bi.drop("observed", axis=1))

            if self._run_structural:
                base_struct = _observed_base_structural(sub_target)
                base_struct.index = MultiIndex.from_tuples(
                    [(*i, split, cat) for i in base_struct.index],
                    names=list(base_struct.index.names) + ["label", "cat"],
                )
                base_desc_parts.append(base_struct)
                base_dist_parts.append(base_struct.drop("observed", axis=1))

            for spec in self._density_jobs:
                key = (spec.domain, spec.name)
                obs_feat = cached_subset[key]
                base, _ = _observed_base(spec, obs_feat)
                base.index = MultiIndex.from_tuples(
                    [(spec.domain, spec.name, f, split, cat) for f in base.index],
                    names=["domain", "feature", "segment", "label", "cat"],
                )
                base_desc_parts.append(base.assign(unit=spec.description_name))
                base_dist_parts.append(
                    base[["observed__weight"]].assign(unit=spec.distance_name)
                )

        # _base_desc / _base_dist are the "observed" half of the wide DataFrames
        # that compare_splits will build by concatenating model columns alongside.
        self._base_desc = concat(base_desc_parts)
        self._base_dist = concat(base_dist_parts)
        self.collected_descriptions: dict[str, DataFrame] = {}
        self.collected_distances: dict[str, DataFrame] = {}

    def compare(
        self,
        synthetic: dict[str, DataFrame],
        attributes: dict[str, DataFrame] | None = None,
        verbose: bool = False,
    ) -> "EvalResult":
        """Compare synthetic populations against pre-computed target features.

        Args:
            synthetic: ``{model_name: schedules_df}``.
            attributes: Optional ``{model_name: attributes_df}`` with ``pid``
                column.  If provided, enables attribute-based splitting and
                exposes ``label_*`` frames on the result.
            verbose: Print progress for each (split, category) subset.
        """
        if attributes is not None:
            return self.compare_populations(
                synthetic_schedules=synthetic,
                synthetic_attributes=attributes,
                verbose=verbose,
            )
        synthetic = {m: _coerce_to_pandas(df) for m, df in synthetic.items()}
        synth_attrs = {
            m: DataFrame({"pid": df["pid"].unique(), "__split__": "all"})
            for m, df in synthetic.items()
        }
        return self.compare_populations(
            synthetic_schedules=synthetic,
            synthetic_attributes=synth_attrs,
            verbose=verbose,
        )

    def compare_population(
        self,
        model: str,
        schedule: DataFrame,
        attributes: DataFrame | None = None,
        verbose: bool = False,
    ) -> None:
        """Compute description and distance columns for a single synthetic population.

        Results are stored on ``self._population_descs[model]`` and
        ``self._population_dists[model]``.  Call ``report()`` after all models
        have been compared to assemble the final ``EvalResult``.

        Args:
            model: Model name.
            schedule: Schedules DataFrame for this model.
            attributes: Attributes DataFrame for this model (with ``pid`` column).
                Required when the evaluator was initialised with splits; omit
                (or pass ``None``) when no splits are in use.
            verbose: Print progress.
        """
        schedule = _coerce_to_pandas(schedule)
        if attributes is not None:
            attributes = _coerce_to_pandas(attributes)
        uses_real_splits = self._split_on != ["__split__"]
        if attributes is None:
            if uses_real_splits:
                raise ValueError(
                    "attributes is required when the Evaluator was initialised with splits"
                )
            attributes = DataFrame(
                {"pid": schedule["pid"].unique(), "__split__": "all"}
            )
        else:
            if "pid" not in attributes.columns:
                raise ValueError(
                    f"attributes DataFrame for model '{model}' is missing required column 'pid'"
                )
            missing = [c for c in self._split_on if c not in attributes.columns]
            if missing:
                raise ValueError(
                    f"attributes DataFrame for model '{model}' is missing split column(s) {missing}"
                )
        attributes = self._apply_numeric_bins(attributes)
        pop = Population(schedule)
        pid_features = {
            (spec.domain, spec.name): spec.feature_fn(pop)
            for spec in _get_density_jobs(self._config_path)
        }
        description_parts: list[DataFrame] = []
        distance_parts: list[DataFrame] = []

        # Phase 1: pre-compute full-population features once (mirrors
        # _precompute_target Phase 1).  Creativity hashes and structural
        # feasibility flags are computed here for the entire synthetic
        # population; the split loop below only subsets them.
        # Per-pid hashes are needed by both creativity metrics and by structural
        # (to replicate the filter_novel step per split without re-scanning the
        # schedule each time).
        if self._run_creativity or self._run_structural:
            pid_hashes = creativity.hash_per_pid(schedule)
        if self._run_structural:
            feasibility_flags = structural.feasibility(pop)

        # Phase 2: iterate over (split, category) combinations and subset the
        # pre-computed features down to the relevant pids.
        for split, cat, _, cached_subset in self._split_cat_info:
            sample_pids = attributes[attributes[split] == cat].pid.values
            if verbose:
                print(f">>> Subsampled {model} {split}={cat} with {len(sample_pids)}")
            synth_dense_pids = pop.dense_pids_from_original(sample_pids)
            # Used below to skip density segments whose key activity is absent
            # from this synthetic sub-population entirely.
            synth_sub_acts = frozenset(
                schedule.loc[schedule.pid.isin(sample_pids), "act"].unique()
            )

            if self._run_creativity:
                # Subset pre-computed hashes to this split's pids; no
                # re-hashing of the schedule is needed.
                c_desc, c_dist = _model_cols_creativity(
                    model, pid_hashes, sample_pids, self._obs_hashes[(split, cat)]
                )
                for df in (c_desc, c_dist):
                    df.index = MultiIndex.from_tuples(
                        [(*i, split, cat) for i in df.index],
                        names=list(df.index.names) + ["label", "cat"],
                    )
                description_parts.append(c_desc)
                distance_parts.append(c_dist)

            if self._run_structural:
                # Find novel dense pids: synthetic persons in this split whose
                # sequence is not present in the corresponding target subset.
                obs_hash = self._obs_hashes[(split, cat)]
                novel_sample_pids = np.array(
                    [p for p in sample_pids if pid_hashes.get(p) not in obs_hash]
                )
                novel_dense_pids = pop.dense_pids_from_original(novel_sample_pids)
                s_cols = _model_cols_structural(
                    model, feasibility_flags, novel_dense_pids
                )
                for parts in (description_parts, distance_parts):
                    tagged = s_cols.copy()
                    tagged.index = MultiIndex.from_tuples(
                        [(*i, split, cat) for i in tagged.index],
                        names=list(tagged.index.names) + ["label", "cat"],
                    )
                    parts.append(tagged)

            for spec in self._density_jobs:
                key = (spec.domain, spec.name)
                obs_feat = cached_subset[key]
                # default holds the observed distribution shape; used as a
                # fallback when the synthetic model has no data for a segment.
                default = _make_default(obs_feat)

                # Aggregate pre-computed per-pid features for just the pids in
                # this split category, then drop segments where:
                #   - the array is empty, or
                #   - the segment's key activity is absent from the synthetic
                #     sub-population (avoids spurious missing-activity penalties).
                raw_synth = pid_features[key].subset(synth_dense_pids).aggregate()
                synth_feat = {
                    k: v
                    for k, v in raw_synth.items()
                    if len(v[0]) > 0
                    and (
                        _key_activities(k) is None
                        or _key_activities(k).issubset(synth_sub_acts)
                    )
                }

                # w = weights, d = descriptive values, s = distance scores
                w, d, s = _model_contribution(
                    model, spec, obs_feat, synth_feat, default
                )
                desc_part = DataFrame({f"{model}__weight": w, model: d})
                dist_part = DataFrame(
                    {f"{model}__weight": w.reindex(s.index, fill_value=0), model: s}
                )
                desc_part.index = MultiIndex.from_tuples(
                    [(spec.domain, spec.name, f, split, cat) for f in desc_part.index],
                    names=["domain", "feature", "segment", "label", "cat"],
                )
                dist_part.index = MultiIndex.from_tuples(
                    [(spec.domain, spec.name, f, split, cat) for f in dist_part.index],
                    names=["domain", "feature", "segment", "label", "cat"],
                )
                description_parts.append(desc_part)
                distance_parts.append(dist_part)

        # Store results so compare_splits can later concat them with _base_desc/dist.
        self.collected_descriptions[model] = concat(
            [p for p in description_parts if not p.empty]
        )
        self.collected_distances[model] = concat(
            [p for p in distance_parts if not p.empty]
        )

    def report(self) -> EvalResult:
        """Assemble an ``EvalResult`` from previously accumulated model comparisons.

        Call this after one or more ``compare_population`` calls.

        Returns:
            EvalResult wrapping the raw segment-level data.  Call
            ``result.aggregate()`` or access named properties to get the
            three-tier aggregated output.
        """
        descriptions = concat(
            [self._base_desc] + list(self.collected_descriptions.values()), axis=1
        )
        distances = concat(
            [self._base_dist] + list(self.collected_distances.values()), axis=1
        )
        return EvalResult(raw_desc=descriptions, raw_dist=distances)

    def compare_populations(
        self,
        synthetic_schedules: dict[str, DataFrame],
        synthetic_attributes: dict[str, DataFrame],
        verbose: bool = False,
    ) -> "EvalResult":
        """Compare synthetic populations against target, split by attribute categories.

        Convenience wrapper: resets state, calls ``compare_population`` for each
        model, then returns ``report()``.

        Args:
            synthetic_schedules: ``{model_name: schedules_df}``.
            synthetic_attributes: ``{model_name: attributes_df}`` with ``pid`` column.
            verbose: Print progress.

        Returns:
            EvalResult wrapping raw segment-level data.
        """
        self.collected_descriptions = {}
        self.collected_distances = {}
        for model, schedule in synthetic_schedules.items():
            self.compare_population(
                model=model,
                schedule=schedule,
                attributes=synthetic_attributes[model],
                verbose=verbose,
            )

        return self.report()


def compare(
    observed: DataFrame,
    synthetic,
    attributes: dict[str, DataFrame] | None = None,
    verbose: bool = False,
) -> EvalResult:
    """Compare observed and synthetic activity schedule populations.

    Args:
        observed: Observed schedules with columns pid, act, start, end, duration.
        synthetic: Single synthetic DataFrame or dict mapping model names to DataFrames.
        attributes: Optional ``{model_name: attributes_df}`` with ``pid`` column.
            If provided, enables attribute-based splitting (exposes ``label_*`` frames).
        verbose: Print progress for each (split, category) subset.

    Returns:
        EvalResult with raw segment-level data; use ``aggregate()`` or named
        properties for the three-tier output.
    """
    if _is_dataframe(synthetic):
        synthetic = {"synthetic": synthetic}
    return Evaluator(observed).compare(synthetic, attributes=attributes, verbose=verbose)


def compare_splits(
    observed: DataFrame,
    synthetic_schedules: dict[str, DataFrame],
    synthetic_attributes: dict[str, DataFrame],
    target_attributes: DataFrame,
    split_on: list[str],
    verbose: bool = False,
) -> EvalResult:
    """Compare observed and synthetic populations, split by attribute categories.

    Convenience wrapper around ``Evaluator.compare_populations``.

    Args:
        observed: Observed schedules with columns pid, act, start, end, duration.
        synthetic_schedules: ``{model_name: schedules_df}``.
        synthetic_attributes: ``{model_name: attributes_df}`` with ``pid`` column.
        target_attributes: Target attributes DataFrame with ``pid`` column.
        split_on: Attribute columns to split on.
        verbose: Print progress.

    Returns:
        EvalResult with raw segment-level data; includes ``label_*`` frames
        when real attribute splits are present.
    """
    evaluator = Evaluator(observed, target_attributes, split_on)
    return evaluator.compare_populations(
        synthetic_schedules=synthetic_schedules,
        synthetic_attributes=synthetic_attributes,
        verbose=verbose,
    )
