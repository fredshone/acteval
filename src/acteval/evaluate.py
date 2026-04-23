import warnings
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, concat

from acteval._aggregation import DEFAULT_REMOVE_FEATURES, DEFAULT_REMOVE_GROUPS
from acteval._compat import _coerce_to_pandas, _is_dataframe
from acteval._jobs import get_jobs
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
from acteval._result_frame import ResultFrame
from acteval._splits import (
    _get_density_jobs,
    _key_activities,
)
from acteval.features import creativity, structural
from acteval.population import Population


class SplitNotAvailableError(AttributeError):
    """Raised when ``.by_attribute`` or ``.by_category`` is accessed on an
    ``EvalResult`` produced without ``split_on``."""


class AggregatedResult:
    """A pair of descriptions and distances DataFrames at one aggregation level.

    Returned by ``ScheduleView.combined``, ``.by_attribute``, and ``.by_category``.
    """

    def __init__(
        self,
        descriptions: DataFrame,
        distances: DataFrame,
        _label: str = "",
    ):
        self.descriptions = descriptions
        self.distances = distances
        self._label = _label

    def __repr__(self) -> str:
        header = (
            f"AggregatedResult [{self._label}]" if self._label else "AggregatedResult"
        )
        return f"{header}\n\n{self.distances.to_string()}"

    def save(self, path: str | Path) -> None:
        """Write ``descriptions.csv`` and ``distances.csv`` to *path*."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.descriptions.to_csv(out / "descriptions.csv")
        self.distances.to_csv(out / "distances.csv")


class ScheduleView:
    """Accessor for one schedule aggregation level (features / groups / domains).

    Access ``.combined``, ``.by_attribute``, or ``.by_category`` to get an
    ``AggregatedResult``.  Split-based views raise ``SplitNotAvailableError``
    when the parent ``EvalResult`` was produced without ``split_on``.
    """

    def __init__(
        self,
        raw_desc: DataFrame,
        raw_dist: DataFrame,
        schedule: Literal["features", "groups", "domains"],
        has_splits: bool,
        drop_features,
        drop_groups,
    ):
        self._raw_desc = raw_desc
        self._raw_dist = raw_dist
        self._schedule = schedule
        self._has_splits = has_splits
        self._drop_features = drop_features
        self._drop_groups = drop_groups

    def _compute(self, extra: list[str]) -> AggregatedResult:
        from acteval._aggregation import (
            descriptions_to_domain_level,
            descriptions_to_group_level,
            distances_to_domain_level,
            distances_to_group_level,
        )

        split_name = (
            "combined"
            if not extra
            else ("by_attribute" if extra == ["label"] else "by_category")
        )
        label = f"{self._schedule} × {split_name}"
        if self._schedule == "features":
            desc, dist = _aggregate_features(
                self._raw_desc, self._raw_dist, extra=extra
            )
        elif self._schedule == "groups":
            desc = descriptions_to_group_level(
                self._raw_desc, extra=extra, drop=self._drop_features
            )
            dist = distances_to_group_level(
                self._raw_dist, extra=extra, drop=self._drop_features
            )
        else:  # domains
            group_desc = descriptions_to_group_level(
                self._raw_desc, extra=extra, drop=self._drop_features
            )
            group_dist = distances_to_group_level(
                self._raw_dist, extra=extra, drop=self._drop_features
            )
            desc = descriptions_to_domain_level(
                group_desc, extra=extra, drop=self._drop_groups
            )
            dist = distances_to_domain_level(
                group_dist, extra=extra, drop=self._drop_groups
            )
        return AggregatedResult(desc, dist, _label=label)

    @cached_property
    def combined(self) -> AggregatedResult:
        """Aggregated result with splits merged away."""
        return self._compute([])

    @cached_property
    def by_attribute(self) -> AggregatedResult:
        """Aggregated result split by attribute (one row per label value).

        Raises ``SplitNotAvailableError`` if the parent ``EvalResult`` was
        produced without ``split_on``.
        """
        if not self._has_splits:
            raise SplitNotAvailableError(
                "by_attribute is not available: this EvalResult was produced without split_on. "
                "Pass target_attributes and split_on to Evaluator to enable split-based views."
            )
        return self._compute(["label"])

    @cached_property
    def by_category(self) -> AggregatedResult:
        """Aggregated result split by attribute category (one row per label × cat).

        Raises ``SplitNotAvailableError`` if the parent ``EvalResult`` was
        produced without ``split_on``.
        """
        if not self._has_splits:
            raise SplitNotAvailableError(
                "by_category is not available: this EvalResult was produced without split_on. "
                "Pass target_attributes and split_on to Evaluator to enable split-based views."
            )
        return self._compute(["label", "cat"])

    def __repr__(self) -> str:
        available = [".combined"]
        if self._has_splits:
            available += [".by_attribute", ".by_category"]
        attrs = " / ".join(available)
        return (
            f"ScheduleView [{self._schedule}]  ({attrs})\n\n"
            f"{self.combined.distances.to_string()}"
        )


class EvalResult:
    """Stores raw segment-level data; computes three-tier aggregation on demand.

    Access ``result.features``, ``result.groups``, or ``result.domains`` to get
    a ``ScheduleView``, then ``.combined``, ``.by_attribute``, or
    ``.by_category`` to obtain an ``AggregatedResult``.

    Examples::

        result.domains.combined.distances      # domain-level distances
        result.groups.by_attribute.distances   # group-level, split by attribute
        result.features.by_category.save("out/raw/")
    """

    def __init__(self, raw_desc: DataFrame, raw_dist: DataFrame):
        self._raw_desc = raw_desc  # (domain, feature, segment, label, cat) wide
        self._raw_dist = raw_dist

    # --- raw access ---

    @property
    def raw(self) -> dict[str, ResultFrame]:
        """Pre-aggregation data as ``ResultFrame`` objects (desc + dist)."""
        return {
            "descriptions": ResultFrame.from_wide(self._raw_desc),
            "distances": ResultFrame.from_wide(self._raw_dist),
        }

    # --- split availability ---

    @property
    def has_splits(self) -> bool:
        """True when the ``Evaluator`` was run with ``split_on``."""
        return not (
            self._raw_desc.index.get_level_values("label").unique().tolist()
            == ["__split__"]
        )

    # --- schedule-level accessors ---

    @cached_property
    def features(self) -> ScheduleView:
        """Feature-level view: index ``(domain, feature, segment[, ...])``.

        Most granular schedule level; useful for disk storage.
        """
        return ScheduleView(
            self._raw_desc,
            self._raw_dist,
            schedule="features",
            has_splits=self.has_splits,
            drop_features=DEFAULT_REMOVE_FEATURES,
            drop_groups=DEFAULT_REMOVE_GROUPS,
        )

    @cached_property
    def groups(self) -> ScheduleView:
        """Group-level view: index ``(domain, feature[, ...])``.

        Intermediate schedule level; one row per feature group.
        """
        return ScheduleView(
            self._raw_desc,
            self._raw_dist,
            schedule="groups",
            has_splits=self.has_splits,
            drop_features=DEFAULT_REMOVE_FEATURES,
            drop_groups=DEFAULT_REMOVE_GROUPS,
        )

    @cached_property
    def domains(self) -> ScheduleView:
        """Domain-level view: index ``(domain[, ...])``.

        Most aggregated level; best for terminal output and quick review.
        """
        return ScheduleView(
            self._raw_desc,
            self._raw_dist,
            schedule="domains",
            has_splits=self.has_splits,
            drop_features=DEFAULT_REMOVE_FEATURES,
            drop_groups=DEFAULT_REMOVE_GROUPS,
        )

    # --- model introspection ---

    @property
    def model_names(self) -> list[str]:
        """Model column names, excluding 'observed', 'mean', 'std', and weight columns."""
        skip = {"observed", "mean", "std"}
        return [
            c
            for c in self.domains.combined.distances.columns
            if c not in skip and not c.endswith("__weight")
        ]

    def summary(self) -> DataFrame:
        """Domain-level distances for each model (no observed/mean/std columns)."""
        return self.domains.combined.distances[self.model_names]

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

    # --- persistence ---

    def save(self, path: str | Path) -> None:
        """Save aggregated frames to CSV files under *path*.

        Creates subdirectories for each schedule × split combination.
        Combined tiers are always written; split-based tiers are written only
        when the ``EvalResult`` was produced with ``split_on``.
        """
        out = Path(path)
        self.features.combined.save(out / "features")
        self.groups.combined.save(out / "groups")
        self.domains.combined.save(out / "domains")
        if self.has_splits:
            self.features.by_attribute.save(out / "features_by_attribute")
            self.features.by_category.save(out / "features_by_category")
            self.groups.by_attribute.save(out / "groups_by_attribute")
            self.groups.by_category.save(out / "groups_by_category")
            self.domains.by_attribute.save(out / "domains_by_attribute")
            self.domains.by_category.save(out / "domains_by_category")


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
                        _, edges = pd.qcut(series, q=5, retbins=True, duplicates="drop")
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

    def compare_populations(
        self,
        synthetic_schedules: dict[str, DataFrame],
        synthetic_attributes: dict[str, DataFrame] | None = None,
        verbose: bool = False,
    ) -> "EvalResult":
        """Compare synthetic populations against target, split by attribute categories.

        Convenience wrapper: resets state, calls ``compare_population`` for each
        model, then returns ``report()``.

        Args:
            synthetic_schedules: ``{model_name: schedules_df}``.
            synthetic_attributes: Optional ``{model_name: attributes_df}`` with ``pid``
                column.  If omitted, no attribute-based splitting is applied.
            verbose: Print progress.

        Returns:
            EvalResult wrapping raw segment-level data.
        """
        self.collected_descriptions = {}
        self.collected_distances = {}
        for model, schedule in synthetic_schedules.items():
            attrs = (
                synthetic_attributes[model]
                if synthetic_attributes is not None
                else None
            )
            self.compare_population(
                model=model,
                schedule=schedule,
                attributes=attrs,
                verbose=verbose,
            )

        return self.report()

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
    return Evaluator(observed).compare(
        synthetic, attributes=attributes, verbose=verbose
    )


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
