"""Comparison entry points: Evaluator, compare(), compare_grid(), compare_many().

`EvalResult` and the classes/helpers it's built from live in `results.py`;
this module is just the API surface that produces one.
"""

import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series, concat

from acteval._compat import _coerce_to_pandas, _is_dataframe
from acteval._jobs import EvalConfig, get_jobs
from acteval._pipeline import (
    _make_default,
    _model_cols_creativity,
    _model_cols_structural,
    _model_contribution,
    _observed_base,
    _observed_base_creativity,
    _observed_base_structural,
)
from acteval._progress import bar_scope as _bar_scope
from acteval._progress import bar_set_item as _bar_set_item
from acteval._progress import make_bar as _make_bar
from acteval._result_frame import ResultFrame
from acteval._splits import build_slices, group_pids
from acteval.features import creativity, structural
from acteval.population import Population
from acteval.results import (
    EvalResult,
    _append_split_cat_index,
    _concat_tagged,
    _tag_density_index,
    combine,
)

# Column name the target's own descriptions take alongside the model columns.
_TARGET_COL = "target"
# Column names the results use alongside model columns, so no model may take
# them: the target's descriptions, and the unit column added at the feature
# and group levels.
_RESERVED_MODEL_NAMES = {
    _TARGET_COL: "the target population's column",
    "unit": "the unit column",
}


def _check_model_names(models) -> None:
    for name, use in _RESERVED_MODEL_NAMES.items():
        if name in models:
            raise ValueError(
                f"model name {name!r} is reserved for {use} in the results; "
                "rename that model"
            )


def _check_pids_match(schedules: DataFrame, attributes: DataFrame, who: str) -> None:
    """Raise unless *schedules* and *attributes* cover exactly the same pids."""
    sched_pids = pd.Index(pd.unique(schedules["pid"]))
    attr_pids = pd.Index(pd.unique(attributes["pid"]))
    problems = []
    for missing, has, lacks in (
        (sched_pids.difference(attr_pids), "schedules", "attributes"),
        (attr_pids.difference(sched_pids), "attributes", "schedules"),
    ):
        if len(missing):
            examples = ", ".join(repr(p) for p in missing[:5])
            more = ", ..." if len(missing) > 5 else ""
            problems.append(
                f"{len(missing)} pid(s) in the {has} have no {lacks} "
                f"(e.g. {examples}{more})"
            )
    if problems:
        raise ValueError(
            f"{who}: schedules and attributes must cover the same persons; "
            + "; ".join(problems)
        )


class Evaluator:
    """Pre-computes target features once; compare multiple synthetic populations."""

    def __init__(
        self,
        target: DataFrame,
        target_attributes: DataFrame | None = None,
        split_on: list[str] | None = None,
        config_path=None,
        jobs: EvalConfig | None = None,
        progress: bool = False,
        disable: list[str] | None = None,
        split_order: int | None = None,
        min_split_size: int | None = None,
    ):
        target = _coerce_to_pandas(target)
        if target_attributes is not None:
            target_attributes = _coerce_to_pandas(target_attributes)
        if jobs is not None and (config_path is not None or disable):
            raise ValueError(
                "jobs is a fully-built EvalConfig and cannot be combined with "
                "config_path or disable; pass one or the other"
            )
        if split_order is not None and split_order > 1 and split_on is None:
            raise ValueError("split_order > 1 requires split_on")
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
        if target_attributes is not None:
            _check_pids_match(target, target_attributes, "target")
        self._config_path = config_path
        if target_attributes is None:
            target_attributes = DataFrame(
                {"pid": target["pid"].unique(), "__split__": "all"}
            )
            split_on = ["__split__"]
        self._target_attributes = target_attributes
        self._split_on = split_on
        self._target_pid_features = {}
        self._jobs: EvalConfig = (
            jobs if jobs is not None else get_jobs(config_path, disable)
        )
        self._uses_real_splits = split_on != ["__split__"]
        if not self._uses_real_splits:
            split_order = 1
        elif split_order is None:
            split_order = self._jobs.splits.order
        self._split_order = split_order
        min_size = (
            min_split_size if min_split_size is not None else self._jobs.splits.min_size
        )
        self._slices = build_slices(
            self._target_attributes,
            split_on,
            self._split_order,
            min_size if self._uses_real_splits else 0,
        )
        self._progress = progress
        self._precomputed = False
        # Progress bars for the model currently being processed by
        # compare_population(); set by _compare_populations() so bar plumbing
        # doesn't have to appear in compare_population()'s public signature.
        self._active_feature_bar = None
        self._active_splits_bar = None

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
            lines.append(f"  split     : {split} → {cats}")
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

    def _precompute_target(self, feature_bar=None, splits_bar=None) -> None:
        # Phase 1: run every feature function over the full target population,
        # storing per-pid results keyed by (domain, name).  These are cheap to
        # subset later, so we compute them once here rather than once per split.
        with _bar_scope(
            feature_bar,
            "target [features]",
            len(self._jobs.density),
            self._progress,
            colour="cyan",
        ) as feature_bar:
            for spec in self._jobs.density:
                _bar_set_item(feature_bar, spec.name)
                key = (spec.domain, spec.name)
                self._target_pid_features[key] = spec.feature_fn(self._target_pop)
                feature_bar.update(1)

        # Phase 2: for each slice (a joint assignment of split_order attributes,
        # built in __init__), restrict the target population to the relevant
        # pids and aggregate their features. The result —
        # (slice, sub_schedule, cached_feature_dict, n_target_persons) — is
        # stored in _split_cat_info so compare_population can iterate over it.
        self._split_cat_info: list[tuple] = []
        for slice_, target_orig_pids in self._slices:
            target_dense_pids = self._target_pop.dense_pids_from_original(
                target_orig_pids
            )
            sub_target = self._target[self._target.pid.isin(target_orig_pids)]
            # Aggregate per-pid features down to population-level distributions
            # for each (domain, name) key — these become the "target" side
            # of every distance calculation. Segments missing from this slice
            # are dropped, so they get no row unless the model has them.
            cached_subset = {
                key: pf.aggregate_present(target_dense_pids)
                for key, pf in self._target_pid_features.items()
            }
            self._split_cat_info.append(
                (slice_, sub_target, cached_subset, len(target_dense_pids))
            )

        # Phase 3: build the target's per-row (value, weight, unit) Series that
        # will sit alongside the per-model columns in the final ResultFrames.
        # Also cache sequence hashes for creativity and novel-structural scoring.
        desc_value_parts: list[Series] = []
        desc_weight_parts: list[Series] = []
        desc_unit_parts: list[Series] = []
        dist_weight_parts: list[Series] = []
        dist_unit_parts: list[Series] = []
        self._obs_hashes: dict[tuple, object] = {}
        _needs_hashes = (
            self._jobs.creativity.enabled or self._jobs.structural.needs_novel_pids
        )

        with _bar_scope(
            splits_bar,
            "target [splits]",
            len(self._split_cat_info),
            self._progress,
            colour="cyan",
        ) as splits_bar:
            for slice_, sub_target, cached_subset, _ in self._split_cat_info:
                split, cat = slice_.key
                _bar_set_item(
                    splits_bar, cat if split == "__split__" else f"{split}={cat}"
                )
                if _needs_hashes:
                    obs_hash = creativity.hash_population(Population(sub_target))
                    self._obs_hashes[(split, cat)] = obs_hash

                if self._jobs.creativity.enabled:
                    dv, dw, du, xw, xu = _observed_base_creativity(
                        sub_target,
                        self._obs_hashes[(split, cat)],
                        self._jobs.creativity,
                    )
                    desc_value_parts.append(_append_split_cat_index(dv, split, cat))
                    desc_weight_parts.append(_append_split_cat_index(dw, split, cat))
                    desc_unit_parts.append(_append_split_cat_index(du, split, cat))
                    dist_weight_parts.append(_append_split_cat_index(xw, split, cat))
                    dist_unit_parts.append(_append_split_cat_index(xu, split, cat))

                if self._jobs.structural.enabled:
                    sv, sw, su = _observed_base_structural(
                        sub_target, self._jobs.structural
                    )
                    sv = _append_split_cat_index(sv, split, cat)
                    sw = _append_split_cat_index(sw, split, cat)
                    su = _append_split_cat_index(su, split, cat)
                    desc_value_parts.append(sv)
                    desc_weight_parts.append(sw)
                    desc_unit_parts.append(su)
                    # Structural reuses the same weight/unit for distances —
                    # there's nothing distance-specific about a feasibility flag.
                    dist_weight_parts.append(sw)
                    dist_unit_parts.append(su)

                for spec in self._jobs.density:
                    key = (spec.domain, spec.name)
                    obs_feat = cached_subset[key]
                    value, weight, _ = _observed_base(
                        spec, obs_feat, self._target_pid_features[key].zero_is_missing
                    )
                    value = _tag_density_index(
                        value, spec.domain, spec.name, split, cat
                    )
                    weight = _tag_density_index(
                        weight, spec.domain, spec.name, split, cat
                    )
                    desc_value_parts.append(value)
                    desc_weight_parts.append(weight)
                    desc_unit_parts.append(
                        Series(spec.description_name, index=value.index)
                    )
                    # Distances reuse the same target weight as descriptions.
                    dist_weight_parts.append(weight)
                    dist_unit_parts.append(
                        Series(spec.distance_name, index=weight.index)
                    )

                splits_bar.update(1)

        # These are the target's contribution to the final ResultFrames that
        # compare_population/report will build by concatenating model columns
        # alongside them.
        if not desc_value_parts:
            raise ValueError(
                "No evaluation jobs are enabled (check config/disable); nothing to report."
            )
        self._target_description_values = _concat_tagged(desc_value_parts)
        self._target_description_weights = _concat_tagged(desc_weight_parts)
        self._target_description_units = _concat_tagged(desc_unit_parts)
        self._target_distance_weights = _concat_tagged(dist_weight_parts)
        self._target_distance_units = _concat_tagged(dist_unit_parts)
        self.collected_description_values: dict[str, Series] = {}
        self.collected_description_weights: dict[str, Series] = {}
        self.collected_distance_values: dict[str, Series] = {}
        self.collected_distance_weights: dict[str, Series] = {}
        self._precomputed = True

    def compare(
        self,
        synthetic: dict[str, DataFrame],
        attributes: dict[str, DataFrame] | None = None,
        verbose: bool = False,
    ) -> EvalResult:
        """Compare synthetic populations against pre-computed target features.

        This is the primary entry point for running multiple synthetic
        comparisons against the same observed data (splits, if any, are
        configured once via the constructor).

        Args:
            synthetic: ``{model_name: schedules_df}``.
            attributes: Optional ``{model_name: attributes_df}`` with ``pid``
                column.  If provided, enables attribute-based splitting and
                exposes ``label_*`` frames on the result.
            verbose: Print progress for each (split, category) subset.
        """
        if attributes is not None:
            return self._compare_populations(
                synthetic_schedules=synthetic,
                synthetic_attributes=attributes,
                verbose=verbose,
            )
        synthetic = {m: _coerce_to_pandas(df) for m, df in synthetic.items()}
        synth_attrs = {
            m: DataFrame({"pid": df["pid"].unique(), "__split__": "all"})
            for m, df in synthetic.items()
        }
        return self._compare_populations(
            synthetic_schedules=synthetic,
            synthetic_attributes=synth_attrs,
            verbose=verbose,
        )

    def _compare_populations(
        self,
        synthetic_schedules: dict[str, DataFrame],
        synthetic_attributes: dict[str, DataFrame] | None = None,
        verbose: bool = False,
    ) -> EvalResult:
        """Shared implementation behind ``Evaluator.compare``."""
        _check_model_names(synthetic_schedules)
        self.collected_description_values = {}
        self.collected_description_weights = {}
        self.collected_distance_values = {}
        self.collected_distance_weights = {}

        if self._uses_real_splits:
            if synthetic_attributes is None:
                raise ValueError(
                    "attributes is required for every model when the Evaluator "
                    "was initialised with splits; missing for: "
                    f"{sorted(synthetic_schedules)}"
                )
            invalid: dict[str, str] = {}
            for model in synthetic_schedules:
                attrs = synthetic_attributes.get(model)
                if attrs is None:
                    invalid[model] = "no attributes provided"
                    continue
                missing_cols = [c for c in self._split_on if c not in attrs.columns]
                if missing_cols:
                    invalid[model] = f"missing split column(s) {missing_cols}"
            if invalid:
                details = "; ".join(f"'{m}' ({why})" for m, why in invalid.items())
                raise ValueError(
                    "attributes with all split_on columns is required for every "
                    f"model when the Evaluator was initialised with splits: {details}"
                )

        # Check every model's pids up front, before any (slow) model runs;
        # compare_population repeats this for direct callers.
        if synthetic_attributes is not None:
            for model, schedule in synthetic_schedules.items():
                attrs = synthetic_attributes.get(model)
                if attrs is not None and "pid" in attrs.columns:
                    _check_pids_match(
                        _coerce_to_pandas(schedule),
                        _coerce_to_pandas(attrs),
                        f"model '{model}'",
                    )

        if not self._progress:
            if not self._precomputed:
                self._precompute_target()
            for model, schedule in synthetic_schedules.items():
                attrs = (
                    synthetic_attributes[model]
                    if synthetic_attributes is not None
                    else None
                )
                self.compare_population(
                    model=model, schedule=schedule, attributes=attrs, verbose=verbose
                )
            return self.report()

        # Create all bars upfront (one feature/splits pair per unit) so the
        # user sees the full scope of work immediately.
        n_density = len(self._jobs.density)
        n_splits = len(self._slices)
        models = list(synthetic_schedules.keys())

        units: list[tuple[str, str]] = [] if self._precomputed else [("target", "cyan")]
        units += [(model, "green") for model in models]

        desc_width = max(len(f"{name} [features]") for name, _ in units)
        bar_pairs = [
            (
                _make_bar(f"{name} [features]", n_density, 2 * i, desc_width, colour),
                _make_bar(f"{name} [splits]", n_splits, 2 * i + 1, desc_width, colour),
            )
            for i, (name, colour) in enumerate(units)
        ]

        remaining_bars = iter(bar_pairs)
        if not self._precomputed:
            self._precompute_target(*next(remaining_bars))

        for model, schedule in synthetic_schedules.items():
            attrs = (
                synthetic_attributes[model]
                if synthetic_attributes is not None
                else None
            )
            self._active_feature_bar, self._active_splits_bar = next(remaining_bars)
            self.compare_population(
                model=model,
                schedule=schedule,
                attributes=attrs,
                verbose=verbose,
            )
        self._active_feature_bar = None
        self._active_splits_bar = None

        for feature_bar, splits_bar in bar_pairs:
            feature_bar.close()
            splits_bar.close()

        return self.report()

    def compare_population(
        self,
        model: str,
        schedule: DataFrame,
        attributes: DataFrame | None = None,
        verbose: bool = False,
    ) -> None:
        """Compute description and distance columns for a single synthetic population.

        Advanced/low-level: for one-model-at-a-time accumulation. Most users
        want ``compare()`` or ``Evaluator.compare()``.

        Results are stored on ``self.collected_description_values[model]``
        (and the parallel ``_weights``/``collected_distance_*`` dicts).  Call
        ``report()`` after all models have been compared to assemble the
        final ``EvalResult``.

        Args:
            model: Model name.
            schedule: Schedules DataFrame for this model.
            attributes: Attributes DataFrame for this model (with ``pid`` column).
                Required when the evaluator was initialised with splits; omit
                (or pass ``None``) when no splits are in use.
            verbose: Print progress.
        """
        _check_model_names([model])
        if not self._precomputed:
            self._precompute_target()

        feature_bar = self._active_feature_bar
        splits_bar = self._active_splits_bar
        schedule = _coerce_to_pandas(schedule)
        if attributes is not None:
            attributes = _coerce_to_pandas(attributes)
        if attributes is None:
            if self._uses_real_splits:
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
            _check_pids_match(schedule, attributes, f"model '{model}'")
        attributes = self._apply_numeric_bins(attributes)
        pop = Population(schedule)

        pid_features = {}
        with _bar_scope(
            feature_bar, f"{model} [features]", len(self._jobs.density), self._progress
        ) as feature_bar:
            for spec in self._jobs.density:
                _bar_set_item(feature_bar, spec.name)
                pid_features[(spec.domain, spec.name)] = spec.feature_fn(pop)
                feature_bar.update(1)

        desc_value_parts: list[Series] = []
        desc_weight_parts: list[Series] = []
        dist_value_parts: list[Series] = []
        dist_weight_parts: list[Series] = []

        # Pre-compute full-population features once (mirrors _precompute_target
        # Phase 1).  Creativity hashes and structural feasibility flags are
        # computed here for the entire synthetic population; the split loop
        # below only subsets them.
        _needs_hashes = (
            self._jobs.creativity.enabled or self._jobs.structural.needs_novel_pids
        )
        if _needs_hashes:
            pid_hashes = creativity.hash_per_pid(pop)
        if self._jobs.structural.enabled:
            feasibility_flags = structural.feasibility(pop)

        # Iterate over (split, category) combinations and subset the
        # pre-computed features down to the relevant pids.
        with _bar_scope(
            splits_bar, f"{model} [splits]", len(self._split_cat_info), self._progress
        ) as splits_bar:
            synth_groups = {
                attrs: group_pids(attributes, attrs)
                for attrs in dict.fromkeys(s.attrs for s, *_ in self._split_cat_info)
            }
            for slice_, _, cached_subset, n_target in self._split_cat_info:
                split, cat = slice_.key
                _bar_set_item(
                    splits_bar, cat if split == "__split__" else f"{split}={cat}"
                )
                sample_pids = synth_groups[slice_.attrs].get(
                    slice_.values, np.array([], dtype=attributes.pid.dtype)
                )
                synth_dense_pids = pop.dense_pids_from_original(sample_pids)
                # Model distance weights are rescaled to the target's slice size
                # (below), so the two sides' weights are comparable whatever
                # the sample sizes; see ResultFrame.aggregate_distances.
                n_model = len(synth_dense_pids)
                weight_scale = n_target / n_model if n_model else 0.0
                slice_dist_start = len(dist_weight_parts)

                if self._jobs.creativity.enabled:
                    dv, dw, xv, xw = _model_cols_creativity(
                        model,
                        pid_hashes,
                        sample_pids,
                        self._obs_hashes[(split, cat)],
                        self._jobs.creativity,
                    )
                    desc_value_parts.append(_append_split_cat_index(dv, split, cat))
                    desc_weight_parts.append(_append_split_cat_index(dw, split, cat))
                    dist_value_parts.append(_append_split_cat_index(xv, split, cat))
                    dist_weight_parts.append(_append_split_cat_index(xw, split, cat))

                if self._jobs.structural.enabled:
                    novel_dense_pids = None
                    if self._jobs.structural.needs_novel_pids:
                        obs_hash = self._obs_hashes[(split, cat)]
                        novel_pids = np.array(
                            [
                                p
                                for p in sample_pids
                                if pid_hashes.get(p) not in obs_hash
                            ]
                        )
                        novel_dense_pids = pop.dense_pids_from_original(novel_pids)
                    sv, sw = _model_cols_structural(
                        model,
                        feasibility_flags,
                        synth_dense_pids,
                        novel_dense_pids,
                        self._jobs.structural,
                    )
                    sv = _append_split_cat_index(sv.copy(), split, cat)
                    sw = _append_split_cat_index(sw.copy(), split, cat)
                    desc_value_parts.append(sv)
                    desc_weight_parts.append(sw)
                    dist_value_parts.append(sv)
                    dist_weight_parts.append(sw)

                for spec in self._jobs.density:
                    key = (spec.domain, spec.name)
                    obs_feat = cached_subset[key]
                    # default holds the observed distribution shape; used as a
                    # fallback when the synthetic model has no data for a segment.
                    default = _make_default(obs_feat)

                    # Aggregate pre-computed per-pid features for just the pids
                    # in this slice, dropping segments missing from it (see
                    # PidFeatures.aggregate_present). A segment missing on both
                    # sides gets no row; one missing on one side is scored by
                    # _score_features' missing-segment rules.
                    synth_pf = pid_features[key]
                    synth_feat = synth_pf.aggregate_present(synth_dense_pids)

                    # w = weights, d = descriptive values, s = distance scores
                    w, d, s = _model_contribution(
                        model,
                        spec,
                        obs_feat,
                        synth_feat,
                        default,
                        synth_pf.zero_is_missing,
                    )
                    dist_weight = w.reindex(s.index, fill_value=0)
                    w = _tag_density_index(w, spec.domain, spec.name, split, cat)
                    d = _tag_density_index(d, spec.domain, spec.name, split, cat)
                    s = _tag_density_index(s, spec.domain, spec.name, split, cat)
                    dist_weight = _tag_density_index(
                        dist_weight, spec.domain, spec.name, split, cat
                    )
                    desc_value_parts.append(d)
                    desc_weight_parts.append(w)
                    dist_value_parts.append(s)
                    dist_weight_parts.append(dist_weight)

                for i in range(slice_dist_start, len(dist_weight_parts)):
                    dist_weight_parts[i] = dist_weight_parts[i] * weight_scale

                splits_bar.update(1)

        # Store results so report() can later concat them alongside the target's
        # own values/weights.
        self.collected_description_values[model] = _concat_tagged(
            [p for p in desc_value_parts if not p.empty]
        )
        self.collected_description_weights[model] = _concat_tagged(
            [p for p in desc_weight_parts if not p.empty]
        )
        self.collected_distance_values[model] = _concat_tagged(
            [p for p in dist_value_parts if not p.empty]
        )
        self.collected_distance_weights[model] = _concat_tagged(
            [p for p in dist_weight_parts if not p.empty]
        )

    def report(self) -> EvalResult:
        """Assemble an ``EvalResult`` from previously accumulated model comparisons.

        Call this after one or more ``compare_population`` calls.

        Returns:
            EvalResult wrapping the raw segment-level data.  Use
            ``result.at(level, split)`` or the named properties
            (``.features``/``.groups``/``.domains``) to get the three-tier
            aggregated output.
        """
        desc_values = concat(
            {_TARGET_COL: self._target_description_values}
            | self.collected_description_values,
            axis=1,
        )
        # Weights are counts: a row one side doesn't cover (e.g. a model uses
        # an activity absent from the target) means zero observations there,
        # not an unknown value — fillna(0.0) so aggregate()/aggregate_distances()
        # treat it as real zero-weight rather than NaN propagating through the
        # weighted-average arithmetic.
        desc_weights = concat(
            {_TARGET_COL: self._target_description_weights}
            | self.collected_description_weights,
            axis=1,
        ).fillna(0.0)
        dist_values = concat(self.collected_distance_values, axis=1)
        dist_weights = concat(self.collected_distance_weights, axis=1).fillna(0.0)
        # The concatenated values DataFrames may have more rows than the
        # target-only Series below (e.g. a model uses an activity absent from
        # the target) — reindex so every attached Series matches the final row
        # set. Units become NaN for such rows (informational only); the
        # weight becomes 0 (a real zero-weight row, not a missing one).
        descriptions = ResultFrame(
            values=desc_values,
            weights=desc_weights,
            units=self._target_description_units.reindex(desc_values.index),
        )
        distances = ResultFrame(
            values=dist_values,
            weights=dist_weights,
            units=self._target_distance_units.reindex(dist_values.index),
        )
        return EvalResult(
            descriptions=descriptions,
            distances=distances,
            target_distance_weights=self._target_distance_weights.reindex(
                dist_values.index, fill_value=0.0
            ),
            slice_assignments={s.key: s.assignment for s, _ in self._slices},
            split_order=self._split_order,
        )


def compare(
    target_schedules: DataFrame,
    synthetic_schedules,
    target_attributes: DataFrame | None = None,
    synthetic_attributes: dict[str, DataFrame] | None = None,
    split_on: list[str] | None = None,
    verbose: bool = False,
    config_path=None,
    disable: list[str] | None = None,
    progress: bool = False,
    split_order: int | None = None,
    min_split_size: int | None = None,
) -> EvalResult:
    """Compare observed and synthetic activity schedule populations.

    This is the primary entry point, for both one-off and split-based
    comparisons. For repeated comparisons against the same observed data,
    use ``Evaluator`` directly so observed features are computed once.

    Args:
        target_schedules: Observed schedules with columns pid, act, start, end, duration.
        synthetic_schedules: Single synthetic DataFrame or dict mapping model names to DataFrames.
        target_attributes: Optional attributes DataFrame for ``observed``, with a
                    ``pid`` column.  Required together with ``split_on``.
        synthetic_attributes: Optional ``{model_name: attributes_df}`` with ``pid`` column.
            If provided, enables attribute-based splitting (exposes ``label_*`` frames).
        split_on: Optional attribute column(s) to split evaluation by (e.g.
            ``["gender"]``).  Requires ``target_attributes`` and ``attributes``.
        verbose: Print progress for each (split, category) subset.
        config_path: Optional path to a custom ``config.toml``; defaults to the
            packaged config.
        disable: Optional dotted ``section.key`` config paths to switch off,
            e.g. ``["jobs.creativity.novelty", "jobs.transitions.4-gram"]`` —
            see ``config.toml`` for the full list of keys. Applied on top of
            whatever ``config_path`` loads; sugar for the common case of
            disabling one or two metrics without writing a config file.
        progress: Show tqdm progress bars while computing features. Useful for
            large populations; pass ``Evaluator(progress=True)`` directly
            instead if you're also making repeated ``compare()`` calls.
        split_order: Number of ``split_on`` attributes combined per slice,
            e.g. ``2`` scores every ``gender & income`` combination. Only this
            order is computed; ``result.at(level, split, order=k)`` pools it
            back down to lower orders. Defaults to the config's
            ``[splits] order`` (1).
        min_split_size: Skip slices with fewer target persons than this.
            Defaults to the config's ``[splits] min_size``, else no minimum at
            order 1 and 30 above.

    Returns:
        EvalResult with raw segment-level data; use ``result.at(...)`` or the
        named properties for the aggregated output.
    """
    if _is_dataframe(synthetic_schedules):
        synthetic_schedules = {"synthetic": synthetic_schedules}
    evaluator = Evaluator(
        target_schedules,
        target_attributes=target_attributes,
        split_on=split_on,
        config_path=config_path,
        disable=disable,
        progress=progress,
        split_order=split_order,
        min_split_size=min_split_size,
    )
    return evaluator.compare(
        synthetic_schedules, attributes=synthetic_attributes, verbose=verbose
    )


def compare_grid(
    schedules_a: dict[str, DataFrame],
    schedules_b: dict[str, DataFrame],
    attributes_a: dict[str, DataFrame] | None = None,
    attributes_b: dict[str, DataFrame] | None = None,
    split_on: list[str] | None = None,
    **kwargs,
) -> EvalResult:
    """Compare the same synthetic models against each of several targets.

    Runs one ordinary `compare()` call per target, then `combine()`s the
    results (see `acteval.results.combine`) into a single `EvalResult` with
    model columns named `"{target_name}::{model_name}"`.

    Args:
        schedules_a: ``{name: schedules_df}``.
        schedules_b: ``{name: schedules_df}``, compared against every schedules in `schedules_a`.
        attributes_a: Optional ``{name: attributes_df}``.
        attributes_b: Optional ``{name: attributes_df}``.
        split_on: Optional attribute column(s) to split each target's evaluation by.
        **kwargs: Passed through to ``compare()`` (e.g. ``config_path``,
            ``disable``, ``progress``).

    Returns:
        A single combined ``EvalResult``; see ``acteval.results.combine`` for
        details and the known target-weight-base limitation.
    """
    results = {
        name: compare(
            target_schedules=schedules,
            synthetic_schedules=schedules_b,
            target_attributes=(attributes_a.get(name) if attributes_a else None),
            synthetic_attributes=attributes_b,
            split_on=split_on,
            **kwargs,
        )
        for name, schedules in schedules_a.items()
    }
    return combine(results)


def compare_many(
    schedules_a: dict[str, DataFrame],
    schedules_b: dict[str, DataFrame],
    attributes_a: dict[str, DataFrame] | None = None,
    attributes_b: dict[str, DataFrame] | None = None,
    split_on: list[str] | None = None,
    **kwargs,
) -> EvalResult:
    """Compare each target against its corresponding synthetic model, pairwise.

    ``schedules_a`` and ``schedules_b`` are paired up positionally (first with
    first, second with second, ...) — unlike `compare_grid`, which compares
    every target against every model. Runs one ordinary `compare()` call per
    pair, then `combine()`s the results (see `acteval.results.combine`) into a
    single `EvalResult` with model columns named `"{name_a}::{name_b}"`.

    Args:
        schedules_a: ``{name: schedules_df}``.
        schedules_b: ``{name: schedules_df}``, compared pairwise against schedules in `schedules_a`
            (paired by position — both dicts must have the same length).
        attributes_a: Optional ``{name: attributes_df}``.
        attributes_b: Optional ``{name: attributes_df}``.
        split_on: Optional attribute column(s) to split each pair's evaluation by.
        **kwargs: Passed through to ``compare()`` (e.g. ``config_path``,
            ``disable``, ``progress``).

    Returns:
        A single combined ``EvalResult``; see ``acteval.results.combine`` for
        details and the known target-weight-base limitation.
    """
    if len(schedules_a) != len(schedules_b):
        raise ValueError(
            "schedules_a and schedules_b must have the same length for a "
            f"pairwise comparison; got {len(schedules_a)} and {len(schedules_b)}"
        )
    results = {
        name_a: compare(
            target_schedules=a,
            synthetic_schedules={name_b: b},
            target_attributes=(attributes_a.get(name_a) if attributes_a else None),
            synthetic_attributes=(
                {name_b: attributes_b[name_b]} if attributes_b else None
            ),
            split_on=split_on,
            **kwargs,
        )
        for (name_a, a), (name_b, b) in zip(
            schedules_a.items(), schedules_b.items(), strict=True
        )
    }
    return combine(results)
