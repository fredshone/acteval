import warnings
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series, concat

from acteval._aggregation import DEFAULT_REMOVE_FEATURES, DEFAULT_REMOVE_GROUPS
from acteval._compat import _coerce_to_pandas, _is_dataframe
from acteval._jobs import EvalConfig, get_jobs
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
from acteval._progress import bar_scope as _bar_scope
from acteval._progress import bar_set_item as _bar_set_item
from acteval._progress import make_bar as _make_bar
from acteval._result_frame import ResultFrame
from acteval._splits import _key_activities
from acteval.features import creativity, structural
from acteval.population import Population


def _append_split_cat_index(
    data: DataFrame | Series, split: str, cat
) -> DataFrame | Series:
    """Append ``(split, cat)`` as trailing ``label``/``cat`` MultiIndex levels.

    Shared by creativity and structural rows, whose index already has its
    final shape (domain, feature, segment, ...) before this split/category
    tag is added. Works on a ``DataFrame`` or a ``Series``.
    """
    data.index = MultiIndex.from_tuples(
        [(*i, split, cat) for i in data.index],
        names=list(data.index.names) + ["label", "cat"],
    )
    return data


def _tag_density_index(
    data: DataFrame | Series, domain: str, feature: str, split: str, cat
) -> DataFrame | Series:
    """Turn a flat segment index into (domain, feature, segment, label, cat).

    Density rows start with only a flat ``segment`` index (unlike creativity/
    structural, which already carry ``domain``/``feature``/``segment``), so
    this injects the two leading levels alongside the split/category tag.
    Works on a ``DataFrame`` or a ``Series``.
    """
    data.index = MultiIndex.from_tuples(
        [(domain, feature, f, split, cat) for f in data.index],
        names=["domain", "feature", "segment", "label", "cat"],
    )
    return data


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
        descriptions: ResultFrame,
        distances: ResultFrame,
        target_distance_weights: Series,
        schedule: Literal["features", "groups", "domains"],
        has_splits: bool,
        drop_features,
        drop_groups,
    ):
        self._descriptions = descriptions
        self._distances = distances
        self._target_distance_weights = target_distance_weights
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
                self._descriptions,
                self._distances,
                self._target_distance_weights,
                extra=extra,
            )
        elif self._schedule == "groups":
            desc_rf = descriptions_to_group_level(
                self._descriptions, extra=extra, drop=self._drop_features
            )
            dist_rf = distances_to_group_level(
                self._distances,
                self._target_distance_weights,
                extra=extra,
                drop=self._drop_features,
            )
            desc = desc_rf.values.copy()
            if desc_rf.units is not None:
                desc["unit"] = desc_rf.units
            dist = dist_rf.values.copy()
            if dist_rf.units is not None:
                dist["unit"] = dist_rf.units
        else:  # domains
            group_desc_rf = descriptions_to_group_level(
                self._descriptions, extra=extra, drop=self._drop_features
            )
            group_dist_rf = distances_to_group_level(
                self._distances,
                self._target_distance_weights,
                extra=extra,
                drop=self._drop_features,
            )
            desc = descriptions_to_domain_level(
                group_desc_rf, extra=extra, drop=self._drop_groups
            ).values
            dist = distances_to_domain_level(
                group_dist_rf, extra=extra, drop=self._drop_groups
            ).values
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

    def __init__(
        self,
        descriptions: ResultFrame,
        distances: ResultFrame,
        target_distance_weights: Series,
    ):
        # descriptions: values/weights columns are ["target"] + model_names.
        # distances: values/weights columns are model_names only — there's no
        # such thing as the target's distance to itself.
        self._descriptions = descriptions
        self._distances = distances
        # Raw per-row target weight, blended into each model's own weight
        # before aggregating distances (see ResultFrame.aggregate_distances).
        self._target_distance_weights = target_distance_weights

    # --- raw access ---

    @property
    def raw(self) -> dict[str, ResultFrame]:
        """Pre-aggregation data as ``ResultFrame`` objects (desc + dist).

        ``descriptions`` includes the target's own value/weight as its
        ``"target"`` column; ``distances`` covers models only. Pair with
        ``target_distance_weights`` if you need to replicate
        ``ResultFrame.aggregate_distances``'s weight blending yourself.
        """
        return {"descriptions": self._descriptions, "distances": self._distances}

    @property
    def target_distance_weights(self) -> Series:
        """Raw per-row target weight used to blend into each model's own
        weight before aggregating distances."""
        return self._target_distance_weights

    # --- split availability ---

    @property
    def has_splits(self) -> bool:
        """True when the ``Evaluator`` was run with ``split_on``."""
        return not (
            self._descriptions.values.index.get_level_values("label").unique().tolist()
            == ["__split__"]
        )

    # --- schedule-level accessors ---

    @cached_property
    def features(self) -> ScheduleView:
        """Feature-level view: index ``(domain, feature, segment[, ...])``.

        Most granular schedule level; useful for disk storage.
        """
        return ScheduleView(
            self._descriptions,
            self._distances,
            self._target_distance_weights,
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
            self._descriptions,
            self._distances,
            self._target_distance_weights,
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
            self._descriptions,
            self._distances,
            self._target_distance_weights,
            schedule="domains",
            has_splits=self.has_splits,
            drop_features=DEFAULT_REMOVE_FEATURES,
            drop_groups=DEFAULT_REMOVE_GROUPS,
        )

    # --- flexible accessor ---

    _LEVELS = ("features", "groups", "domains")
    _SPLITS = ("combined", "by_attribute", "by_category")

    def at(self, level: str = "domains", split: str = "combined") -> AggregatedResult:
        """Get an ``AggregatedResult`` at the given level and split.

        The one thing to remember for anything beyond ``summary()`` /
        ``rank_models()`` / ``best_model``: equivalent to chaining the
        ``.features``/``.groups``/``.domains`` and
        ``.combined``/``.by_attribute``/``.by_category`` properties, e.g.
        ``result.at("groups", "by_attribute")`` is ``result.groups.by_attribute``.

        Args:
            level: One of "features", "groups", "domains".
            split: One of "combined", "by_attribute", "by_category".

        Returns:
            The requested ``AggregatedResult``.

        Raises:
            ValueError: If ``level`` or ``split`` is not one of the allowed values.
        """
        if level not in self._LEVELS:
            raise ValueError(f"level must be one of {self._LEVELS}, got {level!r}")
        if split not in self._SPLITS:
            raise ValueError(f"split must be one of {self._SPLITS}, got {split!r}")
        return getattr(getattr(self, level), split)

    # --- model introspection ---

    @property
    def model_names(self) -> list[str]:
        """Model column names."""
        return list(self._distances.values.columns)

    def summary(self) -> DataFrame:
        """Domain-level distances for each model."""
        return self.domains.combined.distances

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
        jobs: EvalConfig | None = None,
        progress: bool = False,
        disable: list[str] | None = None,
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
        self._jobs: EvalConfig = (
            jobs if jobs is not None else get_jobs(config_path, disable)
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
                # for each (domain, name) key — these become the "target" side
                # of every distance calculation.
                cached_subset = {
                    key: pf.subset(target_dense_pids).aggregate()
                    for key, pf in self._target_pid_features.items()
                }
                self._split_cat_info.append((split, cat, sub_target, cached_subset))

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
            for split, cat, sub_target, cached_subset in self._split_cat_info:
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
                    value, weight, _ = _observed_base(spec, obs_feat)
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
        self._target_description_values = concat(desc_value_parts)
        self._target_description_weights = concat(desc_weight_parts)
        self._target_description_units = concat(desc_unit_parts)
        self._target_distance_weights = concat(dist_weight_parts)
        self._target_distance_units = concat(dist_unit_parts)
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
    ) -> "EvalResult":
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
    ) -> "EvalResult":
        """Shared implementation behind ``Evaluator.compare``."""
        self.collected_description_values = {}
        self.collected_description_weights = {}
        self.collected_distance_values = {}
        self.collected_distance_weights = {}

        uses_real_splits = self._split_on != ["__split__"]
        if uses_real_splits:
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
        n_splits = sum(len(self._target_attributes[s].unique()) for s in self._split_on)
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
        if not self._precomputed:
            self._precompute_target()

        feature_bar = self._active_feature_bar
        splits_bar = self._active_splits_bar
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
            for split, cat, _, cached_subset in self._split_cat_info:
                _bar_set_item(
                    splits_bar, cat if split == "__split__" else f"{split}={cat}"
                )
                sample_pids = attributes[attributes[split] == cat].pid.values
                synth_dense_pids = pop.dense_pids_from_original(sample_pids)
                # Used below to skip density segments whose key activity is absent
                # from this synthetic sub-population entirely.
                synth_sub_acts = frozenset(
                    schedule.loc[schedule.pid.isin(sample_pids), "act"].unique()
                )

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

                splits_bar.update(1)

        # Store results so report() can later concat them alongside the target's
        # own values/weights.
        self.collected_description_values[model] = concat(
            [p for p in desc_value_parts if not p.empty]
        )
        self.collected_description_weights[model] = concat(
            [p for p in desc_weight_parts if not p.empty]
        )
        self.collected_distance_values[model] = concat(
            [p for p in dist_value_parts if not p.empty]
        )
        self.collected_distance_weights[model] = concat(
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
            {"target": self._target_description_values}
            | self.collected_description_values,
            axis=1,
        )
        # Weights are counts: a row one side doesn't cover (e.g. a model uses
        # an activity absent from the target) means zero observations there,
        # not an unknown value — fillna(0.0) so aggregate()/aggregate_distances()
        # treat it as real zero-weight rather than NaN propagating through the
        # weighted-average arithmetic.
        desc_weights = concat(
            {"target": self._target_description_weights}
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
        )


def compare(
    observed: DataFrame,
    synthetic,
    attributes: dict[str, DataFrame] | None = None,
    target_attributes: DataFrame | None = None,
    split_on: list[str] | None = None,
    verbose: bool = False,
    disable: list[str] | None = None,
    progress: bool = False,
) -> EvalResult:
    """Compare observed and synthetic activity schedule populations.

    This is the primary entry point, for both one-off and split-based
    comparisons. For repeated comparisons against the same observed data,
    use ``Evaluator`` directly so observed features are computed once.

    Args:
        observed: Observed schedules with columns pid, act, start, end, duration.
        synthetic: Single synthetic DataFrame or dict mapping model names to DataFrames.
        attributes: Optional ``{model_name: attributes_df}`` with ``pid`` column.
            If provided, enables attribute-based splitting (exposes ``label_*`` frames).
        target_attributes: Optional attributes DataFrame for ``observed``, with a
            ``pid`` column.  Required together with ``split_on``.
        split_on: Optional attribute column(s) to split evaluation by (e.g.
            ``["gender"]``).  Requires ``target_attributes`` and ``attributes``.
        verbose: Print progress for each (split, category) subset.
        disable: Optional dotted ``section.key`` config paths to switch off,
            e.g. ``["jobs.creativity.novelty", "jobs.transitions.4-gram"]`` —
            see ``config.toml`` for the full list of keys. Sugar for the common
            case of disabling one or two metrics without writing a config file;
            pass ``config_path``/``jobs`` via ``Evaluator`` directly for
            anything more involved.
        progress: Show tqdm progress bars while computing features. Useful for
            large populations; pass ``Evaluator(progress=True)`` directly
            instead if you're also making repeated ``compare()`` calls.

    Returns:
        EvalResult with raw segment-level data; use ``result.at(...)`` or the
        named properties for the aggregated output.
    """
    if _is_dataframe(synthetic):
        synthetic = {"synthetic": synthetic}
    evaluator = Evaluator(
        observed,
        target_attributes=target_attributes,
        split_on=split_on,
        disable=disable,
        progress=progress,
    )
    return evaluator.compare(synthetic, attributes=attributes, verbose=verbose)
