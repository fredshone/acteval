"""Reading and combining evaluation results.

Holds the "result" side of the API: `EvalResult` (and the `ScheduleView` /
`AggregatedResult` / `SplitNotAvailableError` classes it's built from), the
index-tagging helpers `Evaluator` uses while assembling one, and `combine()`
for merging several `EvalResult`s (e.g. one per target — see
`Evaluator.compare_many()` in `evaluate.py`) into one.

The comparison entry points themselves (`Evaluator`, `compare()`,
`compare_many()`) live in `evaluate.py`.
"""

import warnings
from functools import cached_property
from pathlib import Path
from typing import Literal

from pandas import DataFrame, MultiIndex, Series, concat

from acteval._aggregation import (
    DEFAULT_REMOVE_FEATURES,
    DEFAULT_REMOVE_GROUPS,
    descriptions_to_domain_level,
    descriptions_to_group_level,
    distances_to_domain_level,
    distances_to_group_level,
)
from acteval._pipeline import _aggregate_features
from acteval._result_frame import ResultFrame


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


def _renamed(df: DataFrame, name: str, columns: list[str]) -> DataFrame:
    """Select `columns` from df, each renamed with a `"{name}::"` prefix."""
    return df[columns].rename(columns={c: f"{name}::{c}" for c in columns})


def combine(results: dict[str, EvalResult]) -> EvalResult:
    """Combine EvalResults from separate compare() calls into one.

    Typical use: comparing the same synthetic models against several targets
    (see `Evaluator.compare_many`/`compare_many` in `evaluate.py`). Every
    model column is renamed `"{source}::{model}"` so models from different
    sources stay distinct in `model_names` / `summary()` / `rank_models()`.

    The target's own values/weights are taken from the *first* result only.
    Every model's distances were already computed against its own source's
    target, so this only affects which weights are used when re-aggregating
    a non-first source's columns from features to groups to domains — a
    known simplification for now, not an error in the distances themselves.
    A future refactor could carry one target per source to remove this
    limitation.

    Args:
        results: `{source_name: EvalResult}`, e.g. one entry per target.

    Returns:
        A single `EvalResult` with every model column namespaced by its source.

    Raises:
        ValueError: If `results` is empty, or inputs mix results that used
            split_on with results that didn't.
        TypeError: If a value in `results` is not an `EvalResult`.
    """
    if not results:
        raise ValueError("combine() requires at least one result")
    for name, r in results.items():
        if not isinstance(r, EvalResult):
            raise TypeError(f"combine(): {name!r} is not an EvalResult")

    names = list(results)
    has_splits = {name: r.has_splits for name, r in results.items()}
    if len(set(has_splits.values())) > 1:
        raise ValueError(
            "combine(): cannot mix results with and without split_on "
            f"({has_splits}); align split_on across all inputs before combining"
        )

    base = results[names[0]]

    mismatched = [
        name
        for name, r in results.items()
        if not r.raw["descriptions"].values.index.equals(
            base.raw["descriptions"].values.index
        )
        or not r.raw["distances"].values.index.equals(
            base.raw["distances"].values.index
        )
    ]
    if mismatched:
        warnings.warn(
            f"combine(): row index differs from {names[0]!r} for source(s) "
            f"{mismatched} (e.g. different activities present in different "
            "targets). Aggregation weighting for non-first sources is "
            "approximate — see combine()'s docstring.",
            UserWarning,
            stacklevel=2,
        )

    desc_values = concat(
        [base.raw["descriptions"].values[["target"]]]
        + [
            _renamed(r.raw["descriptions"].values, n, r.model_names)
            for n, r in results.items()
        ],
        axis=1,
    )
    # Weights are counts: a row one source doesn't cover means zero
    # observations there, not an unknown value — fillna(0.0) so
    # aggregate()/aggregate_distances() treat it as real zero-weight rather
    # than NaN propagating through the weighted-average arithmetic.
    desc_weights = concat(
        [base.raw["descriptions"].weights[["target"]]]
        + [
            _renamed(r.raw["descriptions"].weights, n, r.model_names)
            for n, r in results.items()
        ],
        axis=1,
    ).fillna(0.0)
    dist_values = concat(
        [
            _renamed(r.raw["distances"].values, n, r.model_names)
            for n, r in results.items()
        ],
        axis=1,
    )
    dist_weights = concat(
        [
            _renamed(r.raw["distances"].weights, n, r.model_names)
            for n, r in results.items()
        ],
        axis=1,
    ).fillna(0.0)

    # Sources may cover different rows (e.g. different activities present in
    # different targets) — reindex base's units to the merged row set so they
    # line up positionally with values/weights. Units become NaN for such
    # rows (informational only); target_distance_weights becomes 0 (a real
    # zero-weight row, not a missing one).
    descriptions = ResultFrame(
        values=desc_values,
        weights=desc_weights,
        units=base.raw["descriptions"].units.reindex(desc_values.index),
    )
    distances = ResultFrame(
        values=dist_values,
        weights=dist_weights,
        units=base.raw["distances"].units.reindex(dist_values.index),
    )
    return EvalResult(
        descriptions=descriptions,
        distances=distances,
        target_distance_weights=base.target_distance_weights.reindex(
            dist_values.index, fill_value=0.0
        ),
    )
