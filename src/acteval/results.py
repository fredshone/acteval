"""Reading and combining evaluation results.

Holds the "result" side of the API: `EvalResult` (and the `ScheduleView` /
`AggregatedResult` / `SplitNotAvailableError` classes it's built from), the
index-tagging helpers `Evaluator` uses while assembling one, and `combine()`
for merging several `EvalResult`s (e.g. one per target — see
`compare_grid()`/`compare_many()` in `evaluate.py`) into one.

The comparison entry points themselves (`Evaluator`, `compare()`,
`compare_grid()`, `compare_many()`) live in `evaluate.py`.
"""

import itertools
import warnings
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, concat

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
from acteval._splits import Slice

# ``{(label, cat): ((attr, value), ...)}`` — the attribute assignment behind
# each slice's index strings, so views never have to parse joined strings.
SliceAssignments = dict[tuple, tuple[tuple[str, object], ...]]

_INDEX_NAMES = ["domain", "feature", "segment", "label", "cat"]


def _flat_tuple_index(tuples: list[tuple]) -> Index:
    # A flat object Index of tuples (not a MultiIndex): building and
    # concatenating hundreds of small MultiIndexes per split dominated runtime,
    # so parts stay flat until ``_concat_tagged`` builds one MultiIndex.
    return Index(tuples, tupleize_cols=False, dtype=object)


def _append_split_cat_index(
    data: DataFrame | Series, split: str, cat
) -> DataFrame | Series:
    """Append ``(split, cat)`` as trailing ``label``/``cat`` index entries.

    Shared by creativity and structural rows, whose index already has its
    final shape (domain, feature, segment, ...) before this split/category
    tag is added. Works on a ``DataFrame`` or a ``Series``. The result has a
    flat tuple index; combine parts with ``_concat_tagged``.
    """
    data.index = _flat_tuple_index([(*i, split, cat) for i in data.index])
    return data


def _tag_density_index(
    data: DataFrame | Series, domain: str, feature: str, split: str, cat
) -> DataFrame | Series:
    """Turn a flat segment index into (domain, feature, segment, label, cat).

    Density rows start with only a flat ``segment`` index (unlike creativity/
    structural, which already carry ``domain``/``feature``/``segment``), so
    this injects the two leading levels alongside the split/category tag.
    Works on a ``DataFrame`` or a ``Series``. The result has a flat tuple
    index; combine parts with ``_concat_tagged``.
    """
    data.index = _flat_tuple_index(
        [(domain, feature, f, split, cat) for f in data.index]
    )
    return data


def _concat_tagged(parts: list[Series]) -> Series:
    """Concatenate tagged parts and convert their flat tuple index into the
    ``(domain, feature, segment, label, cat)`` MultiIndex."""
    result = concat(parts)
    result.index = MultiIndex.from_tuples(list(result.index), names=_INDEX_NAMES)
    return result


def _sub_keys(assignment: tuple, order: int) -> list[tuple]:
    """``(label, cat)`` of every order-``order`` sub-assignment of *assignment*."""
    return [
        Slice(attrs=tuple(a for a, _ in sub), values=tuple(v for _, v in sub)).key
        for sub in itertools.combinations(assignment, order)
    ]


def _expansion_positions(
    index: MultiIndex, assignments: SliceAssignments, order: int
) -> tuple[np.ndarray, MultiIndex]:
    """Map each row to every order-``order`` slice its own slice contains.

    Returns ``(take, new_index)``: row positions to take (each source row is
    repeated once per sub-key) and the matching index with ``label``/``cat``
    replaced by the sub-key. Rows whose ``(label, cat)`` has no assignment are
    dropped.
    """
    labels = index.get_level_values("label")
    cats = index.get_level_values("cat")
    sub_keys_by_key = {
        key: _sub_keys(assignment, order) for key, assignment in assignments.items()
    }
    take: list[int] = []
    new_labels: list = []
    new_cats: list = []
    for pos, key in enumerate(zip(labels, cats)):
        for sub_label, sub_cat in sub_keys_by_key.get(key, ()):
            take.append(pos)
            new_labels.append(sub_label)
            new_cats.append(sub_cat)
    take = np.asarray(take, dtype=np.int64)
    new_index = MultiIndex.from_arrays(
        [index.get_level_values(n)[take] for n in ("domain", "feature", "segment")]
        + [new_labels, new_cats],
        names=_INDEX_NAMES,
    )
    return take, new_index


def _expand_to_order(
    rf: ResultFrame, assignments: SliceAssignments, order: int
) -> tuple[ResultFrame, np.ndarray, MultiIndex]:
    """Re-key raw rows from their computed slices to order-``order`` slices.

    A row from slice ``gender=male & income=high`` contributes to both
    ``gender=male`` and ``income=high`` at ``order=1``; aggregating the
    expanded rows then pools every computed slice containing each sub-slice.
    Returns the expanded ``ResultFrame`` plus ``(take, new_index)`` so the
    caller can expand aligned Series the same way.
    """
    take, new_index = _expansion_positions(rf.values.index, assignments, order)
    expanded = ResultFrame(
        values=rf.values.iloc[take].set_axis(new_index),
        weights=rf.weights.iloc[take].set_axis(new_index),
        units=rf.units.iloc[take].set_axis(new_index) if rf.units is not None else None,
    )
    return expanded, take, new_index


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
        slice_assignments: SliceAssignments | None = None,
        split_order: int = 1,
    ):
        self._descriptions = descriptions
        self._distances = distances
        self._target_distance_weights = target_distance_weights
        self._schedule = schedule
        self._has_splits = has_splits
        self._drop_features = drop_features
        self._drop_groups = drop_groups
        self._slice_assignments = slice_assignments or {}
        self._split_order = split_order
        self._views: dict[tuple[str, int], AggregatedResult] = {}

    def _compute(self, extra: list[str], order: int | None = None) -> AggregatedResult:
        split_name = (
            "combined"
            if not extra
            else ("by_attribute" if extra == ["label"] else "by_category")
        )
        label = f"{self._schedule} × {split_name}"
        descriptions = self._descriptions
        distances = self._distances
        target_distance_weights = self._target_distance_weights
        if extra and order is not None and order < self._split_order:
            label += f" (order {order})"
            descriptions, _, _ = _expand_to_order(
                descriptions, self._slice_assignments, order
            )
            distances, take, new_index = _expand_to_order(
                distances, self._slice_assignments, order
            )
            # Already aligned to the distances index (see report()/combine()).
            target_distance_weights = target_distance_weights.iloc[take].set_axis(
                new_index
            )
        if self._schedule == "features":
            desc, dist = _aggregate_features(
                descriptions,
                distances,
                target_distance_weights,
                extra=extra,
            )
        elif self._schedule == "groups":
            desc_rf = descriptions_to_group_level(
                descriptions, extra=extra, drop=self._drop_features
            )
            dist_rf = distances_to_group_level(
                distances,
                target_distance_weights,
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
                descriptions, extra=extra, drop=self._drop_features
            )
            group_dist_rf = distances_to_group_level(
                distances,
                target_distance_weights,
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

    @property
    def by_attribute(self) -> AggregatedResult:
        """Aggregated result split by attribute (one row per label value), at
        the computed ``split_order``.

        Raises ``SplitNotAvailableError`` if the parent ``EvalResult`` was
        produced without ``split_on``.
        """
        return self.view("by_attribute")

    @property
    def by_category(self) -> AggregatedResult:
        """Aggregated result split by attribute category (one row per label × cat),
        at the computed ``split_order``.

        Raises ``SplitNotAvailableError`` if the parent ``EvalResult`` was
        produced without ``split_on``.
        """
        return self.view("by_category")

    def view(self, split: str, order: int | None = None) -> AggregatedResult:
        """Aggregated result for *split* at view order *order*.

        Args:
            split: One of "combined", "by_attribute", "by_category".
            order: Only for "by_attribute"/"by_category": the number of
                attributes per row, ``1 <= order <= split_order`` (defaults to
                ``split_order``). Below ``split_order``, each row pools every
                computed slice containing it — e.g. with ``split_order=2``,
                ``order=1``'s ``gender=male`` row pools every
                ``gender=male & *`` slice.

        Raises:
            SplitNotAvailableError: For a split view without ``split_on``.
            ValueError: If ``split`` or ``order`` is out of range.
        """
        if split == "combined":
            return self.combined
        if split not in ("by_attribute", "by_category"):
            raise ValueError(
                "split must be one of ('combined', 'by_attribute', 'by_category'), "
                f"got {split!r}"
            )
        if not self._has_splits:
            raise SplitNotAvailableError(
                f"{split} is not available: this EvalResult was produced without split_on. "
                "Pass target_attributes and split_on to Evaluator to enable split-based views."
            )
        order = self._split_order if order is None else order
        if not 1 <= order <= self._split_order:
            raise ValueError(
                f"order must be between 1 and split_order={self._split_order}, "
                f"got {order}"
            )
        if (split, order) not in self._views:
            extra = ["label"] if split == "by_attribute" else ["label", "cat"]
            self._views[(split, order)] = self._compute(extra, order)
        return self._views[(split, order)]

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
        slice_assignments: SliceAssignments | None = None,
        split_order: int = 1,
    ):
        # descriptions: values/weights columns are ["target"] + model_names.
        # distances: values/weights columns are model_names only — there's no
        # such thing as the target's distance to itself.
        self._descriptions = descriptions
        self._distances = distances
        # Raw per-row target weight, combined with each model's own weight
        # before aggregating distances (see ResultFrame.aggregate_distances).
        self._target_distance_weights = target_distance_weights
        # Attribute assignment behind each (label, cat), for lower-order views.
        self._slice_assignments = slice_assignments or {}
        self._split_order = split_order

    # --- raw access ---

    @property
    def raw(self) -> dict[str, ResultFrame]:
        """Pre-aggregation data as ``ResultFrame`` objects (desc + dist).

        ``descriptions`` includes the target's own value/weight as its
        ``"target"`` column; ``distances`` covers models only, with model
        weights rescaled to the target's size within each slice. Pair with
        ``target_distance_weights`` if you need to replicate
        ``ResultFrame.aggregate_distances``'s weighting yourself.
        """
        return {"descriptions": self._descriptions, "distances": self._distances}

    @property
    def split_order(self) -> int:
        """Number of ``split_on`` attributes combined per computed slice."""
        return self._split_order

    @property
    def slice_assignments(self) -> SliceAssignments:
        """``{(label, cat): ((attr, value), ...)}`` for every computed slice."""
        return self._slice_assignments

    @property
    def target_distance_weights(self) -> Series:
        """Raw per-row target weight, combined with each model's own weight
        (their max) before aggregating distances."""
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
            slice_assignments=self._slice_assignments,
            split_order=self._split_order,
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
            slice_assignments=self._slice_assignments,
            split_order=self._split_order,
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
            slice_assignments=self._slice_assignments,
            split_order=self._split_order,
        )

    # --- flexible accessor ---

    _LEVELS = ("features", "groups", "domains")

    def at(
        self, level: str = "domains", split: str = "combined", order: int | None = None
    ) -> AggregatedResult:
        """Get an ``AggregatedResult`` at the given level and split.

        The one thing to remember for anything beyond ``summary()`` /
        ``rank_models()`` / ``best_model``: equivalent to chaining the
        ``.features``/``.groups``/``.domains`` and
        ``.combined``/``.by_attribute``/``.by_category`` properties, e.g.
        ``result.at("groups", "by_attribute")`` is ``result.groups.by_attribute``.

        Args:
            level: One of "features", "groups", "domains".
            split: One of "combined", "by_attribute", "by_category".
            order: View order for "by_attribute"/"by_category",
                ``1 <= order <= split_order`` (default ``split_order``). Lower
                orders pool every computed slice containing each row's
                attributes — see ``ScheduleView.view``. Ignored for "combined".

        Returns:
            The requested ``AggregatedResult``.

        Raises:
            ValueError: If ``level``, ``split`` or ``order`` is out of range.
        """
        if level not in self._LEVELS:
            raise ValueError(f"level must be one of {self._LEVELS}, got {level!r}")
        return getattr(self, level).view(split, order)

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
        when the ``EvalResult`` was produced with ``split_on``. With
        ``split_order > 1``, ``*_order1`` directories hold the 1st-order
        containment views too.
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
        if self.has_splits and self._split_order > 1:
            # Also write the 1st-order (marginal) containment views.
            for level in self._LEVELS:
                for split in ("by_attribute", "by_category"):
                    self.at(level, split, order=1).save(out / f"{level}_{split}_order1")


def _renamed(df: DataFrame, name: str, columns: list[str]) -> DataFrame:
    """Select `columns` from df, each renamed with a `"{name}::"` prefix."""
    return df[columns].rename(columns={c: f"{name}::{c}" for c in columns})


def combine(results: dict[str, EvalResult]) -> EvalResult:
    """Combine EvalResults from separate compare() calls into one.

    Typical use: comparing the same synthetic models against several targets,
    or pairing up targets and models one-to-one (see `compare_grid`/
    `compare_many` in `evaluate.py`). Every model column is renamed
    `"{source}::{model}"` so models from different sources stay distinct in
    `model_names` / `summary()` / `rank_models()`.

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
            split_on with results that didn't, or used different split_order.
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

    split_orders = {name: r.split_order for name, r in results.items()}
    if len(set(split_orders.values())) > 1:
        raise ValueError(
            "combine(): cannot mix results with different split_order "
            f"({split_orders}); align split_order across all inputs before combining"
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
        slice_assignments={
            k: v for r in results.values() for k, v in r.slice_assignments.items()
        },
        split_order=base.split_order,
    )
