"""Combine EvalResults from separate compare() calls into one.

`compare()`/`Evaluator` compare N synthetic models against a single target.
This module adds the missing piece for comparing the same synthetic models
against *several* targets: `compare_many()` runs one ordinary `compare()`
call per target and `combine()`s the results into a single `EvalResult`,
with every model column namespaced by its target so nothing collides.
"""

import warnings

from pandas import DataFrame, concat

from acteval._result_frame import ResultFrame
from acteval.evaluate import EvalResult, compare


def _renamed(df: DataFrame, name: str, columns: list[str]) -> DataFrame:
    """Select `columns` from df, each renamed with a `"{name}::"` prefix."""
    return df[columns].rename(columns={c: f"{name}::{c}" for c in columns})


def combine(results: dict[str, EvalResult]) -> EvalResult:
    """Combine EvalResults from separate compare() calls into one.

    Typical use: comparing the same synthetic models against several targets
    (see `compare_many`). Every model column is renamed `"{source}::{model}"`
    so models from different sources stay distinct in `model_names` /
    `summary()` / `rank_models()`.

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


def compare_many(
    targets: dict[str, DataFrame],
    synthetic: dict[str, DataFrame],
    attributes: dict[str, DataFrame] | None = None,
    target_attributes: dict[str, DataFrame] | None = None,
    split_on: list[str] | None = None,
    **kwargs,
) -> EvalResult:
    """Compare the same synthetic models against each of several targets.

    Runs one ordinary `compare()` call per target, then `combine()`s the
    results into a single `EvalResult` with model columns named
    `"{target_name}::{model_name}"`.

    Args:
        targets: `{target_name: observed_schedules_df}`.
        synthetic: `{model_name: schedules_df}`, compared against every target.
        attributes: Optional `{model_name: attributes_df}`, shared across targets.
        target_attributes: Optional `{target_name: attributes_df}` — per-target
            attributes, required together with `split_on`.
        split_on: Optional attribute column(s) to split each target's evaluation by.
        **kwargs: Passed through to `compare()` (e.g. `disable`, `progress`).

    Returns:
        A single combined `EvalResult`; see `combine()` for details and the
        known target-weight-base limitation.
    """
    results = {
        name: compare(
            target_df,
            synthetic,
            attributes=attributes,
            target_attributes=(
                target_attributes.get(name) if target_attributes else None
            ),
            split_on=split_on,
            **kwargs,
        )
        for name, target_df in targets.items()
    }
    return combine(results)
