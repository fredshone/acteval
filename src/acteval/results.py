"""Combine EvalResults from separate compare() calls into one.

`compare()`/`Evaluator` compare N synthetic models against a single target.
This module adds the missing piece for comparing the same synthetic models
against *several* targets: `compare_many()` runs one ordinary `compare()`
call per target and `combine()`s the results into a single `EvalResult`,
with every model column namespaced by its target so nothing collides.
"""

from pandas import DataFrame, concat

from acteval.evaluate import EvalResult, compare

_BASE_COLS = {"observed", "observed__weight", "unit"}


def _model_columns(df: DataFrame, name: str) -> DataFrame:
    """Return df's non-base columns, renamed with a `"{name}::"` prefix so
    models from different sources never collide."""
    cols = [c for c in df.columns if c not in _BASE_COLS]
    renamed = {
        c: (
            f"{name}::{c[: -len('__weight')]}__weight"
            if c.endswith("__weight")
            else f"{name}::{c}"
        )
        for c in cols
    }
    return df[cols].rename(columns=renamed)


def combine(results: dict[str, EvalResult]) -> EvalResult:
    """Combine EvalResults from separate compare() calls into one.

    Typical use: comparing the same synthetic models against several targets
    (see `compare_many`). Every model column is renamed `"{source}::{model}"`
    so models from different sources stay distinct in `model_names` /
    `summary()` / `rank_models()`.

    The shared `observed`/`observed__weight`/`unit` columns are taken from
    the *first* result only. Every model's distances were already computed
    against its own source's target, so this only affects which weights are
    used when re-aggregating a non-first source's columns from features to
    groups to domains — a known simplification for now, not an error in the
    distances themselves. A future refactor could carry one base per source
    (e.g. `EvalResult.raw_base_distances: dict[str, DataFrame]`) to remove
    this limitation.

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

    base_result = results[names[0]]
    base_desc = base_result._raw_desc[
        [c for c in base_result._raw_desc.columns if c in _BASE_COLS]
    ]
    base_dist = base_result._raw_dist[
        [c for c in base_result._raw_dist.columns if c in _BASE_COLS]
    ]

    raw_desc = concat(
        [base_desc] + [_model_columns(r._raw_desc, n) for n, r in results.items()],
        axis=1,
    )
    raw_dist = concat(
        [base_dist] + [_model_columns(r._raw_dist, n) for n, r in results.items()],
        axis=1,
    )
    return EvalResult(raw_desc=raw_desc, raw_dist=raw_dist)


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
        known base-column limitation.
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
