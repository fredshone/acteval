"""ResultFrame: values, weights, and units as parallel DataFrames.

The pipeline computes a value and a weight (observation count) for every
(row, column) cell — column being a model name, or ``"target"`` for
descriptions. ``ResultFrame`` keeps those two DataFrames (plus a per-row
``units`` Series) genuinely separate, sharing the same index and column set,
so aggregation is plain DataFrame arithmetic — no column-name convention of
any kind.

Usage in aggregation functions::

    group_rf = rf.aggregate(["domain", "feature"])
    domain_rf = group_rf.mean(["domain"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from pandas import DataFrame, Series


@dataclass
class ResultFrame:
    """Structured pipeline output: values, weights, and units.

    Attributes:
        values:  ``index × column`` float DataFrame (distances or descriptions).
        weights: ``index × column`` float DataFrame (observation counts),
                 same shape as ``values``.
        units:   ``index → str`` Series (unit label per row, e.g. ``"EMD"``).
                 ``None`` at levels where unit has been dropped (domain).
    """

    values: DataFrame
    weights: DataFrame
    units: Series | None

    # ------------------------------------------------------------------
    # Row filtering
    # ------------------------------------------------------------------

    def drop_rows(self, keys: Sequence[tuple]) -> "ResultFrame":
        """Return a new ``ResultFrame`` with rows matching ``keys`` removed.

        Matching is by index prefix: a key of length ``n`` drops rows whose
        first ``n`` index levels equal the key.  This mirrors the behaviour
        of ``_drop_features`` in ``_aggregation.py``.
        """
        if not keys:
            return self
        iloc = self.kept_positions(keys)
        return ResultFrame(
            values=self.values.iloc[iloc],
            weights=self.weights.iloc[iloc],
            units=self.units.iloc[iloc] if self.units is not None else None,
        )

    def kept_positions(self, keys: Sequence[tuple]) -> list[int]:
        """Row positions ``drop_rows(keys)`` keeps, for dropping the same rows
        from a Series aligned to this frame (e.g. target weights)."""
        if not keys:
            return list(range(len(self.values)))
        n = len(keys[0])
        key_set = set(keys)
        return [i for i, idx in enumerate(self.values.index) if idx[:n] not in key_set]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, groupby: list[str]) -> "ResultFrame":
        """Weighted average aggregation (for descriptions).

        For each column ``c``:
            ``agg[c] = sum(values[c] * weights[c]) / sum(weights[c])``

        Returns a ``ResultFrame`` whose ``weights`` are the summed weights
        (useful for chained aggregation or user inspection).
        """
        total_w = self.weights.groupby(groupby).sum()
        weighted_v = (self.values.mul(self.weights)).groupby(groupby).sum()
        agg_values = weighted_v.div(total_w).fillna(0.0)
        units = self.units.groupby(groupby).first() if self.units is not None else None
        return ResultFrame(values=agg_values, weights=total_w, units=units)

    def aggregate_distances(
        self, groupby: list[str], target_weights: Series
    ) -> "ResultFrame":
        """Weighted average aggregation for distances, using both sides' weights.

        Each row is weighted by the larger of the model's and the target's
        weight, so a segment present on only one side (e.g. an activity the
        model never produces, or one it invents) keeps that side's full weight.
        Model weights are already rescaled to the target's slice size (see
        ``Evaluator.compare_population``), so neither side dominates because
        of its sample size.

        The denominator only counts rows the target has (``target_weights >
        0``): rows only the model has (e.g. invented n-grams) add to the
        numerator but not the denominator, so they can only increase the
        distance. Otherwise many low-distance invented segments would dilute
        the average and make the model look better. A group with no target
        rows at all (e.g. a single invented segment at feature level, or a
        slice whose target has no 4-grams) falls back to the plain weighted
        mean over its rows.

            ``combined_w = max(weights[c], target_weights)``
            ``agg[c] = sum(values[c] * combined_w) / sum(combined_w[target_weights > 0])``

        The result can therefore exceed the largest per-row distance when the
        model's invented segments outweigh the target's own.

        Args:
            target_weights: Raw per-row target weight, in the same row order
                as this ``ResultFrame``. Combined by position, not by label:
                lower-order views repeat index labels, which label alignment
                would cross-multiply.
        """
        if len(target_weights) != len(self.values):
            raise ValueError(
                f"target_weights has {len(target_weights)} rows, expected "
                f"{len(self.values)} (one per row, in the same order)"
            )
        target = target_weights.to_numpy(dtype=float)
        target_present = target > 0
        agg_values: dict[str, Series] = {}
        agg_weights: dict[str, Series] = {}
        for col in self.values.columns:
            combined = np.maximum(self.weights[col].to_numpy(dtype=float), target)
            combined_w = Series(combined, index=self.values.index)
            total = combined_w.groupby(groupby).sum()
            present_total = (
                Series(np.where(target_present, combined, 0.0), index=self.values.index)
                .groupby(groupby)
                .sum()
            )
            denominator = present_total.where(present_total > 0, total)
            weighted = (self.values[col].mul(combined_w)).groupby(groupby).sum()
            agg_values[col] = (weighted / denominator).fillna(0.0)
            agg_weights[col] = total
        units = self.units.groupby(groupby).first() if self.units is not None else None
        return ResultFrame(
            values=DataFrame(agg_values),
            weights=DataFrame(agg_weights),
            units=units,
        )

    def mean(self, groupby: list[str]) -> "ResultFrame":
        """Unweighted mean aggregation (for domain-level collapse).

        At the domain level the pipeline uses a simple ``groupby.mean()`` so
        that every feature group contributes equally regardless of observation
        counts.  This method preserves that behaviour.
        """
        agg_values = self.values.groupby(groupby).mean()
        agg_weights = self.weights.groupby(groupby).sum()
        units = self.units.groupby(groupby).first() if self.units is not None else None
        return ResultFrame(values=agg_values, weights=agg_weights, units=units)
