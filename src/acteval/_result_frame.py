"""ResultFrame: separates values, weights, and units from the pipeline's wide DataFrames.

The pipeline internally builds wide DataFrames with interleaved ``{col}`` and
``{col}__weight`` columns plus a ``unit`` string column.  This mixed schema
requires fragile suffix-based filtering in every aggregation function.

``ResultFrame`` splits those three concerns into distinct attributes so that
aggregation is clean — no column-suffix tricks, no drop-and-reattach dance.

Usage in aggregation functions::

    rf = ResultFrame.from_wide(raw_df)
    group_rf = rf.drop_rows(drop_list).aggregate(["domain", "feature"])
    domain_rf = group_rf.mean(["domain"])
    result_df = domain_rf.to_wide()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from pandas import DataFrame, Series

_WEIGHT_SUFFIX = "__weight"


@dataclass
class ResultFrame:
    """Structured view of a pipeline output DataFrame.

    Attributes:
        values:  ``index × model_name`` float DataFrame (distances or descriptions).
        weights: ``index × model_name`` float DataFrame (observation counts).
        units:   ``index → str`` Series (unit label per row, e.g. ``"EMD"``).
                 May be ``None`` at levels where unit has been dropped (domain).
    """

    values: DataFrame
    weights: DataFrame
    units: Series | None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_wide(cls, df: DataFrame) -> "ResultFrame":
        """Parse a wide-format pipeline DataFrame into a ``ResultFrame``.

        Splits columns on the ``__weight`` suffix convention:
        - Columns ending in ``__weight``  → ``weights``
        - ``"unit"`` column              → ``units``
        - All other columns              → ``values``
        """
        unit = df["unit"] if "unit" in df.columns else None
        value_cols = [
            c for c in df.columns if not c.endswith(_WEIGHT_SUFFIX) and c != "unit"
        ]
        values = df[value_cols].copy()

        # Build weights for every key that has a __weight column.
        # This includes weight-only keys (e.g. "observed__weight" with no
        # corresponding "observed" value column — as in the raw distances
        # DataFrame) so that aggregate_distances can access weights["observed"].
        weights_map: dict[str, Series] = {}
        for c in df.columns:
            if c.endswith(_WEIGHT_SUFFIX):
                key = c[: -len(_WEIGHT_SUFFIX)]
                weights_map[key] = df[c]
        # Value columns without a weight column get zero weights.
        for c in value_cols:
            if c not in weights_map:
                weights_map[c] = pd.Series(0.0, index=df.index)
        weights = DataFrame(weights_map)

        return cls(values=values, weights=weights, units=unit)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_wide(self) -> DataFrame:
        """Convert back to interleaved ``{col__weight, col, ..., unit}`` format.

        Preserves the column order the existing pipeline functions expect.
        """
        parts: dict[str, Series] = {}
        for c in self.values.columns:
            parts[f"{c}{_WEIGHT_SUFFIX}"] = self.weights[c]
            parts[c] = self.values[c]
        out = DataFrame(parts, index=self.values.index)
        if self.units is not None:
            out["unit"] = self.units
        return out

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
        n = len(keys[0])
        key_set = set(keys)
        mask = [idx[:n] not in key_set for idx in self.values.index]
        iloc = [i for i, m in enumerate(mask) if m]
        return ResultFrame(
            values=self.values.iloc[iloc],
            weights=self.weights.iloc[iloc],
            units=self.units.iloc[iloc] if self.units is not None else None,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, groupby: list[str]) -> "ResultFrame":
        """Weighted average aggregation (for descriptions).

        For each model column ``c``:
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
        self, groupby: list[str], observed_col: str = "observed"
    ) -> "ResultFrame":
        """Asymmetric weighted average aggregation (for distances).

        Averages each model's weight with the observed weight before computing
        the weighted mean.  This handles asymmetric feature coverage: a feature
        present in only one side gets half-weight rather than zero-weight.

            ``combined_w = (weights[c] + weights[observed_col]) / 2``
            ``agg[c] = sum(values[c] * combined_w) / sum(combined_w)``
        """
        obs_w = self.weights[observed_col]
        agg_values: dict[str, Series] = {}
        agg_weights: dict[str, Series] = {}
        for col in self.values.columns:
            combined_w = (self.weights[col] + obs_w) / 2
            total = combined_w.groupby(groupby).sum()
            weighted = (self.values[col].mul(combined_w)).groupby(groupby).sum()
            agg_values[col] = weighted / total
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
