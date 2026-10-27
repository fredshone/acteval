import itertools
import re
import warnings
from dataclasses import dataclass

import numpy as np
from pandas import DataFrame

from acteval._jobs import JobSpec, get_jobs
from acteval.features._pid_features import PidFeatures

_TRAILING_DIGITS = re.compile(r"\d+$")

# Default minimum target persons per slice for order >= 2 (order 1 keeps every
# observed category, matching pre-``split_order`` behaviour).
DEFAULT_HIGHER_ORDER_MIN_SIZE = 30

# Warn when a run would evaluate more slices than this.
_MANY_SLICES = 1000

LABEL_SEP = "&"


@dataclass(frozen=True)
class Slice:
    """One population slice: a joint assignment of ``order`` attributes.

    ``attrs`` follow ``split_on`` order; ``values`` are aligned with ``attrs``.
    """

    attrs: tuple[str, ...]
    values: tuple

    @property
    def order(self) -> int:
        return len(self.attrs)

    @property
    def label(self) -> str:
        """Index ``label`` string, e.g. ``"gender"`` or ``"gender&income"``."""
        return LABEL_SEP.join(self.attrs)

    @property
    def cat(self):
        """Index ``cat`` value: the raw value at order 1 (unchanged from
        pre-``split_order`` output), else the joined string, e.g. ``"male&high"``."""
        if self.order == 1:
            return self.values[0]
        return LABEL_SEP.join(str(v) for v in self.values)

    @property
    def key(self) -> tuple[str, object]:
        return (self.label, self.cat)

    @property
    def assignment(self) -> tuple[tuple[str, object], ...]:
        """``((attr, value), ...)`` — what containment views expand from."""
        return tuple(zip(self.attrs, self.values))


def resolve_min_size(order: int, min_size: int | None) -> int:
    """``None`` means auto: no minimum at order 1, else
    ``DEFAULT_HIGHER_ORDER_MIN_SIZE``."""
    if min_size is not None:
        return min_size
    return 0 if order == 1 else DEFAULT_HIGHER_ORDER_MIN_SIZE


def group_pids(
    attributes: DataFrame, attrs: tuple[str, ...]
) -> dict[tuple, np.ndarray]:
    """``{values_tuple: original_pids}`` for every observed combination of
    ``attrs``, in order of first appearance."""
    grouped = attributes.groupby(list(attrs), sort=False, dropna=True).pid
    return {(k if isinstance(k, tuple) else (k,)): pids.values for k, pids in grouped}


def build_slices(
    attributes: DataFrame,
    split_on: list[str],
    order: int,
    min_size: int | None = None,
) -> list[tuple[Slice, np.ndarray]]:
    """Build every order-``order`` slice observed in *attributes*.

    Args:
        attributes: Target attributes with a ``pid`` column and every
            ``split_on`` column.
        split_on: Attribute columns to combine.
        order: Number of attributes per slice (``1 <= order <= len(split_on)``).
        min_size: Minimum target persons per slice; ``None`` for the default
            (see ``resolve_min_size``).

    Returns:
        ``[(slice, target_original_pids), ...]``, grouped by attribute
        combination in ``itertools.combinations(split_on, order)`` order.

    Raises:
        ValueError: If ``order`` is out of range, or every slice is smaller
            than ``min_size``.
    """
    if not 1 <= order <= len(split_on):
        raise ValueError(
            f"split_order must be between 1 and len(split_on)={len(split_on)}, "
            f"got {order}"
        )
    min_size = resolve_min_size(order, min_size)
    slices: list[tuple[Slice, np.ndarray]] = []
    dropped: list[str] = []
    for attrs in itertools.combinations(split_on, order):
        for values, pids in group_pids(attributes, attrs).items():
            s = Slice(attrs=attrs, values=values)
            if len(np.unique(pids)) < min_size:
                dropped.append(f"{s.label}={s.cat}")
                continue
            slices.append((s, pids))
    if dropped:
        if not slices:
            raise ValueError(
                f"every order-{order} slice has fewer than min_split_size={min_size} "
                "target persons; lower split_order or min_split_size"
            )
        preview = ", ".join(dropped[:5]) + (", ..." if len(dropped) > 5 else "")
        warnings.warn(
            f"skipped {len(dropped)} of {len(dropped) + len(slices)} order-{order} "
            f"slices with fewer than min_split_size={min_size} target persons: "
            f"{preview}",
            UserWarning,
            stacklevel=3,
        )
    if len(slices) > _MANY_SLICES:
        warnings.warn(
            f"split_order={order} produces {len(slices)} slices; expect roughly "
            "0.1-0.25s per slice for the target and again per model. Consider a "
            "lower split_order, fewer split_on columns, or a larger min_split_size.",
            UserWarning,
            stacklevel=3,
        )
    return slices


def _get_density_jobs(config_path=None) -> list[JobSpec]:
    """Return flat list of JobSpec for all active density jobs."""
    return get_jobs(config_path).density


def _key_activities(key: str) -> frozenset[str] | None:
    """Return activity names referenced by a feature key, or None for non-activity keys.

    Key formats:
    - n-gram transitions: "act1>act2>act3" (split by ">")
    - joint participation: "act1+act2" (split by "+")
    - timing / single-act: "actN" or "act" (strip trailing digits)
    - non-activity keys like "sequence lengths" contain spaces → return None
    """
    if ">" in key:
        return frozenset(key.split(">"))
    if "+" in key:
        return frozenset(key.split("+"))
    stripped = _TRAILING_DIGITS.sub("", key)
    if not stripped or " " in stripped:
        return None
    return frozenset([stripped])


def _subset_pid_features(
    pid_features: dict[str, dict[tuple, "PidFeatures"]],
    dense_pids: dict[str, np.ndarray],
    subset_acts: dict[str, frozenset[str]] | None = None,
) -> dict[str, dict[tuple, dict]]:
    """Subset each model's PidFeatures to dense_pids and aggregate.

    Empty entries and entries for activities absent from the subset are dropped
    so the result matches what ``feature_fn(Population(sub_df))`` would return.

    Args:
        subset_acts: Optional ``{model: frozenset_of_activity_names}`` used to
            filter out keys referencing activities not in the subset.
    """
    result: dict[str, dict[tuple, dict]] = {}
    for model, model_features in pid_features.items():
        acts = subset_acts[model] if subset_acts is not None else None
        model_result: dict[tuple, dict] = {}
        for key, pf in model_features.items():
            aggregated = pf.aggregate_subset(dense_pids[model])
            filtered: dict = {}
            for k, v in aggregated.items():
                if len(v[0]) == 0:
                    continue
                if acts is not None:
                    key_acts = _key_activities(k)
                    if key_acts is not None and not key_acts.issubset(acts):
                        continue
                filtered[k] = v
            model_result[key] = filtered
        result[model] = model_result
    return result
