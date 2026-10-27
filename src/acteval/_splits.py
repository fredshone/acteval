import itertools
import warnings
from dataclasses import dataclass

import numpy as np
from pandas import DataFrame

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
