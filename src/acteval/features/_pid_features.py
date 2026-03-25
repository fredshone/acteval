"""PidFeatures: pre-compression feature data with pid tracking for efficient subsetting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from acteval.features._utils import compress_feature


@dataclass
class PidFeatures:
    """Per-pid feature data that can be subset and aggregated.

    Stores raw (pre-compression) values alongside pid arrays so that features
    can be efficiently filtered to a subset of persons and then compressed
    into the standard ``{key: (values, weights)}`` format.

    Args:
        data: ``{key: (values, pids)}`` where *values* is a 1-D (or 2-D)
            array and *pids* is a same-length array of dense pid ints.
        bin_size: Optional bin size forwarded to ``compress_feature``.
        factor: Divisor forwarded to ``compress_feature``.
    """

    data: dict[str, tuple[ndarray, ndarray]]
    bin_size: int | None = None
    factor: int = 1

    # Internal cache: maps ``id(pid_array)`` → mask for a given pid set,
    # so that per-person features sharing one pid array compute the mask once.
    _mask_cache: dict[int, ndarray] = field(default_factory=dict, repr=False)

    def subset(self, pid_set: ndarray) -> PidFeatures:
        """Return a new PidFeatures containing only values for *pid_set*.

        Args:
            pid_set: 1-D array of dense pid ints to keep.
        """
        pid_set_unique = np.unique(pid_set)
        new_data: dict[str, tuple[ndarray, ndarray]] = {}
        mask_cache: dict[int, ndarray] = {}

        for key, (values, pids) in self.data.items():
            pid_id = id(pids)
            if pid_id not in mask_cache:
                mask_cache[pid_id] = np.isin(pids, pid_set_unique)
            mask = mask_cache[pid_id]
            new_data[key] = (values[mask], pids[mask])

        return PidFeatures(
            data=new_data,
            bin_size=self.bin_size,
            factor=self.factor,
        )

    def aggregate(self) -> dict[str, tuple[ndarray, ndarray]]:
        """Compress per-pid data into ``{key: (unique_values, counts)}``.

        Produces identical output to the existing pipeline's
        ``weighted_features`` / ``compress_feature`` calls.
        """
        result: dict[str, tuple[ndarray, ndarray]] = {}
        for key, (values, _pids) in self.data.items():
            if len(values) == 0:
                result[key] = (np.array([]), np.array([], dtype=np.int64))
            else:
                result[key] = compress_feature(
                    values, bin_size=self.bin_size, factor=self.factor
                )
        return result
