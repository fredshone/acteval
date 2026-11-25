"""PidFeatures: pre-compression feature data with pid tracking for efficient subsetting."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from acteval.features._utils import bin_values, compress_feature


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
        zero_is_missing: True for count features (e.g. participation
            counts), where a segment that is zero for everyone means it didn't
            occur, so ``aggregate_present`` treats it as missing. Values are
            then per-person counts, so a segment's weight is their sum (see
            ``_aggregation.segment_count``).
    """

    data: dict[str, tuple[ndarray, ndarray]]
    bin_size: int | None = None
    factor: int = 1
    zero_is_missing: bool = False

    # Internal cache: ``key`` → ``(scaled_uniques, inverse)``, so repeated
    # ``aggregate_subset`` calls count codes instead of re-sorting values.
    _encoded: dict[str, tuple[ndarray, ndarray]] = field(
        default_factory=dict, repr=False
    )

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
            zero_is_missing=self.zero_is_missing,
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

    def _encode(self, key: str) -> tuple:
        """Cache ``key``'s values as integer codes into their sorted uniques.

        Dense form: ``(scaled_uniques, inverse)`` with one code per value.

        Sparse form, for 1-D unbinned ``zero_is_missing`` count features
        (mostly zeros, e.g. one column per n-gram over every person):
        ``(scaled_uniques, nz_inverse, nz_pids, zero_code)``, keeping only the
        nonzero values; a subset's zero count is derived from how many of its
        entries are nonzero, so per-subset work scales with the nonzeros, not
        the population.
        """
        if key not in self._encoded:
            values, pids = self.data[key]
            values = np.asarray(values)
            if self.zero_is_missing and self.bin_size is None and values.ndim == 1:
                nz = np.flatnonzero(values)
                uniques, nz_inverse = np.unique(
                    np.concatenate([[0], values[nz]]), return_inverse=True
                )
                zero_code = int(nz_inverse[0])
                self._encoded[key] = (
                    uniques / self.factor,
                    nz_inverse[1:].reshape(-1),
                    np.asarray(pids)[nz],
                    zero_code,
                )
            else:
                if self.bin_size is not None:
                    values = bin_values(values, self.bin_size)
                uniques, inverse = np.unique(values, axis=0, return_inverse=True)
                self._encoded[key] = (uniques / self.factor, inverse.reshape(-1))
        return self._encoded[key]

    def encode_all(self) -> None:
        """Fill the per-key encoding cache up front, so concurrent
        ``aggregate_subset`` calls only read it."""
        for key, (values, _pids) in self.data.items():
            if len(values):
                self._encode(key)

    def aggregate_subset(self, pid_set: ndarray) -> dict[str, tuple[ndarray, ndarray]]:
        """Equivalent to ``self.subset(pid_set).aggregate()``, but faster when
        called repeatedly on the same object: values are encoded once and each
        call only counts the codes belonging to *pid_set*.

        Args:
            pid_set: 1-D array of dense pid ints to keep.
        """
        pid_set = np.asarray(pid_set, dtype=np.int64)
        result: dict[str, tuple[ndarray, ndarray]] = {}
        if len(pid_set) == 0:
            for key in self.data:
                result[key] = (np.array([]), np.array([], dtype=np.int64))
            return result
        # Pid-space membership lookup, grown to cover every pids array seen.
        member = np.zeros(pid_set.max() + 1, dtype=bool)
        member[pid_set] = True
        mask_cache: dict[int, ndarray] = {}
        n_cache: dict[int, int] = {}

        def lookup(pids: ndarray) -> ndarray:
            nonlocal member
            top = pids.max() if len(pids) else -1
            if top >= len(member):
                member = np.concatenate(
                    [member, np.zeros(top + 1 - len(member), dtype=bool)]
                )
            return member[pids]

        for key, (values, pids) in self.data.items():
            if len(values) == 0:
                result[key] = (np.array([]), np.array([], dtype=np.int64))
                continue
            pid_id = id(pids)
            encoded = self._encode(key)
            if len(encoded) == 4:
                uniques, nz_inverse, nz_pids, zero_code = encoded
                # Entries of this pids array in the subset (shared across keys).
                if pid_id not in n_cache:
                    n_cache[pid_id] = int(lookup(pids).sum())
                n_in_subset = n_cache[pid_id]
                nz_mask = lookup(nz_pids)
                counts = np.bincount(nz_inverse[nz_mask], minlength=len(uniques))
                counts[zero_code] += n_in_subset - int(nz_mask.sum())
            else:
                uniques, inverse = encoded
                if pid_id not in mask_cache:
                    mask_cache[pid_id] = lookup(pids)
                counts = np.bincount(
                    inverse[mask_cache[pid_id]], minlength=len(uniques)
                )
            present = counts > 0
            if not present.any():
                result[key] = (np.array([]), np.array([], dtype=np.int64))
            else:
                result[key] = (uniques[present], counts[present].astype(np.int64))
        return result

    def aggregate_present(self, pid_set: ndarray) -> dict[str, tuple[ndarray, ndarray]]:
        """``aggregate_subset`` without the segments missing from *pid_set*:
        those with no values, or, when ``zero_is_missing``, only zeros.

        Args:
            pid_set: 1-D array of dense pid ints to keep.
        """
        return {
            k: (values, weights)
            for k, (values, weights) in self.aggregate_subset(pid_set).items()
            if len(values) > 0 and not (self.zero_is_missing and not values.any())
        }
