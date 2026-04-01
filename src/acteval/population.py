"""Population: precomputed numpy representation of a schedule population DataFrame."""

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from acteval._compat import _coerce_to_pandas
from acteval.features._utils import _cumcount


class Population:
    """Precomputed numpy representation of a schedule population DataFrame.

    Converts a DataFrame once and caches all derived quantities to avoid
    redundant computation across feature extraction functions.

    Eager attributes (always computed):
        acts: activity strings.
        act_codes: integer-encoded activities (0, 1, ...).
        starts: start times.
        ends: end times.
        durations: durations.
        pids: dense integer-encoded pids (0, 1, ...).
        unique_acts: sorted unique activity strings.
        n_act_types: number of unique activity types.
        act_to_int: dict mapping activity string → int code.
        int_to_act: ndarray indexed by int code → activity string.
        n: number of unique persons.
        pid_counts: activity count per pid.
        pid_starts: first row index per pid.
        pid_ends: exclusive end row index per pid.
        pid_boundaries: row indices where pid changes (== pid_starts[1:]).
        first_idx: first row index per pid (== pid_starts).
        last_idx: last row index per pid (== pid_ends - 1).

    Lazy properties (computed on first access):
        act_enum_key: cumcount within (pid, act), e.g. "home0", "work0", "home1".
        seq_key: cumcount within pid, e.g. "0home", "1work", "2home".
        act_count_matrix: shape (n, n_act_types) — activity counts per pid.
    """

    def __init__(self, df: DataFrame):
        """Construct a Population from a schedule DataFrame.

        Args:
            df: pandas or Polars DataFrame with columns pid, act, start, end, duration.
                Rows are assumed sorted by pid; a defensive sort is applied if not.
                The act column is optional. For timing, any two of start/end/duration
                are sufficient — the third is derived automatically.
        """
        df = _coerce_to_pandas(df)
        if "pid" not in df.columns:
            raise ValueError("schedule DataFrame is missing required column 'pid'")
        if df["pid"].isna().any():
            raise ValueError(
                "column 'pid' contains NaN values; all rows must have a valid person ID"
            )

        if len(df) == 0:
            self._init_empty()
            return

        has_start = "start" in df.columns
        has_end = "end" in df.columns
        has_dur = "duration" in df.columns

        for col in ("start", "end", "duration"):
            if col in df.columns:
                if not np.issubdtype(df[col].dtype, np.number):
                    raise ValueError(
                        f"column '{col}' must be numeric, got dtype '{df[col].dtype}'"
                    )
                if df[col].isna().any():
                    raise ValueError(f"column '{col}' contains NaN values")

        if has_start and has_end and has_dur:
            if not np.allclose(
                df["duration"].values, df["end"].values - df["start"].values, atol=1e-6
            ):
                raise ValueError(
                    "columns 'start', 'end', and 'duration' are inconsistent: "
                    "duration must equal end - start"
                )

        raw_pids = df["pid"].values

        # Sort by pid if needed (defensive)
        if len(raw_pids) > 1 and not np.all(raw_pids[:-1] <= raw_pids[1:]):
            order = np.argsort(raw_pids, kind="stable")
            raw_pids = raw_pids[order]
        else:
            order = None

        def _col(name, default_dtype=float):
            if name not in df.columns:
                return np.empty(0, dtype=default_dtype)
            vals = df[name].values
            return vals[order] if order is not None else vals

        self.acts = _col("act", object)

        starts = _col("start", float)
        ends = _col("end", float)
        durations = _col("duration", float)

        if has_start and has_end and not has_dur:
            durations = ends - starts
        elif has_start and has_dur and not has_end:
            ends = starts + durations
        elif has_end and has_dur and not has_start:
            starts = ends - durations

        if len(starts) > 0 and len(ends) > 0 and (starts > ends).any():
            raise ValueError(
                "some rows have start > end; all activity intervals must be non-negative"
            )

        self.starts = starts
        self.ends = ends
        self.durations = durations

        # Activity encoding
        if len(self.acts) > 0:
            self.unique_acts, self.act_codes = np.unique(self.acts, return_inverse=True)
        else:
            self.unique_acts = np.array([], dtype=object)
            self.act_codes = np.array([], dtype=np.int64)
        self.n_act_types = len(self.unique_acts)
        self.act_to_int = {a: i for i, a in enumerate(self.unique_acts)}
        self.int_to_act = self.unique_acts  # indexed by int code → act string

        # Pid grouping
        self.unique_pids_original, self.pids, self.pid_counts = np.unique(
            raw_pids, return_inverse=True, return_counts=True
        )
        self.n = len(self.pid_counts)
        self.pid_starts = np.concatenate([[0], np.cumsum(self.pid_counts)[:-1]])
        self.pid_ends = self.pid_starts + self.pid_counts
        self.pid_boundaries = self.pid_starts[1:]  # free — no extra computation
        self.first_idx = self.pid_starts
        self.last_idx = self.pid_ends - 1

        self._lazy_act_enum_key = None
        self._lazy_seq_key = None
        self._lazy_count_matrix = None

    def _init_empty(self) -> None:
        """Initialise all arrays to empty for an empty DataFrame."""
        self.acts = np.array([], dtype=object)
        self.starts = np.empty(0, dtype=float)
        self.ends = np.empty(0, dtype=float)
        self.durations = np.empty(0, dtype=float)
        self.unique_acts = np.array([], dtype=object)
        self.act_codes = np.array([], dtype=np.int64)
        self.n_act_types = 0
        self.act_to_int = {}
        self.int_to_act = np.array([], dtype=object)
        self.unique_pids_original = np.array([], dtype=object)
        self.pids = np.array([], dtype=np.int64)
        self.pid_counts = np.array([], dtype=np.int64)
        self.n = 0
        self.pid_starts = np.array([], dtype=np.int64)
        self.pid_ends = np.array([], dtype=np.int64)
        self.pid_boundaries = np.array([], dtype=np.int64)
        self.first_idx = np.array([], dtype=np.int64)
        self.last_idx = np.array([], dtype=np.int64)
        self._lazy_act_enum_key = None
        self._lazy_seq_key = None
        self._lazy_count_matrix = None

    def __len__(self) -> int:
        """Return total number of activity rows."""
        return len(self.acts)

    def __bool__(self) -> bool:
        """Return False for an empty population."""
        return len(self.acts) > 0

    @property
    def is_empty(self) -> bool:
        """True when the population contains no rows."""
        return len(self.acts) == 0

    @property
    def act_enum_key(self) -> ndarray:
        """Cumcount within (pid, act) group, e.g. ["home0", "work0", "home1", ...]."""
        if self._lazy_act_enum_key is None:
            compound = self.pids * self.n_act_types + self.act_codes
            cumcounts = _cumcount(compound)
            self._lazy_act_enum_key = np.array(
                [str(a) + str(c) for a, c in zip(self.acts, cumcounts)], dtype=object
            )
        return self._lazy_act_enum_key

    @property
    def seq_key(self) -> ndarray:
        """Cumcount within pid group, e.g. ["0home", "1work", "2home", ...]."""
        if self._lazy_seq_key is None:
            cumcounts = _cumcount(self.pids)
            self._lazy_seq_key = np.array(
                [str(c) + str(a) for c, a in zip(cumcounts, self.acts)], dtype=object
            )
        return self._lazy_seq_key

    def dense_pids_from_original(self, original_pids: ndarray) -> ndarray:
        """Map an array of original pid values to dense integer indices.

        Original pids not present in this population are silently dropped.
        """
        orig_to_dense = {orig: i for i, orig in enumerate(self.unique_pids_original)}
        return np.array(
            [orig_to_dense[p] for p in original_pids if p in orig_to_dense],
            dtype=np.int64,
        )

    @property
    def act_count_matrix(self) -> ndarray:
        """Activity count matrix, shape (n, n_act_types)."""
        if self._lazy_count_matrix is None:
            matrix = np.zeros((self.n, self.n_act_types), dtype=np.int64)
            if self.n > 0 and self.n_act_types > 0:
                np.add.at(matrix, (self.pids, self.act_codes), 1)
            self._lazy_count_matrix = matrix
        return self._lazy_count_matrix
