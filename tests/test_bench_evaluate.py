"""pytest-benchmark harness for acteval.evaluate.

Run benchmarks:
    uv run pytest tests/bench_evaluate.py --benchmark-only -v

Compare two runs:
    uv run pytest tests/bench_evaluate.py --benchmark-only --benchmark-compare

Normal test runs (uv run pytest tests/) execute these as smoke tests with no
timing overhead, because pyproject.toml sets addopts = "--benchmark-disable".
"""

import numpy as np
import pytest
from pandas import DataFrame

from acteval.evaluate import compare

ACTIVITIES = ["home", "work", "shop", "education", "leisure"]

SIZES = [
    pytest.param(1_000, id="1k"),
    pytest.param(20_000, id="20k"),
    pytest.param(100_000, id="100k"),
]


def generate_population(n_rows: int, n_activities: int = 5, seed: int = 42) -> DataFrame:
    """Generate a synthetic schedule population with approximately n_rows rows.

    Fully vectorized: all person/episode decisions are made in bulk numpy
    calls, making 100k-row datasets cheap to generate (< 1s).

    Args:
        n_rows: Target number of rows. Actual count may differ slightly
            because episodes-per-person are drawn randomly (2–8).
        n_activities: Number of distinct activity types to use (max 5).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns pid, act, start, end, duration.
        Each person's episodes tile [0, 24) exactly.
    """
    rng = np.random.default_rng(seed)
    acts = np.array(ACTIVITIES[:n_activities])

    # Target n_rows total rows across all persons (mean 5 episodes/person)
    n_people = max(1, n_rows // 5)
    eps_per_person = rng.integers(2, 9, size=n_people)  # (n_people,)
    total_rows = int(eps_per_person.sum())

    # pid for each row
    pids = np.repeat(np.arange(n_people), eps_per_person)

    # Activity for each episode (random)
    act_col = acts[rng.integers(0, len(acts), size=total_rows)]

    # Durations: random weights normalised so each person's episodes sum to 24
    weights = rng.uniform(0.5, 3.0, size=total_rows)
    boundaries = np.concatenate([[0], eps_per_person.cumsum()[:-1]])
    person_sums = np.add.reduceat(weights, boundaries)
    norm = np.repeat(person_sums, eps_per_person)
    durations = weights / norm * 24.0

    # Starts: cumulative sum of durations within each person, reset at boundaries
    dur_groups = np.split(durations, eps_per_person.cumsum()[:-1])
    start_groups = [np.concatenate([[0.0], np.cumsum(d[:-1])]) for d in dur_groups]
    starts = np.concatenate(start_groups)
    ends = starts + durations

    return DataFrame({"pid": pids, "act": act_col, "start": starts, "end": ends, "duration": durations})


@pytest.fixture(params=SIZES, scope="session")
def dataset(request):
    """Pre-generate observed + synthetic DataFrames at the requested scale."""
    n = request.param
    observed = generate_population(n, seed=42)
    synthetic = generate_population(n, seed=99)
    return observed, synthetic


def test_bench_compare(benchmark, dataset):
    """Benchmark the full compare() pipeline at three dataset scales."""
    observed, synthetic = dataset
    benchmark(compare, observed, {"synth": synthetic})
