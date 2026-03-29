"""pytest-benchmark harness for acteval.pairwise_distances.

Run benchmarks:
    uv run pytest tests/test_bench_pairwise.py --benchmark-only -v

Compare two runs:
    uv run pytest tests/test_bench_pairwise.py --benchmark-only --benchmark-compare

Normal test runs (uv run pytest tests/) execute these as smoke tests with no
timing overhead, because pyproject.toml sets addopts = "--benchmark-disable".
"""

import numpy as np
import pytest
from pandas import DataFrame

from acteval.pairwise import pairwise_distances

ACTIVITIES = ["home", "work", "shop", "education", "leisure"]

SIZES = [
    pytest.param(256, id="256"),
    pytest.param(512, id="512"),
    pytest.param(1024, id="1024"),
]


def generate_schedules(n_persons: int, seed: int = 42) -> DataFrame:
    """Generate a synthetic schedule DataFrame with exactly n_persons unique pids.

    Each person gets 2–8 episodes that tile [0, 24).
    """
    rng = np.random.default_rng(seed)
    acts = np.array(ACTIVITIES)

    eps_per_person = rng.integers(2, 9, size=n_persons)
    total_rows = int(eps_per_person.sum())

    pids = np.repeat(np.arange(n_persons), eps_per_person)
    act_col = acts[rng.integers(0, len(acts), size=total_rows)]

    weights = rng.uniform(0.5, 3.0, size=total_rows)
    boundaries = np.concatenate([[0], eps_per_person.cumsum()[:-1]])
    person_sums = np.add.reduceat(weights, boundaries)
    norm = np.repeat(person_sums, eps_per_person)
    durations = weights / norm * 24.0

    dur_groups = np.split(durations, eps_per_person.cumsum()[:-1])
    start_groups = [np.concatenate([[0.0], np.cumsum(d[:-1])]) for d in dur_groups]
    starts = np.concatenate(start_groups)
    ends = starts + durations

    return DataFrame({"pid": pids, "act": act_col, "start": starts, "end": ends, "duration": durations})


@pytest.fixture(params=SIZES, scope="session")
def schedules(request):
    return generate_schedules(request.param, seed=42)


def test_bench_pairwise_distances(benchmark, schedules):
    """Benchmark pairwise_distances() at three schedule-count scales."""
    benchmark(pairwise_distances, schedules)
