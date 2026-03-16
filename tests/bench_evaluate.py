"""Benchmark harness for acteval.evaluate.

Usage:
    python tests/bench_evaluate.py --population 1000 --activities 10
    python tests/bench_evaluate.py --population 5000 --activities 15
"""

import argparse
import time

import numpy as np
from pandas import DataFrame

from acteval.evaluate import process_metrics

ACTIVITIES = ["home", "work", "shop", "education", "leisure"]


def generate_population(n_people: int, n_activities: int, seed: int = 42) -> DataFrame:
    """Generate a synthetic schedule population using plain numpy/pandas.

    Args:
        n_people: Number of people (schedules) to generate.
        n_activities: Number of activity types to use (max 5).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns pid, act, start, end, duration.
    """
    rng = np.random.default_rng(seed)
    acts = ACTIVITIES[:n_activities]
    rows = []
    for pid in range(n_people):
        t = 0
        while t < 24:
            act = acts[rng.integers(len(acts))]
            dur = float(rng.integers(1, max(2, (24 - t))))
            end = min(t + dur, 24)
            rows.append({"pid": pid, "act": act, "start": t, "end": end, "duration": end - t})
            t = end
            if t >= 24:
                break
    return DataFrame(rows)


def bench(n_people: int, n_activities: int):
    """Run the benchmark: generate data, time process_metrics."""
    print(f"Generating target population ({n_people} people, {n_activities} activities)...")
    t0 = time.perf_counter()
    target = generate_population(n_people, n_activities, seed=42)
    t_gen_target = time.perf_counter() - t0
    print(f"  target: {len(target)} rows in {t_gen_target:.3f}s")

    t0 = time.perf_counter()
    synthetic = generate_population(n_people, n_activities, seed=99)
    t_gen_synth = time.perf_counter() - t0
    print(f"  synthetic: {len(synthetic)} rows in {t_gen_synth:.3f}s")

    print(f"\nRunning process_metrics()...")
    t_start = time.perf_counter()
    process_metrics(
        synthetic_schedules={"synth": synthetic},
        target_schedules=target,
        verbose=True,
    )
    t_total = time.perf_counter() - t_start
    print(f"\nTotal process_metrics time: {t_total:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark acteval evaluate module")
    parser.add_argument(
        "--population", type=int, default=20000, help="Number of people per population"
    )
    parser.add_argument(
        "--activities", type=int, default=5, help="Number of activity types (max 5)"
    )
    args = parser.parse_args()
    bench(args.population, args.activities)
