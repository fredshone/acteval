# acteval

A Python library for evaluating synthetic activity schedules by comparing them to observed data. Given a population of daily activity sequences (who did what and when), `acteval` measures how well a synthetic population reproduces the observed distribution across multiple dimensions: activity frequencies, timing, transitions, participation rates, and novelty.

## Install

```bash
pip install acteval
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add acteval
```

## Quick start

```python
import pandas as pd
from acteval import compare

observed = pd.DataFrame([
    {"pid": 0, "act": "home", "start": 0,  "end": 8,  "duration": 8},
    {"pid": 0, "act": "work", "start": 8,  "end": 16, "duration": 8},
    {"pid": 0, "act": "home", "start": 16, "end": 24, "duration": 8},
    {"pid": 1, "act": "home", "start": 0,  "end": 12, "duration": 12},
    {"pid": 1, "act": "shop", "start": 12, "end": 13, "duration": 1},
    {"pid": 1, "act": "home", "start": 13, "end": 24, "duration": 11},
])

synthetic = pd.DataFrame([
    {"pid": 0, "act": "home", "start": 0,  "end": 9,  "duration": 9},
    {"pid": 0, "act": "work", "start": 9,  "end": 17, "duration": 8},
    {"pid": 0, "act": "home", "start": 17, "end": 24, "duration": 7},
    {"pid": 1, "act": "home", "start": 0,  "end": 8,  "duration": 8},
    {"pid": 1, "act": "work", "start": 8,  "end": 16, "duration": 8},
    {"pid": 1, "act": "home", "start": 16, "end": 24, "duration": 8},
    {"pid": 2, "act": "home", "start": 0,  "end": 8,  "duration": 8},
    {"pid": 2, "act": "home", "start": 8, "end": 24, "duration": 16},
])

result = compare(observed, {"my_model": synthetic})
print(result.summary())
# creativity      0.166667
# feasibility     0.500000
# participations  0.162037
# timing          0.082728
# transitions     0.380952
```

## Input format

Data is passed as a pandas DataFrame with one row per activity episode:

| column | type | description |
|--------|------|-------------|
| `pid` | int/str | Person identifier |
| `act` | str | Activity label (e.g. `"home"`, `"work"`, `"shop"`) |
| `start` | numeric | Start time (any consistent unit, e.g. hours) |
| `end` | numeric | End time |


```python
import pandas as pd

observed = pd.DataFrame([
    {"pid": 0, "act": "home", "start": 0,  "end": 6,  "duration": 6},
    {"pid": 0, "act": "work", "start": 6,  "end": 14, "duration": 8},
    {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
    {"pid": 1, "act": "home", "start": 0,  "end": 10, "duration": 10},
    {"pid": 1, "act": "work", "start": 10, "end": 24, "duration": 14},
])
```

## API

### `compare(observed, synthetic, **kwargs)`

Compare one or more synthetic populations to an observed population.

```python
from acteval import compare

# Single synthetic population
result = compare(observed, synthetic)

# Multiple models side-by-side
result = compare(observed, {"model_a": synthetic_a, "model_b": synthetic_b})
```

`result` is a dict with six DataFrames:

| key | description |
|-----|-------------|
| `"distances"` | Per-feature distances |
| `"group_distances"` | Distances aggregated by feature group |
| `"domain_distances"` | Distances aggregated by domain (participations, transitions, timing) |
| `"descriptions"` | Per-feature descriptive statistics |
| `"group_descriptions"` | Group-level descriptive statistics |
| `"domain_descriptions"` | Domain-level descriptive statistics |

Pass `report_stats=True` to include additional statistics in the output.

### `Evaluator`

Use `Evaluator` when comparing multiple synthetic populations against the same observed data — it computes and caches the observed features once. Each `compare()` call is independent.

```python
from acteval import Evaluator

evaluator = Evaluator(observed)

result_v1 = evaluator.compare({"v1": synthetic_v1})
result_v2 = evaluator.compare({"v2": synthetic_v2})
```

For fine-grained control, compare one model at a time then assemble:

```python
evaluator.compare_population("v1", synthetic_v1)
evaluator.compare_population("v2", synthetic_v2)
result = evaluator.report()
```

### `pairwise_distances(schedules, specs=None)`

Compute a single NxN distance matrix between individual schedules. Useful for clustering, outlier detection, or directly comparing a small batch of schedules.

```python
from acteval import pairwise_distances

result = pairwise_distances(schedules)
result.matrix          # numpy array, shape (N, N)
result.pids            # original pid values, length N

# Get a labeled DataFrame with original pid values as index/columns
df = result.to_dataframe()

# Example: find the two most similar schedules
import numpy as np
dist = result.to_dataframe()
dist.values[np.arange(len(dist)), np.arange(len(dist))] = np.inf
i, j = np.unravel_index(dist.values.argmin(), dist.shape)
print(f"Most similar: {dist.index[i]} and {dist.columns[j]}")
```

The result matrix is symmetric with zeros on the diagonal. All values are in **0–1**.

#### Pluggable distance specs

By default, three equal-weight semantic-distance specs are used (participations, transitions, timing via MAE). Pass a custom `specs` list to change the metrics or their relative weights:

```python
from acteval.pairwise import chamfer_spec, soft_dtw_spec, default_pairwise_specs

# Chamfer distance on EOS-padded activity sequences
result = pairwise_distances(schedules, specs=[chamfer_spec()])

# Soft-DTW on EOS-padded activity sequences
result = pairwise_distances(schedules, specs=[soft_dtw_spec(gamma=1.0)])

# Mix metrics with custom weights
result = pairwise_distances(schedules, specs=[
    *default_pairwise_specs(),       # weight=1.0 each
    chamfer_spec(weight=1.0),
    soft_dtw_spec(weight=2.0),
])
```

Each spec defines a `feature_fn` (extracts a `(N, ...)` array from the population) and a `distance_fn` (computes the `(N, N)` matrix). The final matrix is a weighted average across all active specs.

| Factory | Description |
|---------|-------------|
| `default_pairwise_specs()` | MAE on participation counts, bi-gram counts, mean durations |
| `chamfer_spec(max_len, weight)` | Chamfer distance on EOS-padded `(N, L, 2)` sequences |
| `soft_dtw_spec(max_len, gamma, weight)` | Soft-DTW on EOS-padded `(N, L, 2)` sequences |

### `Population`

For direct access to the underlying data structure:

```python
from acteval import Population

pop = Population(observed)
print(pop.acts)       # activity labels per episode
print(pop.durations)  # durations as numpy array
```

## Reading the results

### The six DataFrames

`compare()` returns an `EvalResult` with six DataFrames at three levels of aggregation:

| Property | Index | Content |
|----------|-------|---------|
| `result.distances` | `(domain, feature, segment)` | Per-feature distances — lower is closer to observed |
| `result.descriptions` | `(domain, feature, segment)` | Per-feature descriptive statistics (e.g. average start time) |
| `result.group_distances` | `(domain, feature)` | Distances averaged across segments |
| `result.group_descriptions` | `(domain, feature)` | Descriptions averaged across segments |
| `result.domain_distances` | `(domain,)` | Distances averaged across features — one row per domain |
| `result.domain_descriptions` | `(domain,)` | Descriptions averaged across features |

Distances are in the range **0–1** (lower is better). A distance of `0.0` means the synthetic distribution perfectly matches observed; `1.0` is the maximum penalty.

> **Note on timing features:** A distance of `1.0` for a timing feature means the activity is *entirely absent* from the synthetic population — not just timed differently. This is treated as a maximum-penalty missing feature rather than a distributional difference.

### Evaluation domains

| Domain | What it measures |
|--------|-----------------|
| `participations` | Who does what and how often — participation rates, joint participation, sequence lengths |
| `transitions` | Activity sequences — 2-, 3-, and 4-gram transition patterns |
| `timing` | When and how long — start times, durations, and their joint distributions |
| `creativity` | How novel and diverse the synthetic schedules are relative to observed |
| `feasibility` | Structural validity — home-based schedules, no consecutive duplicate activities |

### Ranking models

```python
result = compare(observed, {"model_a": df_a, "model_b": df_b})

# Mean domain distance per model (lower is better)
print(result.rank_models())
# model_a    0.12
# model_b    0.19

# Best model
print(result.best_model)   # "model_a"

# Domain-level summary table
print(result.summary())
#                   model_a  model_b
# domain
# creativity           0.08     0.14
# feasibility          0.03     0.05
# participations       0.11     0.18
# timing               0.16     0.22
# transitions          0.22     0.35
```

## Development

```bash
# Install dependencies (including dev tools)
uv sync

# Run tests (benchmarks excluded by default)
uv run pytest tests/

# Run tests with coverage
uv run pytest --cov=src/acteval tests/

# Lint with ruff
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Check formatting with black
black --check src/ tests/

# Auto-format
black src/ tests/
```

## Benchmarks

Two benchmark suites are included, both using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).

### Population evaluation (`compare`)

Tests `compare()` at 1k, 20k, and 100k rows (observed + synthetic populations):

```bash
uv run pytest tests/test_bench_evaluate.py --benchmark-only -v
```

### Pairwise distances (`pairwise_distances`)

Tests `pairwise_distances()` at 256, 512, and 1024 schedules:

```bash
uv run pytest tests/test_bench_pairwise.py --benchmark-only -v
```

### Both suites together

```bash
uv run pytest tests/test_bench_evaluate.py tests/test_bench_pairwise.py --benchmark-only -v
```

### Comparing runs

```bash
# Save a baseline
uv run pytest tests/test_bench_evaluate.py tests/test_bench_pairwise.py --benchmark-only --benchmark-save=baseline

# Compare against it after making changes
uv run pytest tests/test_bench_evaluate.py tests/test_bench_pairwise.py --benchmark-only --benchmark-compare=baseline
```
