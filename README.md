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

## Development

```bash
# Install dependencies (including dev tools)
uv sync

# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=src/acteval tests/

# Lint with ruff
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Check formatting with black
black --check src/ tests/

# Auto-format
black src/ tests/
```

## Input format

Data is passed as a pandas DataFrame with one row per activity episode:

| column | type | description |
|--------|------|-------------|
| `pid` | int/str | Person identifier |
| `act` | str | Activity label (e.g. `"home"`, `"work"`, `"shop"`) |
| `start` | numeric | Start time (any consistent unit, e.g. hours) |
| `end` | numeric | End time |
| `duration` | numeric | Duration (`end - start`) |

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
