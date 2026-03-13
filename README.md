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

Use `Evaluator` when comparing multiple synthetic populations against the same observed data — it computes and caches the observed features once.

```python
from acteval import Evaluator

evaluator = Evaluator(observed)

result_v1 = evaluator.compare({"v1": synthetic_v1})
result_v2 = evaluator.compare({"v2": synthetic_v2})
```

### `Population`

For direct access to the underlying data structure:

```python
from acteval import Population

pop = Population(observed)
print(pop.acts)       # activity labels per episode
print(pop.durations)  # durations as numpy array
```
