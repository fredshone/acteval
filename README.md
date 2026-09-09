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

result = compare(observed, synthetic)
print(result.summary())
#                 synthetic
# domain
# creativity      0.166667
# feasibility     0.333333
# participations  0.162037
# timing          0.082728
# transitions     0.380952
```

`synthetic` can also be a `{name: DataFrame}` dict to compare several models side-by-side in one call — see [Comparing populations](#comparing-populations).

`result` is an `EvalResult`. See [Reading the results](#reading-the-results) for how to dig deeper — the fast path is `summary()` / `rank_models()` / `best_model`; `result.at(...)` is the one accessor to remember for everything else.

Prefer the command line? Jump to [CLI](#cli) — it wraps the same `compare()` call for CSV/Parquet files without writing Python.

## Input format

Data is passed as a pandas (or [polars](https://pola.rs)) DataFrame with one row per activity episode, in the same shape as `observed`/`synthetic` above:

| column | type | description |
|--------|------|-------------|
| `pid` | int/str | Person identifier |
| `act` | str | Activity label (e.g. `"home"`, `"work"`, `"shop"`) |
| `start` | numeric | Start time (any consistent unit, e.g. hours) |
| `end` | numeric | End time |
| `duration` | numeric | Duration (`end - start`); can be omitted when both `start` and `end` are provided |

Any two of `start`, `end`, and `duration` are sufficient — the third is derived automatically. A polars DataFrame in the same shape works anywhere a pandas one does; it's converted internally.

## API

### Comparing populations

This is the primary workflow — comparing one or more synthetic populations against
observed data. Start with `compare()`; reach for `Evaluator` only once you're
calling it repeatedly against the same observed data.

#### `compare(observed, synthetic, **kwargs)`

`synthetic` can be a single DataFrame, as in Quick start (the result column is
named `"synthetic"`), or a `{name: DataFrame}` dict to compare several models
side-by-side in one call:

```python
result = compare(observed, {"model_a": synthetic_a, "model_b": synthetic_b})
```

`result` is an `EvalResult` object. See [Reading the results](#reading-the-results) for how to access distances and descriptions at feature, group, and domain level.

#### `Evaluator`

Use `Evaluator` when comparing multiple synthetic populations against the same observed data — it computes and caches the observed features once. Each `compare()` call is independent.

```python
from acteval import Evaluator

evaluator = Evaluator(observed)

result_v1 = evaluator.compare({"v1": synthetic_v1})
result_v2 = evaluator.compare({"v2": synthetic_v2})
```

For incremental accumulation — adding one model at a time, e.g. inside a loop with
inspection between models — see `Evaluator.compare_population()` / `.report()` in
the docstrings. Advanced; most users want `compare()` or `Evaluator.compare()`.

Pass `progress=True` to either `compare()` or `Evaluator(...)` to show tqdm
progress bars while features are computed — useful for large populations.

#### Splitting by attribute

Pass `target_attributes` (for `observed`), `attributes` (per synthetic model), and `split_on` to evaluate separately within each category of an attribute (e.g. gender) instead of over the whole population — the Python equivalent of the CLI's `--split-on` flag:

```python
target_attrs = pd.DataFrame({"pid": [0, 1], "gender": ["M", "F"]})
synthetic_attributes = {"my_model": pd.DataFrame({"pid": [0, 1], "gender": ["M", "F"]})}

result = compare(
    observed,
    {"my_model": synthetic},
    attributes=synthetic_attributes,
    target_attributes=target_attrs,
    split_on=["gender"],
)
```

This makes the `by_attribute`/`by_category` splits available at every level via
`result.at(level, split)` — see [Reading the results](#reading-the-results) for how
to read them. Without `split_on`, both raise `SplitNotAvailableError`.

> **Numeric split columns are auto-binned.** If a `split_on` column is numeric
> (float, or an integer with more than 10 unique values), it's automatically
> bucketed into up to 5 ordinal bins (`"lowest"`...`"highest"`) via `pd.qcut`,
> with a `UserWarning` noting the bin edges chosen. Encode the column as a
> categorical/string beforehand (e.g. your own age bands) to control the
> buckets yourself and suppress the warning.

#### Disabling specific metrics

Pass `disable` with a list of dotted `section.key` paths matching `config.toml` to switch off individual metrics without writing a custom config file:

```python
result = compare(
    observed,
    {"my_model": synthetic},
    disable=["jobs.creativity.novelty", "jobs.transitions.4-gram"],
)
```

Call `list_disable_keys()` to see every valid dotted path up front, instead of
reading `config.toml` or triggering the `ValueError` an unknown key raises (which
also lists the valid paths). `disable` also works on `Evaluator(observed,
disable=[...])` and the CLI's `--disable` flag; for anything more involved than a
metric or two, pass a custom `config_path` or a pre-built `jobs` (`EvalConfig`)
instead.

#### Comparing against multiple targets

`compare()`/`Evaluator` compare N synthetic models against one target. To compare
the *same* synthetic models against several targets (e.g. several observed
populations) and see them side-by-side, use `acteval.results.compare_many()`:

```python
from acteval.results import compare_many

result = compare_many(
    {"target_a": observed_a, "target_b": observed_b},
    {"model_1": synthetic_1, "model_2": synthetic_2},
)
```

This runs one ordinary `compare()` call per target and merges the results into a
single `EvalResult`, with model columns renamed `"{target_name}::{model_name}"` so
nothing collides:

```python
print(result.model_names)
# ['target_a::model_1', 'target_a::model_2', 'target_b::model_1', 'target_b::model_2']

print(result.rank_models())
# target_b::model_1    0.086425
# target_a::model_1    0.188905
# target_a::model_2    0.695448
# target_b::model_2    0.702917
# dtype: float64
```

For per-target attributes/`split_on`, or to inspect intermediate per-target
results before merging, call `compare()` yourself in a loop and pass the results
to `acteval.results.combine()`:

```python
from acteval.results import combine

results = {
    "target_a": compare(observed_a, synthetic),
    "target_b": compare(observed_b, synthetic),
}
result = combine(results)
```

> **Known limitation:** the shared `target`/`unit` values underlying feature →
> group → domain aggregation are taken from the *first* result only, so
> re-aggregating a non-first target's columns uses that first target's weights
> rather than its own. Each model's distances are still computed against its own
> target — this only affects aggregation weighting, and will be resolved by a
> future refactor to carry one weight base per source.

### Other entry points

`compare()`/`Evaluator` cover population-level evaluation — the thing most users
want. These are separate, optional tools for other use cases.

#### `pairwise_distances(schedules, specs=None)`

Compute a single NxN distance matrix between individual schedules. Useful for clustering, outlier detection, or directly comparing a small batch of schedules — a standalone code path, independent of `compare()`/`Evaluator`/`config.toml`.

```python
from acteval import pairwise_distances

result = pairwise_distances(schedules)
result.matrix          # numpy array, shape (N, N)
result.pids            # original pid values, length N
df = result.to_dataframe()  # labeled DataFrame, original pid values as index/columns
```

The result matrix is symmetric with zeros on the diagonal. All values are in **0–1**.

By default, three equal-weight semantic-distance specs are used (participations,
transitions, timing via MAE). Pass a custom `specs` list to change the metrics or
their relative weights, e.g. `pairwise_distances(schedules, specs=[chamfer_spec()])`.
Each spec defines a `feature_fn` (extracts a `(N, ...)` array from the population)
and a `distance_fn` (computes the `(N, N)` matrix); the final matrix is a weighted
average across all active specs. See `acteval.pairwise` for the available factories —
`default_pairwise_specs()`, `chamfer_spec(max_len, weight)`, `soft_dtw_spec(max_len, gamma, weight)`.

`Population` is the internal numpy-precomputation layer `compare()`/`pairwise_distances()` are built on — most users never construct it directly; see its docstring (`acteval.population.Population`) if you need it.

#### `acteval.plot`

Matplotlib plotting helpers for interactive/notebook use — Gantt charts, time-use
and participation-rate breakdowns, bigram heatmaps, start/end/duration histogram
grids, sequence-probability waterfalls, and heatmap/bar-chart views of a
`compare()` result. Not imported by `compare()`/`Evaluator`/`pairwise_distances`,
so it never runs unless you call into it. Submodules: `frequency`, `times`,
`transitions`, `plot` (schedule-level charts), `results` (result-level charts) —
see each module's docstring for the full function list.

```python
from acteval.plot.plot import gantt

gantt(observed)
```

## Reading the results

### The fast path: `summary()`, `rank_models()`, `best_model`

For comparing models against each other, these three are usually all you need:

```python
# df_a is the Quick start `synthetic`; df_b is a deliberately bad model
# (everyone at "work" all day) to make the comparison obvious.
df_a = synthetic
df_b = pd.DataFrame([
    {"pid": 0, "act": "work", "start": 0, "end": 24, "duration": 24},
    {"pid": 1, "act": "work", "start": 0, "end": 24, "duration": 24},
    {"pid": 2, "act": "work", "start": 0, "end": 24, "duration": 24},
])
result = compare(observed, {"model_a": df_a, "model_b": df_b})

# Mean domain distance per model (lower is better)
print(result.rank_models())
# model_a    0.225144
# model_b    0.698380
# dtype: float64

# Best model
print(result.best_model)   # "model_a"

# Domain-level summary table
print(result.summary())
#                   model_a   model_b
# domain
# creativity       0.166667  0.333333
# feasibility      0.333333  1.000000
# participations   0.162037  0.988889
# timing           0.082728  0.669676
# transitions      0.380952  0.500000
```

Save all levels to CSV at once with `result.save("output_dir/")`.

### The one thing to remember: `result.at(level, split)`

For anything more detailed than the summary table — a specific aggregation level, or a specific split — `result.at(...)` is the one accessor to remember:

```python
result.at()                              # domains × combined (result.at().distances == result.summary())
result.at("groups")                      # groups × combined
result.at("features", "by_attribute")    # features × by_attribute (requires split_on)
result.at("domains", "by_category")      # domains × by_category   (requires split_on)
```

`level` is one of `"features"`, `"groups"`, `"domains"` (most → least granular); `split` is one of `"combined"`, `"by_attribute"`, `"by_category"` (the latter two require `split_on` — see [Splitting by attribute](#splitting-by-attribute)). Each call returns an `AggregatedResult` with `.distances` and `.descriptions` DataFrames — the former is what feeds `summary()`/`rank_models()`, the latter carries descriptive stats (e.g. average start time) at the same index. Passing anything else raises `ValueError` listing the allowed values.

`result.at(level, split)` is a thin dispatcher over chained properties of the same names — `result.at("groups", "by_attribute")` and `result.groups.by_attribute` return the exact same object, so use whichever reads better at the call site. `.descriptions` always includes a `"target"` column alongside each model's, showing the observed population's own value for comparison. `result.raw` exposes the pre-aggregation data (one `ResultFrame` each for descriptions and distances) that every level above is aggregated from — `distances` covers models only (there's no such thing as the target's distance to itself); pair it with `result.target_distance_weights` if you're building custom distance aggregations of your own.

Distances are in the range **0–1** (lower is better). A distance of `0.0` means the synthetic distribution perfectly matches observed; `1.0` is the maximum penalty.

> **Note on timing features:** A distance of `1.0` for a timing feature means the activity is *entirely absent* from the synthetic population — not just timed differently. This is treated as a maximum-penalty missing feature rather than a distributional difference.

### Evaluation domains

| Domain | What it measures | `disable=[...]` prefix |
|--------|-----------------|-------------------------|
| `participations` | Who does what and how often — participation rates, joint participation, sequence lengths | `jobs.participations.*` |
| `transitions` | Activity sequences — 2-, 3-, and 4-gram transition patterns | `jobs.transitions.*` |
| `timing` | When and how long — start times, durations, and their joint distributions | `jobs.timing.*` |
| `creativity` | How novel and diverse the synthetic schedules are relative to observed | `jobs.creativity.*` |
| `feasibility` | Structural validity — home-based schedules, no consecutive duplicate activities | `jobs.feasibility.*` |
| `sequences` | Full abbreviated tour-string distributions (e.g. `h>w>h`); off by default | `jobs.sequences.*` |

The rightmost column is what to pass to `disable=[...]` (or `--disable` on the
CLI) to switch off part of a domain — see [Disabling specific
metrics](#disabling-specific-metrics). Call `list_features()` to see every
individual feature computed within each domain (the finer-grained rows behind
`result.features`), or `list_disable_keys()` for every valid `disable=[...]`
path.

## CLI

`acteval` ships with a command-line interface for comparing models without writing
Python. It has two subcommands: `compare` (the primary one) and `filter`.

```
acteval compare TARGET [--target-attrs PATH] -m NAME SCHEDULE [ATTRS] [-m ...] [options]
```

```bash
# Compare one model to observed data
acteval compare observed.csv -m my_model synthetic.csv

# Compare multiple models side-by-side
acteval compare observed.csv -m model_a synthetic_a.csv -m model_b synthetic_b.csv

# Save results to CSV files
acteval compare observed.csv -m my_model synthetic.csv -o results/

# Show group-level detail instead of domain summary
acteval compare observed.csv -m my_model synthetic.csv -l groups

# Use a custom config file
acteval compare observed.csv -m my_model synthetic.csv -c custom.toml

# Disable specific metrics without a custom config file
acteval compare observed.csv -m my_model synthetic.csv --disable jobs.creativity.novelty
```

Input files can be CSV or Parquet (detected by extension). Run `acteval compare --help` for the full option list.

#### Advanced: splitting by attribute and batch mode

```bash
# Split evaluation by attribute (e.g. gender)
# Per-model attrs are the third argument to -m
# Attributes must be provided for the target and ALL models, or not at all
acteval compare observed.csv --target-attrs target_attrs.csv \
  -m model_a synthetic_a.csv synth_attrs_a.csv \
  -m model_b synthetic_b.csv synth_attrs_b.csv \
  --split-on gender

# Batch mode: auto-discover model subdirectories
# Each subdir becomes a model (name = dir name); schedule and attrs files are
# classified by their columns (pid + act → schedule; pid + other cols → attrs)
acteval compare observed.csv --batch models/

# Batch mode with attribute splitting
acteval compare observed.csv --target-attrs target_attrs.csv --batch models/ --split-on gender
```

**Attribute rules:**
- Attributes must be provided for **all** inputs (target + every model) or **none**. Partial specification raises an error.
- `--split-on` and `--target-attrs` must be specified together.
- In batch mode, if any model subdirectory contains an attributes file, all subdirectories must contain one.

**Batch directory layout:**
```
models/
  model_a/
    schedules.csv      ← has pid, act → classified as schedule
    attributes.csv     ← has pid + other cols, no act → classified as attrs
  model_b/
    output.parquet
```

### `acteval filter`

Filter a schedule file down to persons with a specific structural issue — useful for
spot-checking a synthetic population before running `compare`.

```bash
# Schedules that don't start and end at home
acteval filter non-home-based synthetic.csv -o flagged.csv

# Schedules with consecutive duplicate activities (default: home, work, education)
acteval filter consecutive synthetic.csv --act home shop
```

Without `-o/--output`, filtered rows are printed to stdout as CSV. Run
`acteval filter --help` for the full option list.

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
