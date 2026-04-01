import warnings

import numpy as np
import pytest
from pandas import DataFrame

from acteval.population import Population
from acteval.evaluate import Evaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schedule(**kwargs):
    row = {"pid": 1, "act": "home", "start": 0.0, "end": 8.0, "duration": 8.0}
    row.update(kwargs)
    return DataFrame([row])


# ---------------------------------------------------------------------------
# Population — pid validation
# ---------------------------------------------------------------------------

def test_population_missing_pid():
    df = DataFrame([{"act": "home", "start": 0, "end": 5, "duration": 5}])
    with pytest.raises(ValueError, match="missing required column 'pid'"):
        Population(df)


def test_population_missing_pid_empty_df():
    df = DataFrame(columns=["act", "start", "end", "duration"])
    with pytest.raises(ValueError, match="missing required column 'pid'"):
        Population(df)


def test_population_nan_pid():
    df = DataFrame([{"pid": None, "act": "home", "start": 0.0, "end": 5.0, "duration": 5.0}])
    with pytest.raises(ValueError, match="'pid' contains NaN"):
        Population(df)


# ---------------------------------------------------------------------------
# Population — timing column dtype / NaN
# ---------------------------------------------------------------------------

def test_population_nonnumeric_start():
    df = _make_schedule(start="8am")
    with pytest.raises(ValueError, match="column 'start' must be numeric"):
        Population(df)


def test_population_nonnumeric_end():
    df = _make_schedule(end="5pm")
    with pytest.raises(ValueError, match="column 'end' must be numeric"):
        Population(df)


def test_population_nonnumeric_duration():
    df = _make_schedule(duration="eight hours")
    with pytest.raises(ValueError, match="column 'duration' must be numeric"):
        Population(df)


def test_population_nan_start():
    df = _make_schedule(start=np.nan)
    with pytest.raises(ValueError, match="column 'start' contains NaN"):
        Population(df)


def test_population_nan_duration():
    df = _make_schedule(duration=np.nan)
    with pytest.raises(ValueError, match="column 'duration' contains NaN"):
        Population(df)


# ---------------------------------------------------------------------------
# Population — consistency check (all three columns present)
# ---------------------------------------------------------------------------

def test_population_inconsistent_timing():
    df = DataFrame([{"pid": 1, "act": "home", "start": 0.0, "end": 8.0, "duration": 5.0}])
    with pytest.raises(ValueError, match="inconsistent"):
        Population(df)


# ---------------------------------------------------------------------------
# Population — start > end
# ---------------------------------------------------------------------------

def test_population_start_after_end():
    df = DataFrame([{"pid": 1, "act": "home", "start": 10.0, "end": 5.0, "duration": -5.0}])
    with pytest.raises(ValueError, match="start > end"):
        Population(df)


def test_population_start_after_end_via_derivation():
    # Only start + duration supplied; negative duration → derived end < start
    df = DataFrame([{"pid": 1, "act": "home", "start": 5.0, "duration": -3.0}])
    with pytest.raises(ValueError, match="start > end"):
        Population(df)


# ---------------------------------------------------------------------------
# Population — derivation correctness
# ---------------------------------------------------------------------------

def test_population_derives_duration():
    df = DataFrame([{"pid": 1, "act": "home", "start": 0.0, "end": 8.0}])
    pop = Population(df)
    assert pop.durations[0] == pytest.approx(8.0)


def test_population_derives_end():
    df = DataFrame([{"pid": 1, "act": "home", "start": 2.0, "duration": 6.0}])
    pop = Population(df)
    assert pop.ends[0] == pytest.approx(8.0)


def test_population_derives_start():
    df = DataFrame([{"pid": 1, "act": "home", "end": 8.0, "duration": 6.0}])
    pop = Population(df)
    assert pop.starts[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Evaluator.__init__ — attributes / split_on validation
# ---------------------------------------------------------------------------

def test_evaluator_attributes_without_split_on(observed):
    attrs = DataFrame({"pid": [0, 1], "gender": ["m", "f"]})
    with pytest.raises(ValueError, match="both be provided or both be None"):
        Evaluator(observed, target_attributes=attrs)


def test_evaluator_split_on_without_attributes(observed):
    with pytest.raises(ValueError, match="both be provided or both be None"):
        Evaluator(observed, split_on=["gender"])


def test_evaluator_attributes_missing_pid(observed):
    attrs = DataFrame({"gender": ["m", "f"]})
    with pytest.raises(ValueError, match="target_attributes.*missing required column 'pid'"):
        Evaluator(observed, target_attributes=attrs, split_on=["gender"])


def test_evaluator_split_column_missing(observed):
    attrs = DataFrame({"pid": [0, 1]})
    with pytest.raises(ValueError, match="not found in target_attributes"):
        Evaluator(observed, target_attributes=attrs, split_on=["gender"])


# ---------------------------------------------------------------------------
# Evaluator.compare_population — attributes validation
# ---------------------------------------------------------------------------

def test_compare_population_attributes_missing_pid(observed, synthetic):
    ev = Evaluator(observed)
    attrs = DataFrame({"__split__": ["all"]})
    with pytest.raises(ValueError, match="missing required column 'pid'"):
        ev.compare_population("m", synthetic, attributes=attrs)


# ---------------------------------------------------------------------------
# Evaluator — numeric split_on column handling
# ---------------------------------------------------------------------------

def _make_numeric_fixtures():
    """Ten-person schedule + matching attributes for numeric binning tests."""
    rows = []
    for pid in range(10):
        rows += [
            {"pid": pid, "act": "home", "start": 0, "end": 8, "duration": 8},
            {"pid": pid, "act": "work", "start": 8, "end": 16, "duration": 8},
            {"pid": pid, "act": "home", "start": 16, "end": 24, "duration": 8},
        ]
    schedule = DataFrame(rows)
    return schedule


def test_float_column_warns_and_bins():
    schedule = _make_numeric_fixtures()
    ages = [20.0, 25.5, 30.0, 35.5, 40.0, 45.5, 50.0, 55.5, 60.0, 65.5]
    attrs = DataFrame({"pid": list(range(10)), "age": ages})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ev = Evaluator(schedule, attrs, ["age"])
    assert any("age" in str(x.message) for x in w), "Expected warning about 'age'"
    assert any(issubclass(x.category, UserWarning) for x in w)
    # Stored attributes should contain ordinal labels, not floats
    stored_vals = set(ev._target_attributes["age"].unique())
    ordinal = {"lowest", "low", "mid", "high", "highest"}
    assert stored_vals.issubset(ordinal), f"Unexpected bin labels: {stored_vals}"


def test_large_int_column_warns_and_bins():
    # Build a 15-person schedule so the int income column has >10 unique values
    rows = []
    for pid in range(15):
        rows += [
            {"pid": pid, "act": "home", "start": 0, "end": 8, "duration": 8},
            {"pid": pid, "act": "work", "start": 8, "end": 16, "duration": 8},
            {"pid": pid, "act": "home", "start": 16, "end": 24, "duration": 8},
        ]
    schedule = DataFrame(rows)
    incomes = [i * 5000 for i in range(1, 16)]  # 15 distinct int values
    attrs = DataFrame({"pid": list(range(15)), "income": incomes})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ev = Evaluator(schedule, attrs, ["income"])
    assert any("income" in str(x.message) for x in w)
    stored_vals = set(ev._target_attributes["income"].unique())
    ordinal = {"lowest", "low", "mid", "high", "highest"}
    assert stored_vals.issubset(ordinal), f"Unexpected bin labels: {stored_vals}"


def test_small_int_column_no_warning():
    schedule = _make_numeric_fixtures()
    # Binary int column — should be treated as categorical, no warning
    attrs = DataFrame({"pid": list(range(10)), "employed": [0, 1] * 5})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Evaluator(schedule, attrs, ["employed"])
    numeric_warns = [x for x in w if issubclass(x.category, UserWarning)]
    assert not numeric_warns, f"Unexpected warnings: {numeric_warns}"


def test_out_of_range_synthetic_gets_label():
    schedule = _make_numeric_fixtures()
    ages = [20.0, 25.5, 30.0, 35.5, 40.0, 45.5, 50.0, 55.5, 60.0, 65.5]
    target_attrs = DataFrame({"pid": list(range(10)), "age": ages})
    # Synthetic person has age=99 — well outside training range
    synth_attrs = DataFrame({"pid": list(range(10)), "age": [99.0] * 10})
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ev = Evaluator(schedule, target_attrs, ["age"])
    synth_binned = ev._apply_numeric_bins(synth_attrs)
    assert "nan" not in synth_binned["age"].str.lower().values, (
        "Out-of-range synthetic age should be assigned a bin, not NaN"
    )
