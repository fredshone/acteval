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
