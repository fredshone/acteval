from pandas import DataFrame

from acteval.evaluate import compare


def _observed():
    # Two people: pid 0 has work, pid 1 does not — 50% participation rate
    return DataFrame([
        {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
        {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
        {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
        {"pid": 1, "act": "home", "start": 0, "end": 24, "duration": 24},
    ])


def _synthetic():
    # No work activity — has shop instead
    return DataFrame([
        {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
        {"pid": 0, "act": "shop", "start": 6, "end": 14, "duration": 8},
        {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
    ])


def test_timing_distance_for_absent_activity_is_one():
    result = compare(_observed(), {"m": _synthetic()})
    dists = result.features.combined.distances
    timing_dists = dists.xs("timing", level="domain")
    start_dists = timing_dists.xs("start times", level="feature")
    work_rows = start_dists[start_dists.index.str.startswith("work")]
    assert len(work_rows) > 0, "Expected work* rows in start times"
    assert (work_rows["m"] == 1.0).all()


def test_timing_description_for_absent_activity_is_zero():
    # After weighted aggregation, absent activities have zero weight so their
    # description collapses to 0.0 (not NaN) in the aggregated output.
    result = compare(_observed(), {"m": _synthetic()})
    descs = result.features.combined.descriptions
    timing_descs = descs.xs("timing", level="domain")
    start_descs = timing_descs.xs("start times", level="feature")
    work_rows = start_descs[start_descs.index.str.startswith("work")]
    assert len(work_rows) > 0, "Expected work* rows in start times"
    assert (work_rows["m"] == 0.0).all()


def test_participation_rate_distance_for_absent_activity_not_one():
    result = compare(_observed(), {"m": _synthetic()})
    dists = result.features.combined.distances
    part_dists = dists.xs("participations", level="domain")
    rate_dists = part_dists.xs("participation rate", level="feature")
    work_rows = rate_dists[rate_dists.index.str.startswith("work")]
    assert len(work_rows) > 0, "Expected work rows in participation rate"
    assert (work_rows["m"] != 1.0).all()
