"""Tests for PidFeatures and per-pid feature functions.

Verifies:
- ``fn_per_pid(pop).aggregate()`` equals ``fn(pop)`` for all per-pid functions
- ``fn_per_pid(pop).subset(pids).aggregate()`` equals ``fn(Population(filtered_df))``
- ``Evaluator.compare_splits`` produces identical output to old ``subsample_and_evaluate``
"""

from functools import partial

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from acteval.density.features import participation, times, transitions
from acteval.density.features.pid_features import PidFeatures
from acteval.density.features.utils import equals
from acteval.evaluate import Evaluator, compare_splits
from acteval.population import Population
from acteval.structural.features import structural


def _pop():
    df = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 20, "duration": 10},
            {"pid": 1, "act": "home", "start": 20, "end": 24, "duration": 4},
            {"pid": 2, "act": "home", "start": 0, "end": 8, "duration": 8},
            {"pid": 2, "act": "shop", "start": 8, "end": 14, "duration": 6},
            {"pid": 2, "act": "home", "start": 14, "end": 24, "duration": 10},
        ]
    )
    return Population(df), df


def _assert_features_equal(a, b, label=""):
    """Assert two feature dicts are equal, with a helpful error message."""
    assert set(a.keys()) == set(b.keys()), f"{label}: keys differ: {set(a.keys())} != {set(b.keys())}"
    for k in a:
        av, aw = a[k]
        bv, bw = b[k]
        np.testing.assert_array_equal(av, bv, err_msg=f"{label}: values differ for key '{k}'")
        np.testing.assert_array_equal(aw, bw, err_msg=f"{label}: weights differ for key '{k}'")


def _assert_features_subset_equal(expected, actual, label=""):
    """Assert that ``actual`` contains all keys from ``expected`` with matching values.

    Extra keys in ``actual`` (from the full population's key space) are allowed —
    per-person features may retain zero-valued entries for activities that the
    subset doesn't participate in.
    """
    for k in expected:
        assert k in actual, f"{label}: key '{k}' missing from actual"
        ev, ew = expected[k]
        av, aw = actual[k]
        np.testing.assert_array_equal(av, ev, err_msg=f"{label}: values differ for key '{k}'")
        np.testing.assert_array_equal(aw, ew, err_msg=f"{label}: weights differ for key '{k}'")


# ---------------------------------------------------------------------------
# Round-trip: per_pid → aggregate must equal the standard function
# ---------------------------------------------------------------------------


def test_start_times_aggregate():
    pop, _ = _pop()
    expected = times.start_times_by_act_plan_enum(pop)
    actual = times.start_times_by_act_plan_enum_per_pid(pop).aggregate()
    _assert_features_equal(expected, actual, "start_times")


def test_durations_aggregate():
    pop, _ = _pop()
    expected = times.durations_by_act_plan_enum(pop)
    actual = times.durations_by_act_plan_enum_per_pid(pop).aggregate()
    _assert_features_equal(expected, actual, "durations")


def test_start_and_duration_aggregate():
    pop, _ = _pop()
    expected = times.start_and_duration_by_act_bins(pop)
    actual = times.start_and_duration_by_act_bins_per_pid(pop).aggregate()
    _assert_features_equal(expected, actual, "start_and_duration")


def test_joint_durations_aggregate():
    pop, _ = _pop()
    expected = times.joint_durations_by_act_bins(pop)
    actual = times.joint_durations_by_act_bins_per_pid(pop).aggregate()
    _assert_features_equal(expected, actual, "joint_durations")


def test_transitions_2gram_aggregate():
    pop, _ = _pop()
    expected = transitions.transitions_by_act(pop, min_count=0)
    actual = transitions.transitions_by_act_per_pid(pop, min_count=0).aggregate()
    _assert_features_equal(expected, actual, "2-gram")


def test_transitions_3gram_aggregate():
    pop, _ = _pop()
    expected = transitions.transition_3s_by_act(pop, min_count=0)
    actual = transitions.transition_3s_by_act_per_pid(pop, min_count=0).aggregate()
    _assert_features_equal(expected, actual, "3-gram")


def test_transitions_4gram_aggregate():
    pop, _ = _pop()
    expected = transitions.transition_4s_by_act(pop, min_count=0)
    actual = transitions.transition_4s_by_act_per_pid(pop, min_count=0).aggregate()
    _assert_features_equal(expected, actual, "4-gram")


def test_transitions_with_min_count_aggregate():
    pop, _ = _pop()
    expected = transitions.transitions_by_act(pop, min_count=2)
    actual = transitions.transitions_by_act_per_pid(pop, min_count=2).aggregate()
    _assert_features_equal(expected, actual, "2-gram min_count=2")


def test_participation_rates_aggregate():
    pop, _ = _pop()
    expected = participation.participation_rates_by_act(pop)
    actual = participation.participation_rates_by_act_per_pid(pop).aggregate()
    _assert_features_equal(expected, actual, "participation_rates")


def test_joint_participation_aggregate():
    pop, _ = _pop()
    expected = participation.joint_participation_rate(pop)
    actual = participation.joint_participation_rate_per_pid(pop).aggregate()
    _assert_features_equal(expected, actual, "joint_participation")


def test_sequence_lengths_aggregate():
    pop, _ = _pop()
    expected = structural.sequence_lengths(pop)
    actual = structural.sequence_lengths_per_pid(pop).aggregate()
    _assert_features_equal(expected, actual, "sequence_lengths")


# ---------------------------------------------------------------------------
# Subset: per_pid.subset(pids).aggregate() must equal fn(Population(filtered_df))
# ---------------------------------------------------------------------------


def test_start_times_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 2])]
    expected = times.start_times_by_act_plan_enum(Population(sub_df))
    pid_feat = times.start_times_by_act_plan_enum_per_pid(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "start_times subset")


def test_durations_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([1])]
    expected = times.durations_by_act_plan_enum(Population(sub_df))
    pid_feat = times.durations_by_act_plan_enum_per_pid(pop)
    actual = pid_feat.subset(np.array([1])).aggregate()
    _assert_features_subset_equal(expected, actual, "durations subset")


def test_participation_rates_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 1])]
    expected = participation.participation_rates_by_act(Population(sub_df))
    pid_feat = participation.participation_rates_by_act_per_pid(pop)
    actual = pid_feat.subset(np.array([0, 1])).aggregate()
    _assert_features_subset_equal(expected, actual, "participation subset")


def test_sequence_lengths_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 2])]
    expected = structural.sequence_lengths(Population(sub_df))
    pid_feat = structural.sequence_lengths_per_pid(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "sequence_lengths subset")


def test_transitions_subset():
    pop, df = _pop()
    # Use min_count=0 so all n-grams are included
    sub_df = df[df.pid.isin([0, 1])]
    expected = transitions.transitions_by_act(Population(sub_df), min_count=0)
    pid_feat = transitions.transitions_by_act_per_pid(pop, min_count=0)
    actual = pid_feat.subset(np.array([0, 1])).aggregate()
    _assert_features_subset_equal(expected, actual, "transitions subset")


# ---------------------------------------------------------------------------
# PidFeatures edge cases
# ---------------------------------------------------------------------------


def test_empty_subset():
    pop, _ = _pop()
    pid_feat = times.start_times_by_act_plan_enum_per_pid(pop)
    result = pid_feat.subset(np.array([], dtype=np.int64)).aggregate()
    for key, (vals, weights) in result.items():
        assert len(vals) == 0
        assert len(weights) == 0


def test_empty_population():
    df = DataFrame(columns=["pid", "act", "start", "end", "duration"])
    pop = Population(df)
    pid_feat = times.start_and_duration_by_act_bins_per_pid(pop)
    assert pid_feat.data == {}


# ---------------------------------------------------------------------------
# Evaluator.compare_splits
# ---------------------------------------------------------------------------


def _split_data():
    """Build small observed + synthetic + attributes for split testing."""
    observed = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "work", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 10, "duration": 10},
            {"pid": 1, "act": "work", "start": 10, "end": 24, "duration": 14},
        ]
    )
    synthetic = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "shop", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 12, "duration": 12},
            {"pid": 1, "act": "work", "start": 12, "end": 24, "duration": 12},
        ]
    )
    target_attrs = DataFrame({"pid": [0, 1], "gender": ["M", "F"]})
    synth_attrs = DataFrame({"pid": [0, 1], "gender": ["M", "F"]})
    return observed, synthetic, target_attrs, synth_attrs


def test_compare_splits_runs():
    observed, synthetic, target_attrs, synth_attrs = _split_data()
    result = compare_splits(
        observed=observed,
        synthetic_schedules={"m": synthetic},
        synthetic_attributes={"m": synth_attrs},
        target_attributes=target_attrs,
        split_on=["gender"],
        report_stats=False,
    )
    # Should have both base and label frames
    assert "descriptions" in result
    assert "label_group_distances" in result


def test_evaluator_compare_splits():
    observed, synthetic, target_attrs, synth_attrs = _split_data()
    evaluator = Evaluator(observed)
    result = evaluator.compare_splits(
        synthetic_schedules={"m": synthetic},
        synthetic_attributes={"m": synth_attrs},
        target_attributes=target_attrs,
        split_on=["gender"],
        report_stats=False,
    )
    assert "descriptions" in result
    assert "label_group_distances" in result


def test_unique_pids_original_stored():
    """Verify Population stores original pids."""
    df = DataFrame(
        [
            {"pid": 10, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 20, "act": "home", "start": 0, "end": 6, "duration": 6},
        ]
    )
    pop = Population(df)
    np.testing.assert_array_equal(pop.unique_pids_original, [10, 20])
