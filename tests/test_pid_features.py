"""Tests for PidFeatures and feature functions.

Verifies:
- ``fn(pop).subset(pids).aggregate()`` equals ``fn(Population(filtered_df)).aggregate()``
- ``Evaluator.compare_splits`` produces identical output to old ``subsample_and_evaluate``
"""

import numpy as np
from pandas import DataFrame

from acteval.evaluate import Evaluator, compare_splits
from acteval.features import participation, structural, times
from acteval.features.transitions import full_sequences, ngrams
from acteval.population import Population


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
    assert set(a.keys()) == set(
        b.keys()
    ), f"{label}: keys differ: {set(a.keys())} != {set(b.keys())}"
    for k in a:
        av, aw = a[k]
        bv, bw = b[k]
        np.testing.assert_array_equal(
            av, bv, err_msg=f"{label}: values differ for key '{k}'"
        )
        np.testing.assert_array_equal(
            aw, bw, err_msg=f"{label}: weights differ for key '{k}'"
        )


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
        np.testing.assert_array_equal(
            av, ev, err_msg=f"{label}: values differ for key '{k}'"
        )
        np.testing.assert_array_equal(
            aw, ew, err_msg=f"{label}: weights differ for key '{k}'"
        )


# ---------------------------------------------------------------------------
# Subset: per_pid.subset(pids).aggregate() must equal fn(Population(filtered_df))
# ---------------------------------------------------------------------------


def test_start_times_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 2])]
    expected = times.start_times_by_act_plan_enum(Population(sub_df)).aggregate()
    pid_feat = times.start_times_by_act_plan_enum(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "start_times subset")


def test_durations_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([1])]
    expected = times.durations_by_act_plan_enum(Population(sub_df)).aggregate()
    pid_feat = times.durations_by_act_plan_enum(pop)
    actual = pid_feat.subset(np.array([1])).aggregate()
    _assert_features_subset_equal(expected, actual, "durations subset")


def test_participation_rates_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 1])]
    expected = participation.participation_rates_by_act(Population(sub_df)).aggregate()
    pid_feat = participation.participation_rates_by_act(pop)
    actual = pid_feat.subset(np.array([0, 1])).aggregate()
    _assert_features_subset_equal(expected, actual, "participation subset")


def test_sequence_lengths_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 2])]
    expected = participation.sequence_lengths(Population(sub_df)).aggregate()
    pid_feat = participation.sequence_lengths(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "sequence_lengths subset")


def test_transitions_subset():
    pop, df = _pop()
    # Use min_count=0 so all n-grams are included
    sub_df = df[df.pid.isin([0, 1])]
    expected = ngrams(Population(sub_df), n=2, min_count=0).aggregate()
    pid_feat = ngrams(pop, n=2, min_count=0)
    actual = pid_feat.subset(np.array([0, 1])).aggregate()
    _assert_features_subset_equal(expected, actual, "transitions subset")


def test_seq_participation_rates_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 2])]
    expected = participation.participation_rates_by_seq_act(
        Population(sub_df)
    ).aggregate()
    pid_feat = participation.participation_rates_by_seq_act(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "seq_participation subset")


def test_enum_participation_rates_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([1, 2])]
    expected = participation.participation_rates_by_act_enum(
        Population(sub_df)
    ).aggregate()
    pid_feat = participation.participation_rates_by_act_enum(pop)
    actual = pid_feat.subset(np.array([1, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "enum_participation subset")


def test_start_times_by_act_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 1])]
    expected = times.start_times_by_act(Population(sub_df)).aggregate()
    pid_feat = times.start_times_by_act(pop)
    actual = pid_feat.subset(np.array([0, 1])).aggregate()
    _assert_features_subset_equal(expected, actual, "start_times_by_act subset")


def test_end_times_by_act_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 2])]
    expected = times.end_times_by_act(Population(sub_df)).aggregate()
    pid_feat = times.end_times_by_act(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "end_times_by_act subset")


def test_durations_by_act_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([1])]
    expected = times.durations_by_act(Population(sub_df)).aggregate()
    pid_feat = times.durations_by_act(pop)
    actual = pid_feat.subset(np.array([1])).aggregate()
    _assert_features_subset_equal(expected, actual, "durations_by_act subset")


def test_time_consistency_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 2])]
    expected = structural.time_consistency(Population(sub_df)).aggregate()
    pid_feat = structural.time_consistency(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "time_consistency subset")


def test_full_sequences_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 1])]
    expected = full_sequences(Population(sub_df)).aggregate()
    pid_feat = full_sequences(pop)
    actual = pid_feat.subset(np.array([0, 1])).aggregate()
    _assert_features_subset_equal(expected, actual, "full_sequences subset")


# ---------------------------------------------------------------------------
# PidFeatures edge cases
# ---------------------------------------------------------------------------


def test_empty_subset():
    pop, _ = _pop()
    pid_feat = times.start_times_by_act_plan_enum(pop)
    result = pid_feat.subset(np.array([], dtype=np.int64)).aggregate()
    for key, (vals, weights) in result.items():
        assert len(vals) == 0
        assert len(weights) == 0


def test_empty_population():
    df = DataFrame(columns=["pid", "act", "start", "end", "duration"])
    pop = Population(df)
    pid_feat = times.start_and_duration_by_act_bins(pop)
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
    )
    assert result.has_splits
    assert result.features.combined.distances.index.names == [
        "domain",
        "feature",
        "segment",
    ]
    assert result.groups.by_attribute.distances.index.names == [
        "domain",
        "feature",
        "label",
    ]


def test_evaluator_compare_splits():
    observed, synthetic, target_attrs, synth_attrs = _split_data()
    evaluator = Evaluator(observed, target_attrs, ["gender"])
    result = evaluator.compare_populations(
        synthetic_schedules={"m": synthetic},
        synthetic_attributes={"m": synth_attrs},
    )
    assert result.has_splits
    assert result.groups.by_attribute.distances.index.names == [
        "domain",
        "feature",
        "label",
    ]


def test_compare_splits_two_models():
    observed, synthetic, target_attrs, synth_attrs = _split_data()
    synthetic2 = synthetic.copy()
    result = compare_splits(
        observed=observed,
        synthetic_schedules={"m1": synthetic, "m2": synthetic2},
        synthetic_attributes={"m1": synth_attrs, "m2": synth_attrs},
        target_attributes=target_attrs,
        split_on=["gender"],
    )
    assert result.has_splits
    for view in (result.features, result.groups, result.domains):
        cols = list(view.combined.distances.columns)
        assert any("m1" in c for c in cols), f"m1 missing from {cols}"
        assert any("m2" in c for c in cols), f"m2 missing from {cols}"


def test_compare_population_and_report():
    observed, synthetic, target_attrs, synth_attrs = _split_data()
    evaluator = Evaluator(observed, target_attrs, ["gender"])
    evaluator.compare_population("m", synthetic, synth_attrs)
    result = evaluator.report()
    assert result.has_splits
    assert result.groups.by_attribute.distances.index.names == [
        "domain",
        "feature",
        "label",
    ]


def test_compare_population_no_attributes_no_splits():
    observed, synthetic, _, _ = _split_data()
    evaluator = Evaluator(observed)
    evaluator.compare_population("m", synthetic)
    result = evaluator.report()
    assert not result.has_splits
    assert "m" in result.features.combined.distances.columns


def test_compare_population_matches_compare_splits():
    observed, synthetic, target_attrs, synth_attrs = _split_data()
    evaluator = Evaluator(observed, target_attrs, ["gender"])

    evaluator.compare_population("m", synthetic, synth_attrs)
    manual_result = evaluator.report()

    split_result = evaluator.compare_populations(
        synthetic_schedules={"m": synthetic},
        synthetic_attributes={"m": synth_attrs},
    )

    for getter in (
        lambda r: r.features.combined.distances,
        lambda r: r.groups.combined.distances,
        lambda r: r.domains.combined.distances,
        lambda r: r.groups.by_attribute.distances,
        lambda r: r.domains.by_attribute.distances,
    ):
        assert getter(manual_result).equals(getter(split_result))


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
