"""Tests for PidFeatures and feature functions.

Verifies:
- ``fn(pop).subset(pids).aggregate()`` equals ``fn(Population(filtered_df)).aggregate()``
- ``Evaluator.compare_splits`` produces identical output to old ``subsample_and_evaluate``
"""

import numpy as np
from pandas import DataFrame

from acteval.density.features import participation, times
from acteval.density.features.transitions import ngrams_per_pid
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
    expected = times.start_times_by_act_plan_enum_per_pid(Population(sub_df)).aggregate()
    pid_feat = times.start_times_by_act_plan_enum_per_pid(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "start_times subset")


def test_durations_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([1])]
    expected = times.durations_by_act_plan_enum_per_pid(Population(sub_df)).aggregate()
    pid_feat = times.durations_by_act_plan_enum_per_pid(pop)
    actual = pid_feat.subset(np.array([1])).aggregate()
    _assert_features_subset_equal(expected, actual, "durations subset")


def test_participation_rates_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 1])]
    expected = participation.participation_rates_by_act_per_pid(Population(sub_df)).aggregate()
    pid_feat = participation.participation_rates_by_act_per_pid(pop)
    actual = pid_feat.subset(np.array([0, 1])).aggregate()
    _assert_features_subset_equal(expected, actual, "participation subset")


def test_sequence_lengths_subset():
    pop, df = _pop()
    sub_df = df[df.pid.isin([0, 2])]
    expected = structural.sequence_lengths_per_pid(Population(sub_df)).aggregate()
    pid_feat = structural.sequence_lengths_per_pid(pop)
    actual = pid_feat.subset(np.array([0, 2])).aggregate()
    _assert_features_subset_equal(expected, actual, "sequence_lengths subset")


def test_transitions_subset():
    pop, df = _pop()
    # Use min_count=0 so all n-grams are included
    sub_df = df[df.pid.isin([0, 1])]
    expected = ngrams_per_pid(Population(sub_df), n=2, min_count=0).aggregate()
    pid_feat = ngrams_per_pid(pop, n=2, min_count=0)
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


def test_compare_splits_two_models():
    observed, synthetic, target_attrs, synth_attrs = _split_data()
    synthetic2 = synthetic.copy()
    result = compare_splits(
        observed=observed,
        synthetic_schedules={"m1": synthetic, "m2": synthetic2},
        synthetic_attributes={"m1": synth_attrs, "m2": synth_attrs},
        target_attributes=target_attrs,
        split_on=["gender"],
        report_stats=False,
    )
    assert "descriptions" in result
    assert "label_group_distances" in result
    # Both models should appear as columns
    for frame in result.values():
        cols = list(frame.columns)
        assert any("m1" in c for c in cols), f"m1 missing from {cols}"
        assert any("m2" in c for c in cols), f"m2 missing from {cols}"


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


def test_compare_splits_precomputed_matches_original():
    """Precomputed synthetic features must give numerically identical results."""
    import pandas as pd
    from pandas import MultiIndex, concat

    from acteval.evaluate import (
        _precompute_pid_features,
        _subset_pid_features,
        describe,
        describe_labels,
        process_metrics,
    )

    observed, synthetic, target_attrs, synth_attrs = _split_data()
    evaluator = Evaluator(observed)

    # Build precomputed synthetic structures
    synthetic_pops = {"m": Population(synthetic)}
    synth_pid_features = _precompute_pid_features(synthetic_pops)

    pairs_cached = []
    pairs_original = []

    for cat in target_attrs["gender"].unique():
        target_pids = target_attrs[target_attrs["gender"] == cat].pid.values
        sub_target = observed[observed.pid.isin(target_pids)]
        sample_pids = synth_attrs[synth_attrs["gender"] == cat].pid.values
        sub_synth = {"m": synthetic[synthetic.pid.isin(sample_pids)]}

        target_dense = evaluator._target_pop.dense_pids_from_original(target_pids)
        cached_target = {
            k: pf.subset(target_dense).aggregate()
            for k, pf in evaluator._target_pid_features.items()
        }
        synth_dense = {
            "m": synthetic_pops["m"].dense_pids_from_original(sample_pids)
        }
        synth_sub_acts = {"m": frozenset(sub_synth["m"]["act"].unique())}
        cached_synth = _subset_pid_features(synth_pid_features, synth_dense, synth_sub_acts)

        # Cached (optimised) path
        desc, dist = process_metrics(
            sub_synth, sub_target,
            cached_features=cached_target,
            cached_synthetic_features=cached_synth,
        )
        for r in (desc, dist):
            r.index = MultiIndex.from_tuples(
                [(*i, "gender", cat) for i in r.index],
                names=list(r.index.names) + ["label", "cat"],
            )
        pairs_cached.append((desc, dist))

        # Original (non-cached) path
        desc2, dist2 = process_metrics(
            sub_synth, sub_target,
            cached_features=cached_target,
        )
        for r in (desc2, dist2):
            r.index = MultiIndex.from_tuples(
                [(*i, "gender", cat) for i in r.index],
                names=list(r.index.names) + ["label", "cat"],
            )
        pairs_original.append((desc2, dist2))

    descriptions_cached = concat([d for d, _ in pairs_cached])
    distances_cached = concat([d for _, d in pairs_cached])
    descriptions_orig = concat([d for d, _ in pairs_original])
    distances_orig = concat([d for _, d in pairs_original])

    pd.testing.assert_frame_equal(
        descriptions_cached.sort_index(), descriptions_orig.sort_index()
    )
    pd.testing.assert_frame_equal(
        distances_cached.sort_index(), distances_orig.sort_index()
    )
