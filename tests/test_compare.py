from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

import acteval
from acteval.evaluate import Evaluator, compare


def test_compare_single_df(observed, synthetic):
    result = compare(observed, synthetic)
    # Three schedule levels, each with combined view
    assert result.features.combined.distances.index.names == [
        "domain",
        "feature",
        "segment",
    ]
    assert result.groups.combined.distances.index.names == ["domain", "feature"]
    assert result.domains.combined.distances.index.names == ["domain"]


def test_compare_dict(observed, synthetic):
    result = compare(observed, {"m1": synthetic, "m2": synthetic})
    for view in (result.features, result.groups, result.domains):
        cols = list(view.combined.distances.columns)
        assert any("m1" in c for c in cols)
        assert any("m2" in c for c in cols)


def test_evaluator_caches(observed, synthetic):
    evaluator = Evaluator(observed)
    result1 = evaluator.compare({"m": synthetic})
    result2 = evaluator.compare({"m": synthetic})
    assert result1.domains.combined.distances.equals(result2.domains.combined.distances)
    assert result1.features.combined.distances.equals(
        result2.features.combined.distances
    )


def test_evaluator_reuse_is_independent(observed, synthetic):
    evaluator = Evaluator(observed)
    result1 = evaluator.compare({"a": synthetic})
    result2 = evaluator.compare({"b": synthetic})
    assert list(result1.model_names) == ["a"]
    assert list(result2.model_names) == ["b"]


def test_at_default_matches_domains_combined(observed, synthetic):
    result = compare(observed, synthetic)
    assert result.at().distances.equals(result.domains.combined.distances)


@pytest.mark.parametrize(
    "level,attr",
    [("features", "features"), ("groups", "groups"), ("domains", "domains")],
)
def test_at_level_matches_property(observed, synthetic, level, attr):
    result = compare(observed, synthetic)
    assert result.at(level=level).distances.equals(
        getattr(result, attr).combined.distances
    )


def test_at_invalid_level_raises_value_error(observed, synthetic):
    result = compare(observed, synthetic)
    with pytest.raises(ValueError, match="level"):
        result.at(level="not_a_level")


def test_at_invalid_split_raises_value_error(observed, synthetic):
    result = compare(observed, synthetic)
    with pytest.raises(ValueError, match="split"):
        result.at(split="not_a_split")


def test_models_with_different_activity_coverage_have_no_nan_distances(
    observed, synthetic
):
    """One model does an activity absent from both the target and the other
    model — the resulting asymmetric weight coverage must not produce NaN."""
    other = DataFrame(
        [
            {"pid": 0, "act": "home", "start": 0, "end": 6, "duration": 6},
            {"pid": 0, "act": "leisure", "start": 6, "end": 14, "duration": 8},
            {"pid": 0, "act": "home", "start": 14, "end": 24, "duration": 10},
            {"pid": 1, "act": "home", "start": 0, "end": 12, "duration": 12},
            {"pid": 1, "act": "work", "start": 12, "end": 24, "duration": 12},
        ]
    )
    result = compare(observed, {"m1": synthetic, "m2": other})
    # Aggregated (group/domain) distances resolve to real numbers via
    # zero-weighted rows, not NaN. The raw per-segment "features" level may
    # still show NaN for a segment a given model has literally no data for
    # (e.g. m1's "home+leisure" pair rate) — that's informational, not a bug.
    assert not result.domains.combined.distances.isna().any().any()
    assert not result.groups.combined.distances.drop(columns="unit").isna().any().any()


@pytest.mark.parametrize("odd_act", ["work2", "a>b", "c+d"])
def test_identical_populations_score_zero_with_unusual_act_names(odd_act, tmp_path):
    # Activity names with trailing digits, ">" or "+" used to be mis-parsed
    # out of feature keys, so their segments looked missing from the model.
    config = (Path(acteval.__file__).parent / "config.toml").read_text()
    config = config.replace("seq_rates   = false", "seq_rates   = true")
    config = config.replace("full_sequences = false", "full_sequences = true")
    config_path = tmp_path / "config.toml"
    config_path.write_text(config)

    rows = []
    for pid in range(20):
        shift = 15 * (pid % 4)
        rows += [
            {"pid": pid, "act": "home", "start": 0, "end": 480 + shift},
            {"pid": pid, "act": odd_act, "start": 480 + shift, "end": 1020},
            {"pid": pid, "act": "home", "start": 1020, "end": 1440},
        ]
    df = DataFrame(rows)
    result = compare(df, {"m": df.copy()}, config_path=config_path)
    distances = result.features.combined.distances["m"]
    density = distances.drop(["creativity", "feasibility"], level="domain")
    assert (density == 0).all(), density[density != 0]


@pytest.mark.parametrize("name", ["target", "unit"])
def test_reserved_model_names_are_rejected(name, observed, synthetic):
    with pytest.raises(ValueError, match=f"'{name}' is reserved"):
        compare(observed, {name: synthetic})
    with pytest.raises(ValueError, match=f"'{name}' is reserved"):
        Evaluator(observed).compare_population(name, synthetic)


def test_segments_missing_from_a_slice():
    # Slice "a" does home/work, slice "b" home/shop, in both populations; the
    # model also adds "gym" to slice "a" only.
    def schedules(pids, act, extra=None):
        rows = []
        for pid in pids:
            acts = ["home", act] + ([extra] if extra else []) + ["home"]
            bounds = np.linspace(0, 1440, len(acts) + 1)
            rows += [
                {"pid": pid, "act": a, "start": s, "end": e}
                for a, s, e in zip(acts, bounds[:-1], bounds[1:])
            ]
        return rows

    target = DataFrame(schedules(range(10), "work") + schedules(range(10, 16), "shop"))
    model = DataFrame(
        schedules(range(8), "work", "gym") + schedules(range(8, 12), "shop")
    )
    attrs = lambda n_a, n_b: DataFrame(  # noqa: E731
        {"pid": range(n_a + n_b), "g": ["a"] * n_a + ["b"] * n_b}
    )
    result = compare(
        target,
        {"m": model},
        target_attributes=attrs(10, 6),
        synthetic_attributes={"m": attrs(8, 4)},
        split_on=["g"],
    )
    distances = result.raw["distances"]
    rate = (
        distances.values["m"].sort_index().loc[("participations", "participation rate")]
    )
    segments = {
        cat: set(rate.xs(cat, level="cat").index.get_level_values(0)) for cat in "ab"
    }
    # Missing on both sides: no row. Missing on one side: still scored.
    assert segments == {"a": {"home", "work", "gym"}, "b": {"home", "shop"}}
    assert rate[("gym", "g", "a")] == 1.0  # model-only: gym count vs zero

    # Weights are the slice's head counts, not the whole population's, with
    # the model's rescaled to the target's slice size (8 model people → 10).
    target_w = result.target_distance_weights.sort_index().loc[
        ("participations", "participation rate")
    ]
    model_w = (
        distances.weights["m"]
        .sort_index()
        .loc[("participations", "participation rate")]
    )
    assert target_w[("work", "g", "a")] == 10
    assert model_w[("work", "g", "a")] == 10
    assert target_w[("gym", "g", "a")] == 0
    assert model_w[("gym", "g", "a")] == 10
    assert target_w[("shop", "g", "b")] == 6
    assert model_w[("shop", "g", "b")] == 6


def test_distance_weights_use_max_of_both_sides():
    from acteval._result_frame import ResultFrame

    idx = pd.MultiIndex.from_tuples(
        [("d", "f", "x"), ("d", "f", "y"), ("d", "f", "z")],
        names=["domain", "feature", "segment"],
    )
    rf = ResultFrame(
        values=DataFrame({"m": [0.0, 1.0, 0.5]}, index=idx),
        weights=DataFrame({"m": [10.0, 0.0, 4.0]}, index=idx),
        units=pd.Series("EMD", index=idx),
    )
    target = pd.Series([10.0, 10.0, 2.0], index=idx)
    agg = rf.aggregate_distances(["domain", "feature"], target_weights=target)
    # weights: max(10,10)=10, max(0,10)=10 (target-only keeps full weight), max(4,2)=4
    assert agg.values["m"].iloc[0] == pytest.approx((0 * 10 + 1 * 10 + 0.5 * 4) / 24)
    assert agg.weights["m"].iloc[0] == 24


def _novel_rf(values, model_w, target_w, segments):
    from acteval._result_frame import ResultFrame

    idx = pd.MultiIndex.from_tuples(
        [("d", "f", s) for s in segments], names=["domain", "feature", "segment"]
    )
    rf = ResultFrame(
        values=DataFrame({"m": values}, index=idx),
        weights=DataFrame({"m": model_w}, index=idx),
        units=pd.Series("EMD", index=idx),
    )
    return rf, pd.Series(target_w, index=idx, dtype=float)


def test_model_only_segments_cannot_dilute_distance():
    # Two target segments, then add many low-distance model-only segments:
    # the group distance must not drop below the target-only distance.
    base_rf, base_t = _novel_rf([0.2, 0.4], [10.0, 10.0], [10.0, 10.0], ["x", "y"])
    base = base_rf.aggregate_distances(["domain", "feature"], target_weights=base_t)
    novel = [f"n{i}" for i in range(20)]
    rf, t = _novel_rf(
        [0.2, 0.4] + [0.01] * 20,
        [10.0, 10.0] + [10.0] * 20,
        [10.0, 10.0] + [0.0] * 20,
        ["x", "y"] + novel,
    )
    agg = rf.aggregate_distances(["domain", "feature"], target_weights=t)
    assert base.values["m"].iloc[0] == pytest.approx(0.3)
    # model-only rows add to the numerator only: (2 + 4 + 20 * 0.1) / 20
    assert agg.values["m"].iloc[0] == pytest.approx((2 + 4 + 2) / 20)
    assert agg.values["m"].iloc[0] > base.values["m"].iloc[0]
    assert agg.weights["m"].iloc[0] == 220


def test_group_without_target_rows_falls_back_to_weighted_mean():
    rf, t = _novel_rf([0.1, 0.3], [10.0, 30.0], [0.0, 0.0], ["x", "y"])
    group = rf.aggregate_distances(["domain", "feature"], target_weights=t)
    assert group.values["m"].iloc[0] == pytest.approx((1 + 9) / 40)
    # feature level: each model-only segment reports its own distance
    seg = rf.aggregate_distances(["domain", "feature", "segment"], target_weights=t)
    assert list(seg.values["m"]) == pytest.approx([0.1, 0.3])


def test_group_with_no_rows_weighted_is_zero():
    rf, t = _novel_rf([0.0], [0.0], [0.0], ["x"])
    agg = rf.aggregate_distances(["domain", "feature"], target_weights=t)
    assert agg.values["m"].iloc[0] == 0.0
