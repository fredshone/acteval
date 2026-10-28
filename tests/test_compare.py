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


def test_segment_weights_count_occurrences_for_every_feature_layout():
    # 60 people, each home twice and work or shop once.
    acts = [["home", "work", "home"], ["home", "shop", "home"]] * 30
    result = compare(_schedules(acts), _schedules(acts))
    target = result.target_distance_weights.droplevel(["label", "cat"]).sort_index()
    # count features: occurrences summed over people
    assert target.loc[("participations", "participation rate", "home")] == 120
    assert target.loc[("participations", "participation rate", "work")] == 30
    assert target.loc[("transitions", "2-gram", "home>work")] == 30
    # timing: one value per episode
    assert target.loc[("timing", "start times", "home0")] == 60
    assert target.loc[("timing", "start times", "work0")] == 30
    # one per person
    assert target.loc[("participations", "lengths", "sequence lengths")] == 60
    assert target.loc[("creativity", "homogeneity", "all")] == 60


def _schedules(acts_per_pid):
    rows = []
    for pid, acts in enumerate(acts_per_pid):
        step = 1440 // len(acts)
        for i, act in enumerate(acts):
            end = 1440 if i == len(acts) - 1 else (i + 1) * step
            rows.append((pid, act, i * step, end))
    return DataFrame(rows, columns=["pid", "act", "start", "end"])


def test_rare_invented_ngrams_barely_dilute_transitions():
    # Both models make the same errors on the observed n-grams (55/45 instead
    # of 50/50 work/shop); "novel" also extends 3 schedules with invented
    # activities. Weighted by people, those invented n-grams cut its distance
    # by 20-27%; weighted by occurrences they carry weight 3 against ~50 per
    # observed n-gram, so they can only nudge it.
    hwh, hsh = ["home", "work", "home"], ["home", "shop", "home"]
    extended = hwh + ["escort", "medical", "escort", "home"]
    observed = _schedules([hwh] * 50 + [hsh] * 50)
    conservative = _schedules([hwh] * 55 + [hsh] * 45)
    novel = _schedules([extended] * 3 + [hwh] * 52 + [hsh] * 45)
    result = compare(observed, {"conservative": conservative, "novel": novel})

    weights = result.raw["distances"].weights["novel"].sort_index()
    assert weights.loc[("transitions", "2-gram", "escort>medical")].item() == 3
    # the 3 extended schedules also contain home>work
    assert weights.loc[("transitions", "2-gram", "home>work")].item() == 55

    groups = result.groups.combined.distances
    for feature in ("2-gram", "3-gram"):
        row = groups.loc[("transitions", feature)]
        assert row["conservative"] == pytest.approx(0.05)
        assert row["novel"] >= 0.95 * row["conservative"]
