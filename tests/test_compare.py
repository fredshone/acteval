import pytest
from pandas import DataFrame

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
