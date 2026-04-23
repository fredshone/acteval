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
