from acteval.evaluate import Evaluator, compare


def test_compare_single_df(observed, synthetic):
    result = compare(observed, synthetic, report_stats=False)
    expected_keys = {
        "descriptions",
        "group_descriptions",
        "domain_descriptions",
        "distances",
        "group_distances",
        "domain_distances",
    }
    assert set(result.keys()) == expected_keys


def test_compare_dict(observed, synthetic):
    result = compare(
        observed, {"m1": synthetic, "m2": synthetic}, report_stats=False
    )
    # both model names should appear in output columns
    for frame in result.values():
        cols = set(frame.columns)
        assert "m1" in cols or any("m1" in c for c in cols)
        assert "m2" in cols or any("m2" in c for c in cols)


def test_evaluator_caches(observed, synthetic):
    evaluator = Evaluator(observed)
    result1 = evaluator.compare({"m": synthetic}, report_stats=False)
    result2 = evaluator.compare({"m": synthetic}, report_stats=False)
    for key in result1:
        assert result1[key].equals(result2[key]), f"Mismatch in {key}"


def test_evaluator_reuse_is_independent(observed, synthetic):
    evaluator = Evaluator(observed)
    result1 = evaluator.compare({"a": synthetic}, report_stats=False)
    result2 = evaluator.compare({"b": synthetic}, report_stats=False)
    assert list(result1.model_names) == ["a"]
    assert list(result2.model_names) == ["b"]
