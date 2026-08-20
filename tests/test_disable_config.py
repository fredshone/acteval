import pytest

from acteval._jobs import apply_overrides, get_jobs, load_config
from acteval.evaluate import Evaluator, compare


def test_apply_overrides_disables_one_nested_key():
    cfg = load_config()
    assert cfg["jobs"]["creativity"]["novelty"] is True
    overridden = apply_overrides(cfg, ["jobs.creativity.novelty"])
    assert overridden["jobs"]["creativity"]["novelty"] is False
    # diversity, and the original cfg, are untouched
    assert overridden["jobs"]["creativity"]["diversity"] is True
    assert cfg["jobs"]["creativity"]["novelty"] is True


def test_apply_overrides_disables_multiple_keys():
    cfg = load_config()
    overridden = apply_overrides(
        cfg, ["jobs.creativity.novelty", "jobs.transitions.4-gram"]
    )
    assert overridden["jobs"]["creativity"]["novelty"] is False
    assert overridden["jobs"]["transitions"]["4-gram"] is False
    assert overridden["jobs"]["transitions"]["3-gram"] is True


def test_apply_overrides_unknown_key_raises_with_valid_keys_listed():
    cfg = load_config()
    with pytest.raises(ValueError) as exc_info:
        apply_overrides(cfg, ["jobs.creativity.not_a_real_key"])
    message = str(exc_info.value)
    assert "jobs.creativity.not_a_real_key" in message
    assert "jobs.creativity.novelty" in message


def test_apply_overrides_unknown_section_raises():
    cfg = load_config()
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["jobs.not_a_real_section.novelty"])


def test_get_jobs_disable_turns_off_creativity_novelty():
    jobs = get_jobs(disable=["jobs.creativity.novelty"])
    assert jobs.creativity.novelty is False
    assert jobs.creativity.diversity is True
    assert jobs.creativity.enabled is True  # diversity still on


def test_get_jobs_disable_turns_off_transitions_ngram():
    jobs = get_jobs(disable=["jobs.transitions.4-gram"])
    names = {(spec.domain, spec.name) for spec in jobs.density}
    assert ("transitions", "4-gram") not in names
    assert ("transitions", "3-gram") in names


def test_evaluator_disable_kwarg(observed, synthetic):
    evaluator = Evaluator(observed, disable=["jobs.creativity.novelty"])
    assert evaluator._jobs.creativity.novelty is False
    result = evaluator.compare({"m": synthetic})
    desc_features = result.features.combined.descriptions.index.get_level_values(
        "feature"
    )
    dist_features = result.features.combined.distances.index.get_level_values("feature")
    assert "novelty" not in desc_features
    assert "conservatism" not in dist_features
    assert "diversity" in desc_features  # unaffected


def test_compare_disable_kwarg(observed, synthetic):
    result = compare(observed, synthetic, disable=["jobs.feasibility.home_based"])
    features = result.features.combined.distances.index.get_level_values("feature")
    assert not any(f.startswith("not home based") for f in features)
    assert any(f.startswith("consecutive") for f in features)  # unaffected


def test_compare_disable_unknown_key_raises(observed, synthetic):
    with pytest.raises(ValueError):
        compare(observed, synthetic, disable=["jobs.not_real"])
