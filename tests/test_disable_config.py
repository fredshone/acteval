import pytest

from acteval._jobs import apply_overrides, get_jobs, load_config
from acteval.evaluate import Evaluator, compare, compare_grid, compare_many


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


@pytest.fixture
def no_transitions_config(tmp_path):
    cfg = load_config()
    lines = []
    for section, table in [("ngrams", cfg["ngrams"])] + [
        (f"jobs.{name}", table) for name, table in cfg["jobs"].items()
    ]:
        lines.append(f"[{section}]")
        for key, value in table.items():
            if section == "jobs.transitions":
                value = False
            lines.append(f'"{key}" = {str(value).lower()}')
    path = tmp_path / "config.toml"
    path.write_text("\n".join(lines))
    return path


def _domains(result):
    return set(result.domains.combined.distances.index.get_level_values("domain"))


def test_compare_config_path(observed, synthetic, no_transitions_config):
    result = compare(observed, synthetic, config_path=no_transitions_config)
    domains = _domains(result)
    assert "transitions" not in domains
    assert "timing" in domains  # unaffected


def test_compare_config_path_with_disable(observed, synthetic, no_transitions_config):
    result = compare(
        observed,
        synthetic,
        config_path=no_transitions_config,
        disable=["jobs.feasibility.home_based"],
    )
    assert "transitions" not in _domains(result)
    features = result.features.combined.distances.index.get_level_values("feature")
    assert not any(f.startswith("not home based") for f in features)


def test_compare_grid_and_many_forward_config_path(
    observed, synthetic, no_transitions_config
):
    for fn in (compare_grid, compare_many):
        result = fn(
            {"t": observed}, {"m": synthetic}, config_path=no_transitions_config
        )
        assert "transitions" not in _domains(result)


def test_evaluator_jobs_conflicts_with_config_path_or_disable(observed):
    jobs = get_jobs()
    with pytest.raises(ValueError, match="cannot be combined"):
        Evaluator(observed, jobs=jobs, disable=["jobs.creativity.novelty"])
    with pytest.raises(ValueError, match="cannot be combined"):
        Evaluator(observed, jobs=jobs, config_path="config.toml")
