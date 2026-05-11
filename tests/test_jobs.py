from acteval._jobs import CreativityConfig, EvalConfig, JobSpec, StructuralConfig, get_jobs


def test_get_jobs_returns_eval_config():
    config = get_jobs()
    assert isinstance(config, EvalConfig)
    assert isinstance(config.density, list)
    assert all(isinstance(j, JobSpec) for j in config.density)
    assert isinstance(config.creativity, CreativityConfig)
    assert isinstance(config.structural, StructuralConfig)


def test_all_jobs_have_required_fields():
    for spec in get_jobs().density:
        assert spec.domain in {"participations", "transitions", "timing"}
        assert isinstance(spec.name, str) and spec.name
        assert callable(spec.feature_fn)
        assert callable(spec.size_fn)
        assert callable(spec.describe_fn)
        assert callable(spec.distance_fn)
        assert spec.missing_distance in {None, 1.0}


def test_timing_jobs_have_missing_distance():
    for spec in get_jobs().density:
        if spec.domain == "timing":
            assert spec.missing_distance == 1.0
        else:
            assert spec.missing_distance is None


def test_job_names_are_unique_within_domain():
    seen = set()
    for spec in get_jobs().density:
        key = (spec.domain, spec.name)
        assert key not in seen, f"Duplicate job key: {key}"
        seen.add(key)


def test_creativity_config_defaults():
    config = get_jobs()
    assert config.creativity.diversity is True
    assert config.creativity.novelty is True
    assert config.creativity.enabled is True


def test_structural_config_defaults():
    config = get_jobs()
    assert config.structural.home_based is True
    assert config.structural.consecutive is True
    assert config.structural.home_based_novel is False
    assert config.structural.consecutive_novel is False
    assert config.structural.enabled is True
    assert config.structural.needs_novel_pids is False


def test_creativity_config_enabled_property():
    assert CreativityConfig(diversity=False, novelty=False).enabled is False
    assert CreativityConfig(diversity=True, novelty=False).enabled is True


def test_structural_config_enabled_property():
    assert StructuralConfig(home_based=False, consecutive=False).enabled is False
    assert StructuralConfig(home_based=True, consecutive=False).enabled is True


def test_structural_config_needs_novel_pids():
    assert StructuralConfig(home_based_novel=False, consecutive_novel=False).needs_novel_pids is False
    assert StructuralConfig(home_based_novel=True).needs_novel_pids is True
