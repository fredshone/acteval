from acteval._jobs import JobSpec, get_jobs


def test_get_jobs_returns_flat_job_spec_list():
    jobs, run_creativity, run_structural = get_jobs()
    assert isinstance(jobs, list)
    assert all(isinstance(j, JobSpec) for j in jobs)
    assert isinstance(run_creativity, bool)
    assert isinstance(run_structural, bool)


def test_all_jobs_have_required_fields():
    jobs, _, _ = get_jobs()
    for spec in jobs:
        assert spec.domain in {"participations", "transitions", "timing"}
        assert isinstance(spec.name, str) and spec.name
        assert callable(spec.feature_fn)
        assert callable(spec.size_fn)
        assert callable(spec.describe_fn)
        assert callable(spec.distance_fn)
        assert spec.missing_distance in {None, 1.0}


def test_timing_jobs_have_missing_distance():
    jobs, _, _ = get_jobs()
    for spec in jobs:
        if spec.domain == "timing":
            assert spec.missing_distance == 1.0
        else:
            assert spec.missing_distance is None


def test_job_names_are_unique_within_domain():
    jobs, _, _ = get_jobs()
    seen = set()
    for spec in jobs:
        key = (spec.domain, spec.name)
        assert key not in seen, f"Duplicate job key: {key}"
        seen.add(key)
