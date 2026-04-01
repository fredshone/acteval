import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from acteval._aggregation import average, average2d, feature_weight, time_average
from acteval.distance import emd
from acteval.features import participation, structural, times
from acteval.features.transitions import full_sequences, ngrams

_DEFAULT_CONFIG = Path(__file__).parent / "config.toml"


@dataclass(frozen=True)
class JobSpec:
    domain: str
    name: str
    feature_fn: Callable
    size_fn: Callable
    description_name: str
    describe_fn: Callable
    distance_name: str
    distance_fn: Callable
    missing_distance: float | None


def load_config(path=None) -> dict:
    if path is None:
        path = _DEFAULT_CONFIG
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_density_jobs(cfg: dict) -> list[JobSpec]:
    """Returns active jobs for participations, transitions, and timing domains.

    Returns a flat list of ``JobSpec`` instances. Timing jobs carry
    ``missing_distance=1.0``; participation and transition jobs use ``None``.
    """
    n = cfg.get("ngrams", {}).get("min_count", 3)
    p = cfg.get("jobs", {}).get("participations", {})
    t = cfg.get("jobs", {}).get("transitions", {})
    ti = cfg.get("jobs", {}).get("timing", {})
    s = cfg.get("jobs", {}).get("sequences", {})

    # (cfg_section, cfg_key, domain, name, feature_fn, description_name, describe_fn, missing_distance)
    specs = [
        (
            p,
            "lengths",
            "participations",
            "lengths",
            participation.sequence_lengths,
            "length.",
            average,
            None,
        ),
        (
            p,
            "rates",
            "participations",
            "participation rate",
            participation.participation_rates_by_act,
            "av. rate",
            average,
            None,
        ),
        (
            p,
            "pair_rates",
            "participations",
            "pair participation rate",
            participation.joint_participation_rate,
            "av rate.",
            average,
            None,
        ),
        (
            p,
            "seq_rates",
            "participations",
            "seq participation rate",
            participation.participation_rates_by_seq_act,
            "av. rate",
            average,
            None,
        ),
        (
            p,
            "enum_rates",
            "participations",
            "enum participation rate",
            participation.participation_rates_by_act_enum,
            "av. rate",
            average,
            None,
        ),
        (
            t,
            "2-gram",
            "transitions",
            "2-gram",
            partial(ngrams, n=2, min_count=n),
            "av. rate",
            average,
            None,
        ),
        (
            t,
            "3-gram",
            "transitions",
            "3-gram",
            partial(ngrams, n=3, min_count=n),
            "av. rate",
            average,
            None,
        ),
        (
            t,
            "4-gram",
            "transitions",
            "4-gram",
            partial(ngrams, n=4, min_count=n),
            "av. rate",
            average,
            None,
        ),
        (
            ti,
            "start_times",
            "timing",
            "start times",
            times.start_times_by_act_plan_enum,
            "average",
            time_average,
            1.0,
        ),
        (
            ti,
            "durations",
            "timing",
            "durations",
            times.durations_by_act_plan_enum,
            "average",
            time_average,
            1.0,
        ),
        (
            ti,
            "start_durations",
            "timing",
            "start-durations",
            times.start_and_duration_by_act_bins,
            "average",
            average2d,
            1.0,
        ),
        (
            ti,
            "joint_durations",
            "timing",
            "joint-durations",
            times.joint_durations_by_act_bins,
            "average",
            average2d,
            1.0,
        ),
        (
            ti,
            "start_times_by_act",
            "timing",
            "start times by act",
            times.start_times_by_act,
            "average",
            time_average,
            1.0,
        ),
        (
            ti,
            "end_times_by_act",
            "timing",
            "end times by act",
            times.end_times_by_act,
            "average",
            time_average,
            1.0,
        ),
        (
            ti,
            "durations_by_act",
            "timing",
            "durations by act",
            times.durations_by_act,
            "average",
            time_average,
            1.0,
        ),
        (
            ti,
            "time_consistency",
            "timing",
            "time consistency",
            structural.time_consistency,
            "average",
            average,
            None,
        ),
        (
            s,
            "full_sequences",
            "sequences",
            "full sequences",
            full_sequences,
            "av. rate",
            average,
            None,
        ),
    ]

    return [
        JobSpec(
            domain=domain,
            name=name,
            feature_fn=feature_fn,
            size_fn=feature_weight,
            description_name=description_name,
            describe_fn=describe_fn,
            distance_name="EMD",
            distance_fn=emd,
            missing_distance=missing_distance,
        )
        for cfg_section, cfg_key, domain, name, feature_fn, description_name, describe_fn, missing_distance in specs
        if cfg_section.get(cfg_key, True)
    ]


def build_creativity_jobs(cfg: dict) -> bool:
    """Returns whether to run creativity evaluation."""
    return cfg.get("jobs", {}).get("creativity", {}).get("enabled", True)


def build_structural_jobs(cfg: dict) -> bool:
    """Returns whether to run structural/feasibility evaluation."""
    return cfg.get("jobs", {}).get("structural", {}).get("enabled", True)


def get_jobs(config_path=None) -> tuple[list[JobSpec], bool, bool]:
    """Load config and return active job specification.

    Returns:
        tuple of (density_jobs, run_creativity, run_structural)
        where density_jobs is a flat list of JobSpec.
    """
    cfg = load_config(config_path)
    return (
        build_density_jobs(cfg),
        build_creativity_jobs(cfg),
        build_structural_jobs(cfg),
    )
