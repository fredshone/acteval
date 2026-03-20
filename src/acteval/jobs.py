import sys
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from acteval.density.features import participation, times, transitions
from acteval.distance import emd
from acteval.ops import average, average2d, feature_weight, time_average, time_average2d
from acteval.structural.features import structural

_DEFAULT_CONFIG = Path(__file__).parent / "config.toml"


def load_config(path=None) -> dict:
    if path is None:
        path = _DEFAULT_CONFIG
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_density_jobs(cfg: dict) -> list[tuple]:
    """Returns active jobs for participations, transitions, and timing domains.

    Each job is a 4-tuple of ``(feature, size, description_job, distance_job)``
    where *feature* is a 3-tuple ``(name, fn, per_pid_fn | None)``.
    The per-pid function, when present, returns a ``PidFeatures`` instance.
    """
    ngram_min_count = cfg.get("ngrams", {}).get("min_count", 3)
    participations_cfg = cfg.get("jobs", {}).get("participations", {})
    transitions_cfg = cfg.get("jobs", {}).get("transitions", {})
    timing_cfg = cfg.get("jobs", {}).get("timing", {})

    participation_rate_jobs = []
    if participations_cfg.get("lengths", True):
        participation_rate_jobs.append(
            (
                (
                    "lengths",
                    structural.sequence_lengths,
                    structural.sequence_lengths_per_pid,
                ),
                (feature_weight),
                ("length.", average),
                ("EMD", emd),
            )
        )
    if participations_cfg.get("rates", True):
        participation_rate_jobs.append(
            (
                (
                    "participation rate",
                    participation.participation_rates_by_act,
                    participation.participation_rates_by_act_per_pid,
                ),
                (feature_weight),
                ("av. rate", average),
                ("EMD", emd),
            )
        )
    if participations_cfg.get("pair_rates", True):
        participation_rate_jobs.append(
            (
                (
                    "pair participation rate",
                    participation.joint_participation_rate,
                    participation.joint_participation_rate_per_pid,
                ),
                (feature_weight),
                ("av rate.", average),
                ("EMD", emd),
            )
        )

    transition_jobs = []
    if transitions_cfg.get("2-gram", True):
        transition_jobs.append(
            (
                (
                    "2-gram",
                    partial(transitions.transitions_by_act, min_count=ngram_min_count),
                    partial(
                        transitions.transitions_by_act_per_pid,
                        min_count=ngram_min_count,
                    ),
                ),
                (feature_weight),
                ("av. rate", average),
                ("EMD", emd),
            )
        )
    if transitions_cfg.get("3-gram", True):
        transition_jobs.append(
            (
                (
                    "3-gram",
                    partial(
                        transitions.transition_3s_by_act, min_count=ngram_min_count
                    ),
                    partial(
                        transitions.transition_3s_by_act_per_pid,
                        min_count=ngram_min_count,
                    ),
                ),
                (feature_weight),
                ("av. rate", average),
                ("EMD", emd),
            )
        )
    if transitions_cfg.get("4-gram", True):
        transition_jobs.append(
            (
                (
                    "4-gram",
                    partial(
                        transitions.transition_4s_by_act, min_count=ngram_min_count
                    ),
                    partial(
                        transitions.transition_4s_by_act_per_pid,
                        min_count=ngram_min_count,
                    ),
                ),
                (feature_weight),
                ("av. rate", average),
                ("EMD", emd),
            )
        )

    time_jobs = []
    if timing_cfg.get("start_times", True):
        time_jobs.append(
            (
                (
                    "start times",
                    times.start_times_by_act_plan_enum,
                    times.start_times_by_act_plan_enum_per_pid,
                ),
                (feature_weight),
                ("average", time_average),
                ("EMD", emd),
            )
        )
    if timing_cfg.get("durations", True):
        time_jobs.append(
            (
                (
                    "durations",
                    times.durations_by_act_plan_enum,
                    times.durations_by_act_plan_enum_per_pid,
                ),
                (feature_weight),
                ("average", time_average),
                ("EMD", emd),
            )
        )
    if timing_cfg.get("start_durations", True):
        time_jobs.append(
            (
                (
                    "start-durations",
                    times.start_and_duration_by_act_bins,
                    times.start_and_duration_by_act_bins_per_pid,
                ),
                (feature_weight),
                ("average", time_average2d),
                ("EMD", emd),
            )
        )
    if timing_cfg.get("joint_durations", True):
        time_jobs.append(
            (
                (
                    "joint-durations",
                    times.joint_durations_by_act_bins,
                    times.joint_durations_by_act_bins_per_pid,
                ),
                (feature_weight),
                ("average", time_average2d),
                ("EMD", emd),
            )
        )

    return [
        ("participations", participation_rate_jobs),
        ("transitions", transition_jobs),
        ("timing", time_jobs),
    ]


def build_creativity_jobs(cfg: dict) -> bool:
    """Returns whether to run creativity evaluation."""
    return cfg.get("jobs", {}).get("creativity", {}).get("enabled", True)


def build_structural_jobs(cfg: dict) -> bool:
    """Returns whether to run structural/feasibility evaluation."""
    return cfg.get("jobs", {}).get("structural", {}).get("enabled", True)


def get_jobs(config_path=None):
    """Load config and return active job specification.

    Returns:
        tuple of (density_domain_jobs, run_creativity, run_structural)
        where density_domain_jobs is [(domain_name, [jobs])].
    """
    cfg = load_config(config_path)
    return (
        build_density_jobs(cfg),
        build_creativity_jobs(cfg),
        build_structural_jobs(cfg),
    )
