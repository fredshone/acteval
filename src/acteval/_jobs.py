import sys
from copy import deepcopy
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


@dataclass(frozen=True)
class CreativityConfig:
    diversity: bool = True  # enables diversity (desc) + homogeneity (dist)
    novelty: bool = True  # enables novelty (desc) + conservatism (dist)

    @property
    def enabled(self) -> bool:
        return self.diversity or self.novelty


@dataclass(frozen=True)
class StructuralConfig:
    home_based: bool = True
    home_based_novel: bool = False
    consecutive: bool = True
    consecutive_novel: bool = False

    @property
    def enabled(self) -> bool:
        return (
            self.home_based
            or self.home_based_novel
            or self.consecutive
            or self.consecutive_novel
        )

    @property
    def needs_novel_pids(self) -> bool:
        return self.home_based_novel or self.consecutive_novel


@dataclass(frozen=True)
class EvalConfig:
    density: list[JobSpec]
    creativity: CreativityConfig
    structural: StructuralConfig


def load_config(path=None) -> dict:
    if path is None:
        path = _DEFAULT_CONFIG
    with open(path, "rb") as f:
        return tomllib.load(f)


def _dotted_keys(cfg: dict, prefix: str = "") -> list[str]:
    """Flatten a nested config dict into a sorted list of dotted leaf paths."""
    keys = []
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(_dotted_keys(value, full_key))
        else:
            keys.append(full_key)
    return sorted(keys)


def apply_overrides(cfg: dict, disable: list[str]) -> dict:
    """Return a copy of *cfg* with each dotted ``section.key`` path in *disable* set to False.

    Args:
        cfg: Config dict as returned by ``load_config``.
        disable: Dotted paths matching ``config.toml`` keys, e.g.
            ``"jobs.creativity.novelty"`` or ``"jobs.transitions.4-gram"``.

    Returns:
        A new config dict with the requested keys switched off; *cfg* is left untouched.

    Raises:
        ValueError: If a dotted path does not match any known config key.
    """
    cfg = deepcopy(cfg)
    for path in disable:
        parts = path.split(".")
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ValueError(
                    f"unknown disable key '{path}'; valid keys are: {_dotted_keys(cfg)}"
                )
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(
                f"unknown disable key '{path}'; valid keys are: {_dotted_keys(cfg)}"
            )
        node[leaf] = False
    return cfg


def list_disable_keys(config_path=None) -> list[str]:
    """List every dotted ``jobs.section.key`` path accepted by ``disable=[...]``.

    Discoverability companion to ``apply_overrides``/``compare(disable=...)``
    so valid keys don't require reading ``config.toml`` or triggering the
    ``ValueError`` from an unknown key.

    Args:
        config_path: Optional path to a custom config.toml; defaults to the
            built-in config.

    Returns:
        Sorted list of dotted paths, e.g. ``["jobs.creativity.diversity", ...]``.
    """
    cfg = load_config(config_path)
    return _dotted_keys(cfg.get("jobs", {}), prefix="jobs")


def build_density_jobs(cfg: dict) -> list[JobSpec]:
    """Returns active jobs for participations, transitions, and timing domains.

    Returns a flat list of ``JobSpec`` instances. Timing jobs carry
    ``missing_distance=1.0``; participation and transition jobs use ``None``.
    """
    n = cfg.get("ngrams", {}).get("min_count", 3)
    nt = cfg.get("ngrams", {}).get("min_count_trigger", 10)
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
            partial(ngrams, n=2, min_count=n, min_count_trigger=nt),
            "av. rate",
            average,
            None,
        ),
        (
            t,
            "3-gram",
            "transitions",
            "3-gram",
            partial(ngrams, n=3, min_count=n, min_count_trigger=nt),
            "av. rate",
            average,
            None,
        ),
        (
            t,
            "4-gram",
            "transitions",
            "4-gram",
            partial(ngrams, n=4, min_count=n, min_count_trigger=nt),
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


def build_creativity_config(cfg: dict) -> CreativityConfig:
    c = cfg.get("jobs", {}).get("creativity", {})
    return CreativityConfig(
        diversity=c.get("diversity", True),
        novelty=c.get("novelty", True),
    )


def build_structural_config(cfg: dict) -> StructuralConfig:
    s = cfg.get("jobs", {}).get("feasibility", {})
    return StructuralConfig(
        home_based=s.get("home_based", True),
        home_based_novel=s.get("home_based_novel", False),
        consecutive=s.get("consecutive", True),
        consecutive_novel=s.get("consecutive_novel", False),
    )


def get_jobs(config_path=None, disable: list[str] | None = None) -> EvalConfig:
    """Load config and return active job specification.

    Args:
        config_path: Optional path to a custom ``config.toml``; defaults to the
            packaged config.
        disable: Optional dotted ``section.key`` paths to switch off on top of
            whatever ``config_path`` loads, e.g. ``["jobs.creativity.novelty"]``.
    """
    cfg = load_config(config_path)
    if disable:
        cfg = apply_overrides(cfg, disable)
    return EvalConfig(
        density=build_density_jobs(cfg),
        creativity=build_creativity_config(cfg),
        structural=build_structural_config(cfg),
    )
