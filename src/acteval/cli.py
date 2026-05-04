import argparse
import sys
from pathlib import Path

import pandas as pd

from acteval._report import print_markdown
from acteval.evaluate import Evaluator

_REQUIRED_SCHEDULE_COLS = {"pid", "act"}
_TIMING_COLS = {"start", "end", "duration"}
_KNOWN_COLS = {"pid", "act", "start", "end", "duration"}


def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        try:
            return pd.read_parquet(p)
        except ImportError:
            sys.exit(f"Reading {p} requires pyarrow: pip install pyarrow")
    return pd.read_csv(p)


def _read_columns(path: str) -> set[str]:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq

            return set(pq.read_schema(str(p)).names)
        except ImportError:
            sys.exit(f"Reading {p} requires pyarrow: pip install pyarrow")
    return set(pd.read_csv(p, nrows=0).columns)


def _validate_schedule(df: pd.DataFrame, path: str) -> None:
    cols = set(df.columns)
    missing_required = _REQUIRED_SCHEDULE_COLS - cols
    if missing_required:
        sys.exit(f"{path}: missing required columns {sorted(missing_required)}")
    timing_present = _TIMING_COLS & cols
    if len(timing_present) < 2:
        sys.exit(
            f"{path}: at least two of {sorted(_TIMING_COLS)} are required; "
            f"found only {sorted(timing_present)}"
        )


def _derive_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the missing timing column when only two of start/end/duration are present."""
    has_start = "start" in df.columns
    has_end = "end" in df.columns
    has_dur = "duration" in df.columns
    if has_start and has_end and not has_dur:
        df = df.copy()
        df["duration"] = df["end"] - df["start"]
    elif has_start and has_dur and not has_end:
        df = df.copy()
        df["end"] = df["start"] + df["duration"]
    elif has_end and has_dur and not has_start:
        df = df.copy()
        df["start"] = df["end"] - df["duration"]
    return df


def _validate_attrs(df: pd.DataFrame, path: str, split_on: list | None = None) -> None:
    if "pid" not in df.columns:
        sys.exit(f"{path}: attributes file missing required column 'pid'")
    if split_on:
        missing = [c for c in split_on if c not in df.columns]
        if missing:
            sys.exit(f"{path}: missing split-on columns {missing}")


def _classify_file(path: str) -> str:
    """Return 'schedule' or 'attrs'; exit if the file cannot be classified."""
    cols = _read_columns(path)
    if "pid" in cols and "act" in cols:
        return "schedule"
    if "pid" in cols and "act" not in cols:
        extra = cols - _KNOWN_COLS
        if extra:
            return "attrs"
        sys.exit(
            f"{path}: cannot classify — has 'pid' but no 'act' and no attribute "
            f"columns beyond {sorted(_KNOWN_COLS - {'act'})}; found: {sorted(cols)}"
        )
    sys.exit(f"{path}: cannot classify — missing 'pid' column; found: {sorted(cols)}")


def _discover_batch(batch_dir: str) -> list[tuple[str, str, str | None]]:
    """Discover model subdirs within *batch_dir* and classify their files.

    Returns [(model_name, schedule_path, attrs_path | None), ...].
    Exits with an error if any ambiguity is detected or attrs presence is
    inconsistent across models.
    """
    root = Path(batch_dir)
    if not root.is_dir():
        sys.exit(f"--batch: {batch_dir!r} is not a directory")

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subdirs:
        sys.exit(f"--batch: no subdirectories found in {batch_dir!r}")

    _EXTS = {".csv", ".parquet", ".pq"}
    results: list[tuple[str, str, str | None]] = []

    for subdir in subdirs:
        files = sorted(p for p in subdir.iterdir() if p.suffix in _EXTS and p.is_file())
        schedules: list[Path] = []
        attr_files: list[Path] = []
        for f in files:
            kind = _classify_file(str(f))
            if kind == "schedule":
                schedules.append(f)
            else:
                attr_files.append(f)

        if len(schedules) == 0:
            sys.exit(f"--batch: no schedule file found in {str(subdir)!r}")
        if len(schedules) > 1:
            sys.exit(
                f"--batch: multiple schedule files found in {str(subdir)!r}: "
                f"{[str(s) for s in schedules]}"
            )
        if len(attr_files) > 1:
            sys.exit(
                f"--batch: multiple attributes files found in {str(subdir)!r}: "
                f"{[str(a) for a in attr_files]}"
            )

        attrs_path = str(attr_files[0]) if attr_files else None
        results.append((subdir.name, str(schedules[0]), attrs_path))

    has_attrs_flags = [r[2] is not None for r in results]
    if any(has_attrs_flags) and not all(has_attrs_flags):
        missing = [r[0] for r in results if r[2] is None]
        sys.exit(
            f"--batch: inconsistent attributes — some model directories have "
            f"attributes files and some do not; missing for: {missing}"
        )

    print(f"Discovered {len(results)} model(s) from {batch_dir!r}:")
    for name, sched, attrs in results:
        attrs_str = f", attrs: {attrs}" if attrs else ""
        print(f"  {name}: schedule: {sched}{attrs_str}")

    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acteval",
        description="Compare synthetic activity schedules against observed data.",
    )
    p.add_argument("target", help="Path to observed schedules (CSV or Parquet)")
    p.add_argument(
        "target_attrs",
        nargs="?",
        default=None,
        metavar="TARGET_ATTRS",
        help="Optional path to target population attributes (CSV or Parquet)",
    )
    p.add_argument(
        "--model",
        "-m",
        nargs="+",
        metavar="ARG",
        action="append",
        help="Synthetic model: NAME SCHEDULE_PATH [ATTRS_PATH] (repeatable)",
    )
    p.add_argument(
        "--split-on",
        nargs="+",
        metavar="COL",
        help="Attribute columns to split evaluation on (requires TARGET_ATTRS and per-model attrs)",
    )
    p.add_argument("--config", "-c", metavar="PATH", help="Path to custom config.toml")
    p.add_argument(
        "--level",
        "-l",
        choices=["domains", "groups", "features"],
        default="domains",
        help="Aggregation level to display (default: domains)",
    )
    p.add_argument(
        "--output", "-o", metavar="DIR", help="Directory to save CSV results"
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Print all aggregation levels"
    )
    p.add_argument(
        "--batch",
        "-b",
        metavar="DIR",
        help="Directory of model subdirs; schedule and attrs auto-discovered per subdir",
    )
    return p


def _run(args: argparse.Namespace) -> None:
    # --- mutual exclusivity ---
    if args.batch and args.model:
        sys.exit("--batch and --model are mutually exclusive")
    if not args.batch and not args.model:
        sys.exit("one of --model / -m or --batch / -b is required")

    # --- load target ---
    target = _load_df(args.target)
    _validate_schedule(target, args.target)
    target = _derive_timing(target)

    # --- build model specs: (name, schedule_path, attrs_path | None) ---
    model_specs: list[tuple[str, str, str | None]] = []

    if args.model:
        for spec in args.model:
            if len(spec) == 2:
                model_specs.append((spec[0], spec[1], None))
            elif len(spec) == 3:
                model_specs.append((spec[0], spec[1], spec[2]))
            else:
                sys.exit(
                    f"--model/-m takes 2 or 3 arguments: NAME SCHEDULE_PATH [ATTRS_PATH]; "
                    f"got {len(spec)}: {spec}"
                )
    else:
        model_specs = _discover_batch(args.batch)

    # --- load synthetic models ---
    synthetic: dict[str, pd.DataFrame] = {}
    all_attrs_paths: dict[str, str] = {}

    for name, sched_path, inline_attrs_path in model_specs:
        if name in synthetic:
            sys.exit(f"Duplicate model name '{name}'")
        df = _load_df(sched_path)
        _validate_schedule(df, sched_path)
        synthetic[name] = _derive_timing(df)
        if inline_attrs_path is not None:
            all_attrs_paths[name] = inline_attrs_path

    # --- load attributes ---
    attributes: dict[str, pd.DataFrame] | None = None
    if all_attrs_paths:
        attributes = {}
        for name, path in all_attrs_paths.items():
            df = _load_df(path)
            if args.split_on:
                _validate_attrs(df, path, args.split_on)
            else:
                _validate_attrs(df, path)
            attributes[name] = df

    # --- attrs all-or-nothing consistency ---
    has_target_attrs = args.target_attrs is not None
    has_any_model_attrs = bool(all_attrs_paths)

    if has_target_attrs or has_any_model_attrs:
        if not has_target_attrs:
            sys.exit(
                "target attributes must be provided when model attributes are given; "
                "pass TARGET_ATTRS as the second positional argument"
            )
        missing_model_attrs = [n for n in synthetic if n not in all_attrs_paths]
        if missing_model_attrs:
            sys.exit(
                f"attributes must be provided for all models or none; "
                f"missing for: {missing_model_attrs}"
            )

    # --- split-on requires attrs (and vice versa) ---
    if bool(args.split_on) != bool(args.target_attrs):
        sys.exit("--split-on and TARGET_ATTRS must be specified together")

    # --- load target attributes ---
    target_attributes: pd.DataFrame | None = None
    if args.target_attrs:
        target_attributes = _load_df(args.target_attrs)
        _validate_attrs(target_attributes, args.target_attrs, args.split_on)

    # --- evaluate ---
    print(f"acteval — comparing {len(synthetic)} model(s) to {args.target}\n")
    evaluator = Evaluator(
        target,
        target_attributes=target_attributes,
        split_on=args.split_on,
        config_path=args.config,
    )
    # Attrs are only meaningful for splitting; don't pass them when split_on is absent
    attrs_for_evaluator = attributes if args.split_on else None
    result = evaluator.compare(
        synthetic, attributes=attrs_for_evaluator, verbose=args.verbose
    )

    # --- print results ---
    levels_to_print = {"domains"}
    if args.verbose:
        levels_to_print = {"features", "groups", "domains"}
    elif args.level != "domains":
        levels_to_print = {args.level, "domains"}

    if "features" in levels_to_print:
        print("Feature distances:")
        print_markdown(result.features.combined.distances)
        if result.has_splits:
            print("\nFeature distances by attribute:")
            print_markdown(result.features.by_attribute.distances)

    if "groups" in levels_to_print:
        print("\nGroup distances:")
        print_markdown(result.groups.combined.distances)
        if result.has_splits:
            print("\nGroup distances by attribute:")
            print_markdown(result.groups.by_attribute.distances)

    if "domains" in levels_to_print:
        print("\nDomain distances (lower is better):")
        print_markdown(result.domains.combined.distances)
        if result.has_splits:
            print("\nDomain distances by attribute:")
            print_markdown(result.domains.by_attribute.distances)

    ranked = result.rank_models()
    print(f"\nMean distance: {dict(ranked)}")
    print(f"Best model: {result.best_model}")

    # --- save ---
    if args.output:
        result.save(args.output)
        print(f"\nResults saved to {args.output}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _run(args)
