import argparse
import sys
from pathlib import Path

import pandas as pd

from acteval._report import print_markdown
from acteval.evaluate import Evaluator

_REQUIRED_SCHEDULE_COLS = {"pid", "act"}
_TIMING_COLS = {"start", "end", "duration"}


def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        try:
            return pd.read_parquet(p)
        except ImportError:
            sys.exit(f"Reading {p} requires pyarrow: pip install pyarrow")
    return pd.read_csv(p)


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acteval",
        description="Compare synthetic activity schedules against observed data.",
    )
    p.add_argument("target", help="Path to observed schedules (CSV or Parquet)")
    p.add_argument(
        "--model",
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        required=True,
        help="Synthetic model to compare (repeatable)",
    )
    p.add_argument(
        "--attrs",
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        help="Per-model attributes file (repeatable; NAME must match a --model NAME)",
    )
    p.add_argument("--target-attrs", metavar="PATH", help="Attributes for the target population")
    p.add_argument(
        "--split-on",
        nargs="+",
        metavar="COL",
        help="Attribute columns to split evaluation on (requires --target-attrs)",
    )
    p.add_argument("--config", metavar="PATH", help="Path to custom config.toml")
    p.add_argument(
        "--level",
        choices=["domains", "groups", "features"],
        default="domains",
        help="Aggregation level to display (default: domains)",
    )
    p.add_argument("--output", metavar="DIR", help="Directory to save CSV results")
    p.add_argument("--verbose", action="store_true", help="Print all aggregation levels")
    return p


def _run(args: argparse.Namespace) -> None:
    # --- load target ---
    target = _load_df(args.target)
    _validate_schedule(target, args.target)
    target = _derive_timing(target)

    # --- load synthetic models ---
    synthetic: dict[str, pd.DataFrame] = {}
    for name, path in args.model:
        if name in synthetic:
            sys.exit(f"Duplicate --model name '{name}'")
        df = _load_df(path)
        _validate_schedule(df, path)
        synthetic[name] = _derive_timing(df)

    # --- load per-model attributes ---
    attributes: dict[str, pd.DataFrame] | None = None
    if args.attrs:
        attributes = {}
        for name, path in args.attrs:
            if name not in synthetic:
                sys.exit(f"--attrs NAME '{name}' does not match any --model NAME")
            if name in attributes:
                sys.exit(f"Duplicate --attrs name '{name}'")
            df = _load_df(path)
            if args.split_on:
                _validate_attrs(df, path, args.split_on)
            else:
                _validate_attrs(df, path)
            attributes[name] = df

    # --- validate split-on consistency ---
    if bool(args.split_on) != bool(args.target_attrs):
        sys.exit("--split-on and --target-attrs must be supplied together")

    if args.split_on:
        missing_attrs = [n for n in synthetic if n not in (attributes or {})]
        if missing_attrs:
            sys.exit(
                f"--split-on requires --attrs for every model; missing: {missing_attrs}"
            )

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
    result = evaluator.compare(synthetic, attributes=attributes, verbose=args.verbose)

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
