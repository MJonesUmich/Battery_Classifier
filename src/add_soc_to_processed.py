"""Add SOC column to processed battery datasets.

This script walks through every CSV file under ``assets/processed`` and
calculates a state-of-charge (SOC) profile for each cycle. The augmented files
are written to ``assets/interim`` while preserving the original directory
structure.

The SOC calculation primarily relies on coulomb counting. For each cycle the
script integrates the current over time, normalises the accumulated charge, and
scales the result to 0-100 %. If coulomb counting is not possible (e.g. due to
missing data) the script falls back to the provided ``normalized_time`` column.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

CANONICAL_REQUIRED_COLUMNS = {"elapsed_time_s", "current_a"}
CANONICAL_OPTIONAL_COLUMNS = {"normalized_time", "cycle_index"}
SOC_COLUMN_NAME = "soc"


def _fallback_soc(group: pd.DataFrame) -> np.ndarray:
    """Generate a fallback SOC profile based on normalised time."""

    n_rows = len(group)
    if n_rows == 0:
        return np.asarray([], dtype=float)

    if "normalized_time" in group.columns:
        normalized = pd.to_numeric(group["normalized_time"], errors="coerce").to_numpy(
            dtype=float
        )
        if np.isnan(normalized).any():
            normalized = np.linspace(0.0, 1.0, n_rows)
    else:
        normalized = np.linspace(0.0, 1.0, n_rows)

    mean_current = pd.to_numeric(group["current_a"], errors="coerce").mean()
    if np.isnan(mean_current):
        mean_current = 0.0

    if mean_current < 0.0:
        fallback = 1.0 - normalized
    else:
        fallback = normalized

    return np.clip(fallback, 0.0, 1.0)


def _apply_fallback(group: pd.DataFrame) -> pd.DataFrame:
    result = group.copy()
    result[SOC_COLUMN_NAME] = _fallback_soc(result) * 100.0
    return result


def _compute_soc_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """Compute SOC for a single cycle (group)."""

    result = group.copy()

    elapsed = pd.to_numeric(result["elapsed_time_s"], errors="coerce").to_numpy(
        dtype=float
    )
    current = pd.to_numeric(result["current_a"], errors="coerce").to_numpy(
        dtype=float
    )

    if np.isnan(elapsed).any() or np.isnan(current).any():
        return _apply_fallback(result)

    # Ensure elapsed time starts at zero and is non-decreasing.
    elapsed = elapsed - elapsed[0]
    elapsed = np.maximum.accumulate(elapsed)
    delta_t = np.diff(elapsed, prepend=0.0)
    delta_t = np.clip(delta_t, 0.0, None)

    coulomb = np.cumsum(current * delta_t / 3600.0)
    coulomb = coulomb - np.nanmin(coulomb)
    span = np.nanmax(coulomb)

    if not np.isfinite(span) or span <= 0.0:
        return _apply_fallback(result)

    soc_fraction = np.clip(coulomb / span, 0.0, 1.0)
    result[SOC_COLUMN_NAME] = soc_fraction * 100.0
    return result


def add_soc_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a SOC column appended."""

    missing = [col for col in CANONICAL_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Data frame is missing required columns: {missing}")

    working_df = df.copy()

    if SOC_COLUMN_NAME in working_df.columns:
        working_df = working_df.drop(columns=[SOC_COLUMN_NAME])

    if "cycle_index" in working_df.columns:
        parts = []
        for _, group in working_df.groupby("cycle_index", sort=False):
            parts.append(_compute_soc_for_group(group))
        result = pd.concat(parts, ignore_index=True) if parts else working_df.copy()
    else:
        result = _compute_soc_for_group(working_df)

    return result


def build_column_lookup(columns: pd.Index) -> Dict[str, str]:
    """Create a case-insensitive mapping from canonical to actual column names."""

    lookup: Dict[str, str] = {}
    for original in columns:
        lower = original.lower()
        if lower not in lookup:
            lookup[lower] = original
    return lookup


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a dataframe, add SOC, and restore original column names/order."""

    original_columns = [col for col in df.columns if col.lower() != SOC_COLUMN_NAME]
    df_no_soc = df[original_columns].copy()

    lookup = build_column_lookup(df_no_soc.columns)

    missing = [col for col in CANONICAL_REQUIRED_COLUMNS if col not in lookup]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename_to_canonical: Dict[str, str] = {}
    for canonical in CANONICAL_REQUIRED_COLUMNS.union(CANONICAL_OPTIONAL_COLUMNS):
        original_name = lookup.get(canonical)
        if original_name and original_name != canonical:
            rename_to_canonical[original_name] = canonical

    df_canonical = df_no_soc.rename(columns=rename_to_canonical)
    df_augmented = add_soc_column(df_canonical)

    rename_back = {v: k for k, v in rename_to_canonical.items()}
    df_augmented = df_augmented.rename(columns=rename_back)

    ordered_columns = original_columns + [SOC_COLUMN_NAME]
    return df_augmented[ordered_columns]


def iter_csv_files(processed_root: Path) -> list[Path]:
    return sorted(processed_root.rglob("*.csv"))


def process_file_task(task: Tuple[Path, Path, Path]) -> Tuple[str, bool, str]:
    csv_path, processed_root, output_root = task
    relative_path = csv_path.relative_to(processed_root)
    relative_str = str(relative_path)
    output_path = output_root / relative_path

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pylint: disable=broad-except
        return relative_str, False, f"Failed to read file: {exc}"

    try:
        df_with_soc = process_dataframe(df)
    except ValueError as exc:
        return relative_str, False, str(exc)
    except Exception as exc:  # pylint: disable=broad-except
        return relative_str, False, f"Failed to process file: {exc}"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_soc.to_csv(output_path, index=False)
    except Exception as exc:  # pylint: disable=broad-except
        return relative_str, False, f"Failed to write output: {exc}"

    return relative_str, True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("assets/processed"),
        help="Directory containing processed CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/interim"),
        help="Destination directory for SOC-augmented CSV files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads/processes to use (default: 1).",
    )
    parser.add_argument(
        "--use-processes",
        action="store_true",
        help="Use process-based parallelism instead of threads.",
    )
    args = parser.parse_args()

    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()

    if not processed_root.exists():
        print(f"Processed root does not exist: {processed_root}", file=sys.stderr)
        sys.exit(1)

    csv_paths = iter_csv_files(processed_root)
    if not csv_paths:
        print(f"No CSV files found under {processed_root}")
        return

    if args.workers < 1:
        print("Workers must be at least 1.", file=sys.stderr)
        sys.exit(1)

    processed_count = 0
    skipped_count = 0

    tasks = [(path, processed_root, output_root) for path in csv_paths]

    if args.workers == 1:
        for task in tasks:
            relative, success, message = process_file_task(task)
            if success:
                processed_count += 1
                print(f"[OK] {relative}")
            else:
                skipped_count += 1
                print(f"[SKIP] {relative}: {message}", file=sys.stderr)
    else:
        executor_cls = ProcessPoolExecutor if args.use_processes else ThreadPoolExecutor
        with executor_cls(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(process_file_task, task): task for task in tasks
            }
            for future in as_completed(future_to_task):
                csv_path, _, _ = future_to_task[future]
                default_relative = str(csv_path.relative_to(processed_root))
                try:
                    relative, success, message = future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    skipped_count += 1
                    print(
                        f"[SKIP] {default_relative}: worker raised exception: {exc}",
                        file=sys.stderr,
                    )
                    continue

                if success:
                    processed_count += 1
                    print(f"[OK] {relative}")
                else:
                    skipped_count += 1
                    print(f"[SKIP] {relative}: {message}", file=sys.stderr)

    print(
        f"\nCompleted. Processed {processed_count} file(s), skipped {skipped_count} file(s)."
    )


if __name__ == "__main__":
    main()

