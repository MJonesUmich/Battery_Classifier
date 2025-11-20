"""Convenience launcher for running every parser in sequence with one command."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ParserSpec:
    name: str
    relative_path: Path
    default_args: Sequence[str] = ()

    @property
    def script_path(self) -> Path:
        return REPO_ROOT / self.relative_path


PARSERS: List[ParserSpec] = [
    ParserSpec("dataset1_nca_parser", Path("src/parser/dataset1_nca_parser.py")),
    ParserSpec("cs_cell_parser", Path("src/parser/cs_cell_parser.py")),
    ParserSpec("inr_cell_parser", Path("src/parser/inr_cell_parser.py")),
    ParserSpec("isu_parser", Path("src/parser/isu_parser.py")),
    ParserSpec("mit_parser", Path("src/parser/mit_parser.py")),
    ParserSpec("oxford_cell_parser", Path("src/parser/oxford_cell_parser.py")),
    ParserSpec("pl_cell_parser", Path("src/parser/pl_cell_parser.py")),
    ParserSpec("stanford_cell_parser", Path("src/parser/stanford_cell_parser.py")),
    ParserSpec("tu_finland_cell_parser", Path("src/parser/TU_Finland_cell_parser.py")),
]


def run_parser(spec: ParserSpec) -> int:
    cmd = [sys.executable, str(spec.script_path), *spec.default_args]
    print(f"\n=== Running {spec.name} ===")
    print(" ".join(cmd))
    start = time.perf_counter()
    completed = subprocess.run(cmd, check=False)
    duration = time.perf_counter() - start
    print(f"--- {spec.name} finished in {duration:.2f}s with code {completed.returncode} ---")
    return completed.returncode


def main() -> int:
    successes: List[str] = []
    failures: List[str] = []
    skipped: List[str] = []

    for spec in PARSERS:
        if not spec.script_path.exists():
            skipped.append(spec.name)
            print(f"Skipping {spec.name}: missing script at {spec.script_path}")
            continue

        code = run_parser(spec)
        if code == 0:
            successes.append(spec.name)
        else:
            failures.append(spec.name)

    print("\n=== Summary ===")
    print(f"Successes ({len(successes)}): {', '.join(successes) or 'none'}")
    print(f"Failures ({len(failures)}): {', '.join(failures) or 'none'}")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

