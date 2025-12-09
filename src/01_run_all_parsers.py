import importlib
import time
from pathlib import Path
from typing import List, Tuple

RAW_EXPECTED_SUBDIRS = [
	"CS2",
	"CX2",
	"Dataset_1_NCA_battery",
	"INR",
	"ISU",
	"MIT",
	"Oxford",
	"PL",
	"Stanford",
	"TU_Finland",
]

PARSER_MODULES: List[Tuple[str, str]] = [
	("CS Cell Parser", "parser.cs_cell_parser"),
	("INR Cell Parser", "parser.inr_cell_parser"),
	("ISU Parser", "parser.isu_parser"),
	("MIT Parser", "parser.mit_parser"),
	("Oxford Cell Parser", "parser.oxford_cell_parser"),
	("PL Cell Parser", "parser.pl_cell_parser"),
	("Stanford Cell Parser", "parser.stanford_cell_parser"),
	("TU Finland Cell Parser", "parser.TU_Finland_cell_parser"),
	("NCA Dataset Parser", "parser.tongji_parser"),
]


def _ensure_raw_assets_present() -> None:
	"""
	Parsers require the extracted raw data bundle described in README step 2.
	This guard avoids running when raw_20251207.zip has not been downloaded
	and extracted into assets/raw/.
	"""
	root = Path(__file__).resolve().parent.parent
	raw_dir = root / "assets" / "raw"
	if not raw_dir.is_dir():
		raise FileNotFoundError(
			"Raw data missing. Download raw_20251207.zip from the README link https://drive.google.com/file/d/1sHScf_HNTzuAurPBTFqm3j2pkNYALomt/view?usp=sharing"
			"and extract to assets/raw/ before running parsers."
		)

	if not any((raw_dir / name).exists() for name in RAW_EXPECTED_SUBDIRS):
		raise FileNotFoundError(
			"Raw data appears incomplete. Extract raw_20251207.zip https://drive.google.com/file/d/1sHScf_HNTzuAurPBTFqm3j2pkNYALomt/view?usp=sharing into "
			"assets/raw/ (expected folders such as CS2, INR, MIT, Oxford, PL, "
			"Stanford, TU_Finland)."
		)


def run_all() -> None:
	total = len(PARSER_MODULES)
	success = 0
	failures: List[Tuple[str, str]] = []

	_ensure_raw_assets_present()

	print(f"Running {total} parsers sequentially...")
	start_all = time.time()

	for display_name, module_path in PARSER_MODULES:
		print(f"\n=== {display_name} ({module_path}) ===")
		start = time.time()
		try:
			mod = importlib.import_module(module_path)
		except Exception as e:
			msg = f"Import failed: {e}"
			print(msg)
			failures.append((display_name, msg))
			continue

		try:
			# Each parser exposes: main(config: Optional[ProcessingConfig] = None) -> None
			mod.main()  # type: ignore[attr-defined]
			elapsed = time.time() - start
			print(f"Completed {display_name} in {elapsed:.1f}s")
			success += 1
		except Exception as e:
			elapsed = time.time() - start
			msg = f"Execution failed after {elapsed:.1f}s: {e}"
			print(msg)
			failures.append((display_name, msg))

	total_elapsed = time.time() - start_all
	print("\n=== Summary ===")
	print(f"Succeeded: {success}/{total} in {total_elapsed:.1f}s")
	if failures:
		print("Failures:")
		for name, err in failures:
			print(f"- {name}: {err}")


if __name__ == "__main__":
	run_all()


# === Summary ===
# Succeeded: 9/9 in 3229.6s