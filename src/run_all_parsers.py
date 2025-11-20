import importlib
import time
from typing import List, Tuple


PARSER_MODULES: List[Tuple[str, str]] = [
	("CS Cell Parser", "parser.cs_cell_parser"),
	("INR Cell Parser", "parser.inr_cell_parser"),
	("ISU Parser", "parser.isu_parser"),
	("MIT Parser", "parser.mit_parser"),
	("Oxford Cell Parser", "parser.oxford_cell_parser"),
	("PL Cell Parser", "parser.pl_cell_parser"),
	("Stanford Cell Parser", "parser.stanford_cell_parser"),
	("TU Finland Cell Parser", "parser.TU_Finland_cell_parser"),
	("NCA Dataset Parser", "parser.nca_parser"),
]


def run_all() -> None:
	total = len(PARSER_MODULES)
	success = 0
	failures: List[Tuple[str, str]] = []

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


