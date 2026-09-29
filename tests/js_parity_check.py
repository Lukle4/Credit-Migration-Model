"""
Runs only the Google Sheets engine checks of the stress suite: the generated
sheets/Code.gs is executed under Node and compared with the Python engine
(the same checks are part of the full suite in stress_suite.py and inside the
notebook's stress-test cell -- this is just the quick way to run them alone).

Needs `node` on PATH; without it the checks report SKIPPED (not tested).
Run with:  python tests/js_parity_check.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stress_suite import ALL_CHECKS, run_all
from run_stress import load_src_namespace

SHEETS_CHECKS = [c for c in ALL_CHECKS if c.__name__.startswith("check_sheets_engine")
                 or c.__name__ == "check_apps_script_generator_rejects_bad_weights"]


def main():
    passed = run_all(load_src_namespace(), "Google Sheets engine (Code.gs under Node) vs Python", SHEETS_CHECKS)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
