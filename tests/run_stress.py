"""
Runs the stress suite (stress_suite.py) against TWO independent namespaces:

1. The local src/ modules (normal Python imports).
2. The notebook's own code, extracted from credit_migration_model.ipynb and
   executed fresh into an isolated namespace -- exercising the ACTUAL bytes
   that would run in Colab, not just assuming they match src/ because
   that's where they came from. The notebook build process (docstring
   stripping, import removal, cell reordering) is exactly the kind of thing
   that can silently introduce a divergence, so this checks it directly
   rather than trusting it.

Run with:  python tests/run_stress.py
"""

import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))

from stress_suite import run_all


def load_src_namespace() -> dict:
    import config
    import simulate_macro
    import matrix_builder
    import markov_dgp
    import feature_engineering
    import ols_model
    import markov_engine
    import sheets_export
    import excel_export

    return {
        "config": config,
        "simulate_macro_paths": simulate_macro.simulate_macro_paths,
        "build_feature_matrix": matrix_builder.build_feature_matrix,
        "cascade_row": markov_dgp.cascade_row,
        "build_transition_tensor": markov_dgp.build_transition_tensor,
        "simulate_observed_counts": markov_dgp.simulate_observed_counts,
        "_stage_log_odds_and_weight": feature_engineering._stage_log_odds_and_weight,
        "build_all_features": feature_engineering.build_all_features,
        "fit_ols": ols_model.fit_ols,
        "engine_sanity_check": ols_model.engine_sanity_check,
        "validate_wls_reduces_to_ols": ols_model.validate_wls_reduces_to_ols,
        "validate_softmax_and_chain": markov_engine.validate_softmax_and_chain,
        "assess_scenario_range": markov_engine.assess_scenario_range,
        "compute_x_train_bounds": markov_engine.compute_x_train_bounds,
        "predict_scenario": markov_engine.predict_scenario,
        "build_apps_script": sheets_export.build_apps_script,
        "verify_apps_script": sheets_export.verify_apps_script,
        "build_workbook": excel_export.build_workbook,
        "verify_workbook": excel_export.verify_workbook,
        "model_weights_path": os.path.join(PROJECT_ROOT, "model_weights.json"),
        "code_gs_path": os.path.join(PROJECT_ROOT, "sheets", "Code.gs"),
        "excel_path": os.path.join(PROJECT_ROOT, "credit_migration_scenario_calculator.xlsx"),
    }


def load_notebook_namespace() -> dict:
    nb_path = os.path.join(os.path.dirname(__file__), "..", "credit_migration_model.ipynb")
    nb = json.load(open(nb_path, encoding="utf-8"))
    code = ""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            code += "".join(cell["source"]) + "\n\n"

    ns: dict = {}
    exec(compile(code, nb_path, "exec"), ns)  # noqa: S102 -- deliberately executing our own notebook for testing

    return {
        "config": ns["config"],
        "simulate_macro_paths": ns["simulate_macro_paths"],
        "build_feature_matrix": ns["build_feature_matrix"],
        "cascade_row": ns["cascade_row"],
        "build_transition_tensor": ns["build_transition_tensor"],
        "simulate_observed_counts": ns["simulate_observed_counts"],
        "_stage_log_odds_and_weight": ns["_stage_log_odds_and_weight"],
        "build_all_features": ns["build_all_features"],
        "fit_ols": ns["fit_ols"],
        "engine_sanity_check": ns["engine_sanity_check"],
        "validate_wls_reduces_to_ols": ns["validate_wls_reduces_to_ols"],
        "validate_softmax_and_chain": ns["validate_softmax_and_chain"],
        "assess_scenario_range": ns["assess_scenario_range"],
        "compute_x_train_bounds": ns["compute_x_train_bounds"],
        "predict_scenario": ns["predict_scenario"],
        "build_apps_script": ns["build_apps_script"],
        "verify_apps_script": ns["verify_apps_script"],
        "build_workbook": ns["build_workbook"],
        "verify_workbook": ns["verify_workbook"],
        # the notebook writes these relative to the working directory, so check exactly where it just wrote them
        "model_weights_path": os.path.abspath("model_weights.json"),
        "code_gs_path": os.path.abspath(os.path.join("sheets", "Code.gs")),
        "excel_path": os.path.abspath("credit_migration_scenario_calculator.xlsx"),
    }


def main():
    src_passed = run_all(load_src_namespace(), "src/ (local pipeline)")
    notebook_passed = run_all(load_notebook_namespace(), "credit_migration_model.ipynb (extracted + executed fresh)")

    print(f"\n{'=' * 70}\nOVERALL\n{'=' * 70}")
    print(f"src/:     {'PASS' if src_passed else 'FAIL'}")
    print(f"notebook: {'PASS' if notebook_passed else 'FAIL'}")
    if src_passed != notebook_passed:
        print("\n*** DIVERGENCE: src/ and the notebook gave different results on the same checks. ***")
        print("*** The notebook has likely drifted from src/ -- investigate before trusting either. ***")

    sys.exit(0 if (src_passed and notebook_passed) else 1)


if __name__ == "__main__":
    main()
