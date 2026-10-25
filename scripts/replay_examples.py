"""Recompute three saved benchmark examples without network calls."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search_horizon.io import read, rows, write
from scripts.score_value_forecasts import evaluate
from scripts.score_search_policy import evaluate_policy
from scripts.score_collection_forecasts import evaluate as evaluate_collection


def main():
    data = ROOT / "data/forecast_benchmark"
    binary = data / "cost_confirmation"
    draws = rows(binary / "semantic_flash_v1_draws.jsonl")
    predictions = rows(binary / "predictions/source120_probability_affine.jsonl")
    baseline = read(binary / "source120_baseline.json")
    priors = {"reuse": baseline["probability_priors"]["THINK"],
              "new": baseline["probability_priors"]["SEARCH"]}
    value = evaluate(draws, predictions, "deepseek-flash", baseline["constant_effect"],
                     10000, 2026101511, priors, "equal_strata")
    result = {
        "value": {k: v for k, v in value.items() if k != "units"},
        "policy": evaluate_policy(draws, predictions, "deepseek-flash", 7007.49375,
                                  .005, 10000, 2026101815, "equal_strata"),
        "collection": evaluate_collection(rows(data / "collection_value/test_draws.jsonl"),
            rows(data / "collection_value/predictions/query_aware.jsonl"), 10000, 2026102302),
    }
    expected = read(ROOT / "examples/expected_replay.json")

    def compare(actual, reference):
        if isinstance(reference, dict):
            for key in reference:
                compare(actual[key], reference[key])
        elif isinstance(reference, list):
            if not np.allclose(actual, reference, rtol=1e-9, atol=1e-12):
                raise AssertionError("Saved interval differs from replay")
        elif isinstance(reference, (int, float)) and not isinstance(reference, bool):
            if not np.isclose(actual, reference, rtol=1e-9, atol=1e-12):
                raise AssertionError("Saved numerical result differs from replay")
        elif actual != reference:
            raise AssertionError("Saved result field differs from replay")

    compare(result, expected)
    for name, report in result.items():
        write(ROOT / "results" / (name + ".json"), report)
    print("PASS: Test-1632 value, costed policy, and Collection-96 reproduce the saved scores.")
    print("Action-success Brier:",
          (value["metrics"]["Brier_reuse"]["mean"] + value["metrics"]["Brier_new"]["mean"]) / 2)
    print("Corrected gain MSE:", value["metrics"]["corrected_effect_MSE"]["mean"])
    print("Costed policy utility:", result["policy"]["metrics"]["utility"]["mean"])
    print("Collection gain MSE:", result["collection"]["gain_MSE"]["mean"])


if __name__ == "__main__":
    main()
