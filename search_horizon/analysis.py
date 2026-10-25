"""Frozen-label scoring and question-paired SearchHorizon estimates."""

from __future__ import annotations

import csv
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("STOP", "THINK", "SEARCH")
COMPARISONS = (("SEARCH", "THINK"), ("SEARCH", "STOP"), ("THINK", "STOP"))
MISSING_STATUSES = {"infra_error", "budget_exhausted", "checkpoint_pending"}


def normalize(text: str) -> str:
    """Use the source project's official SQuAD answer normalization."""
    text = "".join(c for c in str(text).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(prediction: str, answers: list[str], status: str = "ok", metric: str = "squad") -> dict:
    """Match rt.metrics.score, including aliases and Hotpot categorical answers."""
    if status != "ok" or prediction.strip().upper() == "UNKNOWN":
        return {"em": 0.0, "f1": 0.0}
    if not answers:
        raise ValueError("The frozen answer list must contain at least one answer")

    def one(gold: str) -> tuple[float, float]:
        p, g = normalize(prediction), normalize(gold)
        em = float(p == g)
        if metric == "hotpot" and (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g:
            return em, 0.0
        a, b = p.split(), g.split()
        overlap = sum((Counter(a) & Counter(b)).values())
        if metric == "squad" and (not a or not b):
            return em, float(a == b)
        return em, 2 * overlap / (len(a) + len(b)) if overlap else 0.0

    values = [one(answer) for answer in answers]
    return {"em": max(x[0] for x in values), "f1": max(x[1] for x in values)}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n or p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    terms = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
             + j * math.log(p) + (n - j) * math.log1p(-p) for j in range(k + 1)]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(term - peak) for term in terms))


def _binomial_root(k: int, n: int, target: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(65):
        mid = (low + high) / 2
        if _binomial_cdf(k, n, mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def exact_binomial_interval(successes: int, n: int, tail: float = 0.025) -> list[float | None]:
    """Clopper-Pearson limits with the given probability in each tail."""
    if not n:
        return [None, None]
    lower = 0.0 if successes == 0 else _binomial_root(successes - 1, n, 1 - tail)
    upper = 1.0 if successes == n else _binomial_root(successes, n, tail)
    return [lower, upper]


def rescue_upper(successes: int, n: int) -> float | None:
    """One-sided exact 95% upper limit for a fixed generation repeat."""
    if not n:
        return None
    if successes == 0:
        return 1 - 0.05 ** (1 / n)
    return exact_binomial_interval(successes, n, tail=0.05)[1]


def _task_clusters(nodes: list[dict], labels: dict, field: str = "subquestion_ids") -> dict[str, str]:
    parent = {node["task_id"]: node["task_id"] for node in nodes}

    def find(task: str) -> str:
        while parent[task] != task:
            parent[task] = parent[parent[task]]
            task = parent[task]
        return task

    owners = {}
    for node in nodes:
        task = node["task_id"]
        ids = labels[task].get(field, []) if field == "supporting_page_titles" else node.get(field, labels[task].get(field, []))
        for question in ids:
            question = str(question)
            if question in owners:
                left, right = find(task), find(owners[question])
                parent[max(left, right)] = min(left, right)
            else:
                owners[question] = task
    return {task: find(task) for task in parent}


def aggregate_question_scores(scored: list[dict], jobs: list[dict], nodes: list[dict]) -> list[dict]:
    """Keep all repeats together and retain explicit planned/completed counts."""
    metadata = {node["task_id"]: node for node in nodes}
    expected = defaultdict(lambda: defaultdict(set))
    phases = defaultdict(set)
    for job in jobs:
        expected[(job["model"], job["task_id"])][job["arm"]].add(job["repeat"])
        phases[(job["model"], job["task_id"])].add(job.get("phase", "main"))
    observed = defaultdict(lambda: defaultdict(dict))
    for row in scored:
        observed[(row["model"], row["task_id"])][row["arm"]][row["repeat"]] = row
    records = []
    for (model, task), planned in sorted(expected.items()):
        node = metadata[task]
        arms = {}
        for arm in ARMS:
            by_repeat = observed[(model, task)][arm]
            required = planned[arm]
            available = {repeat: row for repeat, row in by_repeat.items() if row["status"] not in MISSING_STATUSES}
            complete = bool(required) and required <= available.keys()
            finished = bool(required) and required <= by_repeat.keys() and all(by_repeat[r]["status"] != "checkpoint_pending" for r in required)
            arms[arm] = {
                "expected_repeats": sorted(required), "n_finished": len(by_repeat),
                "n_evaluable": len(available), "complete": complete, "all_finished": finished,
                "em": float(np.mean([available[r]["em"] for r in required])) if complete else None,
                "f1": float(np.mean([available[r]["f1"] for r in required])) if complete else None,
                "all_finished_em": float(np.mean([by_repeat[r]["em"] for r in required])) if finished else None,
                "by_repeat": by_repeat,
            }
        records.append({
            "model": model, "task_id": task, "node_id": node["node_id"], "corpus": node["corpus"],
            "dataset": node.get("dataset", "musique"),
            "producer": node.get("provenance", {}).get("kind", "historical"),
            "phase": "/".join(sorted(phases[(model, task)])),
            "group": node["group"], "jaccard": node.get("jaccard"), "novelty": node.get("novelty"),
            "identical_query": node.get("identical_query", node.get("query1") == node.get("query2")),
            "arms": arms,
        })
    return records


def _bootstrap(values: np.ndarray, strata: list[str], seed: int, repetitions: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.zeros(repetitions)
    for stratum in sorted(set(strata)):
        cell = values[np.array([s == stratum for s in strata])]
        draws += rng.choice(cell, size=(repetitions, len(cell)), replace=True).sum(axis=1)
    return draws / len(values)


def _cluster_bootstrap(values: np.ndarray, clusters: list[str], seed: int, repetitions: int) -> np.ndarray:
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(value)
    sums = np.array([sum(grouped[key]) for key in sorted(grouped)])
    sizes = np.array([len(grouped[key]) for key in sorted(grouped)])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(sums), size=(repetitions, len(sums)))
    return sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)


def _paired_exact_interval(records: list[dict], left: str, right: str) -> list[float | None]:
    if not records:
        return [None, None]
    repeats = records[0]["arms"][left]["expected_repeats"]
    if not repeats:
        return [None, None]
    strata = sorted({(row.get("dataset", "musique"), row["corpus"], row["group"]) for row in records})
    tail = 0.05 / (4 * len(repeats) * len(strata))
    intervals = []
    for repeat in repeats:
        interval = np.zeros(2)
        for stratum in strata:
            cell = [row for row in records if (row.get("dataset", "musique"), row["corpus"], row["group"]) == stratum]
            outcomes = [(row["arms"][left]["by_repeat"][repeat]["em"], row["arms"][right]["by_repeat"][repeat]["em"])
                        for row in cell]
            rescue = sum(a == 1 and b == 0 for a, b in outcomes)
            harm = sum(a == 0 and b == 1 for a, b in outcomes)
            r = exact_binomial_interval(rescue, len(cell), tail)
            h = exact_binomial_interval(harm, len(cell), tail)
            interval += len(cell) / len(records) * np.array([r[0] - h[1], r[1] - h[0]])
        intervals.append(interval)
    return [float(np.mean([ci[i] for ci in intervals])) for i in range(2)]


def paired_estimate(records: list[dict], left: str, right: str, clusters: dict[str, str],
                    seed: int = 20261014, repetitions: int = 10000, margin: float = 0.05,
                    support_clusters: dict[str, str] | None = None) -> dict:
    """Estimate a task mean with frozen-stratum sampling and exact safeguards."""
    paired = [row for row in records if row["arms"][left]["complete"] and row["arms"][right]["complete"]]
    if any(row["arms"][left]["expected_repeats"] != row["arms"][right]["expected_repeats"] for row in paired):
        raise ValueError("Paired arms require the same frozen repeat identifiers")
    if not paired:
        return {"n": 0, "mean": None, "ci95": [None, None], "cluster_ci95": [None, None],
                "n_clusters": 0, "bootstrap_degenerate": False, "conservative_paired_ci95": [None, None],
                "support_page_cluster_ci95": [None, None], "n_support_page_clusters": 0,
                "f1_mean": None, "f1_ci95": [None, None], "first_repeat_rescue_upper95": None,
                "equivalence_supported": False, "repeat_outcomes": [], "metric": "em"}
    values = np.array([row["arms"][left]["em"] - row["arms"][right]["em"] for row in paired])
    strata = [row.get("dataset", "musique") + "/" + row["corpus"] + "/" + row["group"] for row in paired]
    cluster_ids = [clusters[row["task_id"]] for row in paired]
    draws = _bootstrap(values, strata, seed, repetitions)
    cluster_draws = _cluster_bootstrap(values, cluster_ids, seed, repetitions)
    exact = _paired_exact_interval(paired, left, right)
    independent = len(set(cluster_ids)) == len(paired)
    support_ids = [support_clusters[row["task_id"]] for row in paired] if support_clusters else []
    support_draws = _cluster_bootstrap(values, support_ids, seed, repetitions) if support_ids else None
    independent = independent and (not support_ids or len(set(support_ids)) == len(paired))
    repeat_outcomes = []
    for repeat in paired[0]["arms"][left]["expected_repeats"]:
        outcomes = [(row["arms"][left]["by_repeat"][repeat]["em"], row["arms"][right]["by_repeat"][repeat]["em"])
                    for row in paired]
        rescue = sum(a == 1 and b == 0 for a, b in outcomes)
        harm = sum(a == 0 and b == 1 for a, b in outcomes)
        if rescue:
            rescue_limit = 0.0
            cells = sorted(set(strata))
            for cell in cells:
                cell_outcomes = [outcome for stratum, outcome in zip(strata, outcomes) if stratum == cell]
                successes = sum(a == 1 and b == 0 for a, b in cell_outcomes)
                limit = exact_binomial_interval(successes, len(cell_outcomes), tail=0.05 / len(cells))[1]
                rescue_limit += len(cell_outcomes) / len(paired) * limit
            rescue_limit_method = "Weighted stratum-specific one-sided binomial limits with Bonferroni coverage."
        else:
            rescue_limit = rescue_upper(0, len(paired))
            rescue_limit_method = "Zero-rescue bound for independent question indicators; identical probabilities are unnecessary."
        repeat_outcomes.append({
            "repeat": repeat, "n": len(paired), "rescue": rescue, "harm": harm,
            "both_correct": sum(a == b == 1 for a, b in outcomes),
            "both_wrong": sum(a == b == 0 for a, b in outcomes),
            "rescue_rate": rescue / len(paired), "harm_rate": harm / len(paired),
            "rescue_upper95": rescue_limit, "rescue_upper_method": rescue_limit_method,
            "prespecified_first_repeat_bound": repeat == paired[0]["arms"][left]["expected_repeats"][0],
            "independent_question_bound": independent,
        })
    f1 = np.array([row["arms"][left]["f1"] - row["arms"][right]["f1"] for row in paired])
    patterns = Counter()
    for row in paired:
        differences = [row["arms"][left]["by_repeat"][repeat]["em"] - row["arms"][right]["by_repeat"][repeat]["em"]
                       for repeat in row["arms"][left]["expected_repeats"]]
        patterns["rescue_in_every_repeat"] += all(value == 1 for value in differences)
        patterns["harm_in_every_repeat"] += all(value == -1 for value in differences)
        patterns["any_rescue"] += 1 in differences
        patterns["any_harm"] += -1 in differences
        patterns["rescue_and_harm_across_repeats"] += 1 in differences and -1 in differences
    return {
        "n": len(paired), "mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(draws, [.025, .975])],
        "metric": "em", "f1_mean": float(f1.mean()),
        "f1_ci95": [float(x) for x in np.quantile(_bootstrap(f1, strata, seed, repetitions), [.025, .975])],
        "cluster_ci95": [float(x) for x in np.quantile(cluster_draws, [.025, .975])] if len(set(cluster_ids)) > 1 else [None, None],
        "n_clusters": len(set(cluster_ids)), "bootstrap_degenerate": bool(np.ptp(draws) == 0),
        "support_page_cluster_ci95": [float(x) for x in np.quantile(support_draws, [.025, .975])] if len(set(support_ids)) > 1 else [None, None],
        "n_support_page_clusters": len(set(support_ids)),
        "conservative_paired_ci95": exact, "exact_interval_assumes_independent_questions": independent,
        "equivalence_margin": margin,
        "equivalence_supported": independent and exact[0] > -margin and exact[1] < margin,
        "repeat_outcomes": repeat_outcomes, "first_repeat_rescue_upper95": repeat_outcomes[0]["rescue_upper95"],
        "question_repeat_patterns": dict(patterns),
        "question_difference_counts": dict(sorted(Counter(str(float(value)) for value in values).items())),
        "conservative_interval_method": "Weighted dataset/corpus/group-specific Clopper-Pearson components with Bonferroni coverage across strata and repeats.",
    }


def _heterogeneity_cluster_draws(paired: list[dict], clusters: dict[str, str], seed: int, repetitions: int) -> tuple[np.ndarray, int]:
    cluster_keys = sorted({clusters[row["task_id"]] for row in paired})
    cluster_index = {key: i for i, key in enumerate(cluster_keys)}
    sums, sizes = np.zeros((len(cluster_keys), 2)), np.zeros((len(cluster_keys), 2))
    for row in paired:
        i, j = cluster_index[clusters[row["task_id"]]], 0 if row["group"] == "R" else 1
        sums[i, j] += row["arms"]["SEARCH"]["em"] - row["arms"]["THINK"]["em"]
        sizes[i, j] += 1
    indices = np.random.default_rng(seed).integers(len(cluster_keys), size=(repetitions, len(cluster_keys)))
    numerator, denominator = sums[indices].sum(axis=1), sizes[indices].sum(axis=1)
    valid = np.all(denominator > 0, axis=1)
    draws = (numerator[valid] / denominator[valid])[:, 0] - (numerator[valid] / denominator[valid])[:, 1]
    return draws, len(cluster_keys)


def _heterogeneity(records: list[dict], clusters: dict[str, str], seed: int, repetitions: int,
                   support_clusters: dict[str, str] | None = None) -> dict:
    paired = [row for row in records if row["arms"]["SEARCH"]["complete"] and row["arms"]["THINK"]["complete"]]
    groups = {group: [row for row in paired if row["group"] == group] for group in ("R", "N")}
    if any(not rows for rows in groups.values()):
        return {"n_R": len(groups["R"]), "n_N": len(groups["N"]), "mean": None, "ci95": [None, None]}
    arrays = {group: np.array([r["arms"]["SEARCH"]["em"] - r["arms"]["THINK"]["em"] for r in rows])
              for group, rows in groups.items()}
    draws = {group: _bootstrap(arrays[group], [r.get("dataset", "musique") + "/" + r["corpus"] for r in rows], seed + i, repetitions)
             for i, (group, rows) in enumerate(groups.items())}
    cluster_draws, n_clusters = _heterogeneity_cluster_draws(paired, clusters, seed, repetitions)
    support_draws, n_support = _heterogeneity_cluster_draws(paired, support_clusters, seed, repetitions) if support_clusters else (np.array([]), 0)
    return {
        "n_R": len(groups["R"]), "n_N": len(groups["N"]),
        "mean": float(arrays["R"].mean() - arrays["N"].mean()),
        "ci95": [float(x) for x in np.quantile(draws["R"] - draws["N"], [.025, .975])],
        "cluster_ci95": [float(x) for x in np.quantile(cluster_draws, [.025, .975])] if len(cluster_draws) and n_clusters > 1 else [None, None],
        "n_clusters": n_clusters, "valid_cluster_draws": len(cluster_draws),
        "support_page_cluster_ci95": [float(x) for x in np.quantile(support_draws, [.025, .975])] if len(support_draws) and n_support > 1 else [None, None],
        "n_support_page_clusters": n_support,
        "interpretation": "Observed-state treatment heterogeneity; repetition was an observational covariate.",
    }


def _standardized_cluster_draws(paired: list[dict], clusters: dict[str, str], weights: dict[str, float],
                                seed: int, repetitions: int) -> tuple[np.ndarray, int]:
    keys = sorted({clusters[row["task_id"]] for row in paired})
    cells = [(corpus, group) for corpus in weights for group in ("R", "N")]
    key_index, cell_index = {key: i for i, key in enumerate(keys)}, {cell: i for i, cell in enumerate(cells)}
    sums, counts = np.zeros((len(keys), len(cells))), np.zeros((len(keys), len(cells)))
    for row in paired:
        i, j = key_index[clusters[row["task_id"]]], cell_index[(row["corpus"], row["group"])]
        sums[i, j] += row["arms"]["SEARCH"]["em"] - row["arms"]["THINK"]["em"]
        counts[i, j] += 1
    indices = np.random.default_rng(seed).integers(len(keys), size=(repetitions, len(keys)))
    numerator, denominator = sums[indices].sum(axis=1), counts[indices].sum(axis=1)
    valid = np.all(denominator > 0, axis=1)
    signed_weights = np.array([weights[corpus] * (1 if group == "R" else -1) for corpus, group in cells])
    return (numerator[valid] / denominator[valid]) @ signed_weights, len(keys)


def corpus_standardized_heterogeneity(records: list[dict], clusters: dict[str, str], seed: int = 20261014,
                                      repetitions: int = 10000, support_clusters: dict[str, str] | None = None) -> dict:
    """Checkpoint secondary R-N interaction, fixed before independent confirmation."""
    weights = {"clueweb": 0.5, "fineweb": 0.5}
    paired = [row for row in records if row["corpus"] in weights and row["arms"]["SEARCH"]["complete"] and row["arms"]["THINK"]["complete"]]
    cells = {(corpus, group): [row for row in paired if row["corpus"] == corpus and row["group"] == group]
             for corpus in weights for group in ("R", "N")}
    coverage = {corpus + "/" + group: len(rows) for (corpus, group), rows in cells.items()}
    if any(not rows for rows in cells.values()):
        return {"mean": None, "ci95": [None, None], "weights": weights, "cell_n": coverage,
                "cluster_ci95": [None, None], "support_page_cluster_ci95": [None, None]}
    estimate, draws = 0.0, np.zeros(repetitions)
    for index, ((corpus, group), rows) in enumerate(cells.items()):
        values = np.array([row["arms"]["SEARCH"]["em"] - row["arms"]["THINK"]["em"] for row in rows])
        signed_weight = weights[corpus] * (1 if group == "R" else -1)
        estimate += signed_weight * float(values.mean())
        draws += signed_weight * _bootstrap(values, [row.get("dataset", "musique") for row in rows], seed + index, repetitions)
    cluster_draws, n_clusters = _standardized_cluster_draws(paired, clusters, weights, seed, repetitions)
    support_draws, n_support = _standardized_cluster_draws(paired, support_clusters, weights, seed, repetitions) if support_clusters else (np.array([]), 0)
    return {
        "mean": estimate, "ci95": [float(x) for x in np.quantile(draws, [.025, .975])],
        "weights": weights, "cell_n": coverage, "n": len(paired),
        "cluster_ci95": [float(x) for x in np.quantile(cluster_draws, [.025, .975])] if len(cluster_draws) and n_clusters > 1 else [None, None],
        "support_page_cluster_ci95": [float(x) for x in np.quantile(support_draws, [.025, .975])] if len(support_draws) and n_support > 1 else [None, None],
        "n_clusters": n_clusters, "n_support_page_clusters": n_support, "valid_cluster_draws": len(cluster_draws),
        "bootstrap_degenerate": bool(np.ptp(draws) == 0),
        "analysis_timing": "Added at the first-stage checkpoint; weights fixed before independent confirmation.",
        "interpretation": "Secondary observed-state R-N treatment interaction, standardized to equal corpus weights; repetition remains observational.",
    }


def _arm_summary(records: list[dict], arm: str) -> dict:
    complete = [row for row in records if row["arms"][arm]["complete"]]
    flips, distinct = 0, 0
    for row in complete:
        values = [row["arms"][arm]["by_repeat"][repeat] for repeat in row["arms"][arm]["expected_repeats"]]
        flips += len({v["em"] for v in values}) > 1
        distinct += len({normalize(v["answer"]) for v in values}) > 1
    return {
        "n_questions": len(complete),
        "em": float(np.mean([row["arms"][arm]["em"] for row in complete])) if complete else None,
        "f1": float(np.mean([row["arms"][arm]["f1"] for row in complete])) if complete else None,
        "outcome_disagreements": flips, "outcome_disagreement_rate": flips / len(complete) if complete else None,
        "normalized_answer_disagreements": distinct,
        "normalized_answer_disagreement_rate": distinct / len(complete) if complete else None,
    }


def _contrast_rows(records: list[dict], clusters: dict[str, str], seed: int, repetitions: int,
                    support_clusters: dict[str, str] | None = None) -> list[dict]:
    datasets = sorted({row["dataset"] for row in records})
    dataset = datasets[0] if len(datasets) == 1 else "all"
    contrasts = []
    for group in ("R", "N", "all"):
        for corpus in ["all"] + sorted({row["corpus"] for row in records}):
            subset = [row for row in records if (group == "all" or row["group"] == group)
                      and (corpus == "all" or row["corpus"] == corpus)]
            for left, right in COMPARISONS:
                contrasts.append({"dataset": dataset, "group": group, "corpus": corpus, "comparison": f"{left}-{right}"}
                                 | paired_estimate(subset, left, right, clusters, seed, repetitions, support_clusters=support_clusters))
    return contrasts


def _producer_summary(records: list[dict], clusters: dict[str, str], seed: int, repetitions: int,
                       support_clusters: dict[str, str] | None = None) -> dict:
    output = {}
    for producer in sorted({row["producer"] for row in records}):
        subset = [row for row in records if row["producer"] == producer]
        output[producer] = {
            "n_questions": len(subset),
            "contrasts": [
                {"group": group, "corpus": "all", "comparison": "SEARCH-THINK"}
                | paired_estimate([row for row in subset if row["group"] == group], "SEARCH", "THINK", clusters,
                                  seed, repetitions, support_clusters=support_clusters)
                for group in ("R", "N")],
            "interpretation": "Descriptive producer strata; source production differed across historical and newly collected states.",
        }
    return output


def generation_length_summary(results: list[dict], config: dict) -> dict:
    """Separate permitted-search response limits from final-answer budget limits."""
    failures = []
    for result in results:
        if result["status"] != "generation_length" or not result.get("calls"):
            continue
        last = result["calls"][-1]
        step = int(last["call_id"].rsplit("__step", 1)[1])
        permitted_search = result["arm"] == "SEARCH" and step < 2
        cap = result["generation_cap"]
        previous = sum(call["usage"].get("completion_tokens", 0) for call in result["calls"][:-1])
        response_cap = min(cap - previous, config.get("search_decision_call_cap", 3072)) if permitted_search else cap - previous
        failures.append({
            "job_id": result["job_id"], "model": result["model"], "arm": result["arm"],
            "response_stage": "search_permitted" if permitted_search else "final_answer",
            "calls": len(result["calls"]), "extra_searches": result.get("extra_searches", 0),
            "generation_cap": cap, "used_completion_tokens": result["usage"]["completion_tokens"],
            "unused_generation_cap": cap - result["usage"]["completion_tokens"],
            "last_response_cap": response_cap, "last_response_used": last["usage"].get("completion_tokens", 0),
        })
    return {"counts": dict(Counter(row["model"] + "/" + row["arm"] + "/" + row["response_stage"] for row in failures)),
            "failures": sorted(failures, key=lambda row: row["job_id"]),
            "interpretation": "Each length outcome retains its frozen zero score. Unused total cap distinguishes subrequest allocation from final-answer truncation."}


def _cost_summary(scored: list[dict], results: dict[str, dict], jobs: list[dict], attempts: list[dict]) -> dict:
    expected_ids = {job["job_id"] for job in jobs}

    def owner(record: dict) -> str | None:
        call_id = record.get("job_id", "")
        if call_id in expected_ids:
            return call_id
        job_id = re.sub(r"__step\d+$", "", call_id)
        return job_id if job_id in expected_ids else None

    def response_usage(ids) -> dict:
        ids = tuple(ids)
        return {token: sum(ends.get(key, {}).get("usage", {}).get(token, 0) or 0 for key in ids)
                for token in ("prompt_tokens", "completion_tokens", "total_tokens")}

    starts = {record["attempt_id"]: record for record in attempts if record.get("event") == "start"}
    ends = {record["attempt_id"]: record for record in attempts if record.get("event") == "end"}
    run_starts = {key: value for key, value in starts.items() if owner(value) is not None}
    summaries = []
    for model, arm in sorted({(job["model"], job["arm"]) for job in jobs}):
        rows = [results[row["job_id"]] for row in scored if row["model"] == model and row["arm"] == arm]
        usage = {token: sum(row.get("usage", {}).get(token, 0) or 0 for row in rows)
                 for token in ("prompt_tokens", "completion_tokens", "total_tokens")}
        job_ids = {job["job_id"] for job in jobs if job["model"] == model and job["arm"] == arm}
        relevant = {key: value for key, value in run_starts.items() if owner(value) in job_ids}
        model_attempts = [key for key, value in relevant.items() if value.get("kind") == "model"]
        search_attempts = [key for key, value in relevant.items() if value.get("kind") == "search"]
        summaries.append({
            "model": model, "arm": arm, "n_saved_results": len(rows),
            "n_finished_jobs": sum(row["status"] != "checkpoint_pending" for row in rows), "usage": usage,
            "ledger_response_usage": response_usage(model_attempts),
            "mean_total_tokens": usage["total_tokens"] / len(rows) if rows else None,
            "mean_completion_tokens": usage["completion_tokens"] / len(rows) if rows else None,
            "extra_searches": sum(row.get("extra_searches", 0) for row in rows),
            "logical_searches": sum(row.get("logical_searches", 0) for row in rows),
            "cache_hits": sum(row.get("cache_hits", 0) for row in rows),
            "model_http_attempts": len(model_attempts), "search_gateway_attempts": len(search_attempts),
            "model_http_seconds_sum": sum(ends.get(key, {}).get("seconds", 0) for key in model_attempts),
            "search_gateway_seconds_sum": sum(ends.get(key, {}).get("seconds", 0) for key in search_attempts),
            "search_gateway_successes": sum(200 <= ends.get(key, {}).get("status_code", 0) < 300 for key in search_attempts),
            "seconds_sum": sum(row.get("seconds", 0) for row in rows),
            "mean_seconds": float(np.mean([row.get("seconds", 0) for row in rows])) if rows else None,
        })
    by_phase = {}
    for phase in sorted({job.get("phase", "main") for job in jobs}):
        phase_ids = {job["job_id"] for job in jobs if job.get("phase", "main") == phase}
        phase_starts = {key: record for key, record in run_starts.items() if owner(record) in phase_ids}
        by_phase[phase] = {
            "planned_jobs": len(phase_ids), "saved_results": sum(job_id in results for job_id in phase_ids),
            "finished_jobs": sum(job_id in results and results[job_id]["status"] != "checkpoint_pending" for job_id in phase_ids),
            "attempts": dict(Counter(record.get("kind", "unknown") for record in phase_starts.values())),
            "ledger_response_usage": response_usage(key for key, record in phase_starts.items() if record.get("kind") == "model"),
        }
    return {
        "by_model_arm": summaries, "by_phase": by_phase,
        "run_ledger_response_usage": response_usage(key for key, record in run_starts.items() if record.get("kind") == "model"),
        "global_ledger_response_usage": response_usage(key for key, record in starts.items() if record.get("kind") == "model"),
        "run_associated_attempts": dict(Counter(record.get("kind", "unknown") for record in run_starts.values())),
        "global_reserved_attempts": dict(Counter(record.get("kind", "unknown") for record in starts.values())),
        "run_attempts_without_completion_record": sum(key not in ends for key in run_starts),
        "ledger_found": bool(attempts),
        "token_scope": "Saved-result tokens include checkpoint work; ledger response usage accounts for reported tokens across every attributed HTTP attempt. Unreadable or timed-out response usage is unavailable.",
        "time_scope": "Saved branch elapsed time covers its last execution, including cached replay. Ledger HTTP durations include attributed attempts across resumes; concurrent wall time requires timestamp analysis.",
    }


def _format_number(value: float | None) -> str:
    return "pending" if value is None else f"{100 * value:.2f}"


def _format_interval(value: list) -> str:
    return "pending" if value[0] is None else f"[{100 * value[0]:.2f}, {100 * value[1]:.2f}]"


def _report(summary: dict) -> str:
    lines = ["# SearchHorizon frozen-state experiment", "",
             f"Completed {summary['coverage']['finished_jobs']} of {summary['coverage']['expected_jobs']} planned branch jobs. "
             f"The frozen manifest contains {summary['design']['n_questions']} questions in "
             f"{summary['design']['n_subquestion_clusters']} connected subquestion groups.", "",
             "The primary estimand is SEARCH minus THINK EM in repetition states. Each question contributes its mean across "
             "the frozen generation repeats. Task bootstrap draws preserve dataset, corpus and repetition strata; connected-group "
             "bootstraps provide dependence sensitivity analyses. Dataset-specific results are the primary cross-dataset evidence. "
             "A mixed-dataset pool weights questions equally under the frozen allocation and is descriptive. "
             "Scores and differences below use EM percentage points.", "",
             "## Dataset-specific results", "",
             "| Model | Dataset | Corpus | Repetition questions | SEARCH−THINK | Task 95% CI | Page groups | Support-page 95% CI |",
             "| --- | --- | --- | ---: | ---: | --- | ---: | --- |"]
    for model, details in summary["models"].items():
        for dataset, dataset_details in details["by_dataset"].items():
            for row in dataset_details["contrasts"]:
                if row["group"] == "R" and row["comparison"] == "SEARCH-THINK":
                    lines.append(f"| {model} | {dataset} | {row['corpus']} | {row['n']} | {_format_number(row['mean'])} | "
                                 f"{_format_interval(row['ci95'])} | {row['n_support_page_clusters']} | {_format_interval(row['support_page_cluster_ci95'])} |")
    lines += ["", "## All frozen contrasts", "",
             "| Model | Dataset | Group | Corpus | Comparison | Questions | EM difference | EM task 95% CI | F1 difference | F1 task 95% CI | Group 95% CI | Conservative paired CI |",
             "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |"]
    for model, details in summary["models"].items():
        scopes = dict(details["by_dataset"])
        if details["pooled_estimate_descriptive"]:
            scopes["all (descriptive)"] = details
        for dataset, dataset_details in scopes.items():
            for record in dataset_details["contrasts"]:
                lines.append(f"| {model} | {dataset} | {record['group']} | {record['corpus']} | {record['comparison']} | {record['n']} | "
                             f"{_format_number(record['mean'])} | {_format_interval(record['ci95'])} | "
                             f"{_format_number(record['f1_mean'])} | {_format_interval(record['f1_ci95'])} | "
                             f"{_format_interval(record['cluster_ci95'])} | {_format_interval(record['conservative_paired_ci95'])} |")
    lines += ["", "## Within-corpus and standardized R-N interactions", "",
              "The raw pooled R-N interaction is descriptive when the two groups have different corpus allocations. "
              "The equal-corpus standardization was added at the first-stage checkpoint and fixed before independent confirmation. "
              "It gives ClueWeb and FineWeb weight 0.5 each.", "",
              "| Model | Dataset | Corpus | R questions | N questions | Interaction | Task 95% CI |",
              "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for model, details in summary["models"].items():
        for dataset, dataset_details in details["by_dataset"].items():
            for corpus, interaction in dataset_details["heterogeneity_by_corpus"].items():
                lines.append(f"| {model} | {dataset} | {corpus} | {interaction['n_R']} | {interaction['n_N']} | "
                             f"{_format_number(interaction['mean'])} | {_format_interval(interaction['ci95'])} |")
            standardized = dataset_details["corpus_standardized_heterogeneity"]
            r = sum(value for key, value in standardized["cell_n"].items() if key.endswith("/R"))
            n = sum(value for key, value in standardized["cell_n"].items() if key.endswith("/N"))
            lines.append(f"| {model} | {dataset} | 0.5 CW / 0.5 FW | {r} | {n} | "
                         f"{_format_number(standardized['mean'])} | {_format_interval(standardized['ci95'])} |")
    lines += ["", "## Producer strata", "",
              "Producer strata describe the sampled states. Source policies and corpus allocations differ; producer contrasts remain descriptive.", "",
              "| Model | Dataset | Producer | Group | Corpus | Questions | SEARCH−THINK EM | EM 95% CI | F1 difference | F1 95% CI |",
              "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |"]
    for model, details in summary["models"].items():
        for dataset, dataset_details in details["by_dataset"].items():
            for producer, producer_details in dataset_details["by_producer"].items():
                for row in producer_details["contrasts"]:
                    lines.append(f"| {model} | {dataset} | {producer} | {row['group']} | {row['corpus']} | {row['n']} | "
                                 f"{_format_number(row['mean'])} | {_format_interval(row['ci95'])} | "
                                 f"{_format_number(row['f1_mean'])} | {_format_interval(row['f1_ci95'])} |")
    lines += ["", "## Rescue, harm, and generation variation", "",
              "Repeat identifiers define fixed output pairs. The API calls use independent generation noise. A rescue or harm "
              "count describes these paired realizations; within-arm correctness disagreement measures generation variation. "
              "A stable question-level search benefit requires repeated evidence. Per-arm descriptives use each arm's completed "
              "questions. Paired contrasts above use the common completed question pairs for treatment comparisons.", "",
              "| Model | Arm | Complete questions | EM | F1 | Correctness disagreement | Answer disagreement |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model, details in summary["models"].items():
        for arm, stats in details["arms"].items():
            lines.append(f"| {model} | {arm} | {stats['n_questions']} | {_format_number(stats['em'])} | "
                         f"{_format_number(stats['f1'])} | "
                         f"{_format_number(stats['outcome_disagreement_rate'])} | {_format_number(stats['normalized_answer_disagreement_rate'])} |")
    for model, details in summary["models"].items():
        primary = next((r for r in details["contrasts"] if r["group"] == "R" and r["corpus"] == "all" and r["comparison"] == "SEARCH-THINK"), None)
        if primary:
            lines += ["", f"{model}, repetition-state SEARCH versus THINK:", ""]
            for outcome in primary["repeat_outcomes"]:
                bound = ""
                if outcome["prespecified_first_repeat_bound"]:
                    bound = f"; independence-based first-repeat rescue upper limit {_format_number(outcome['rescue_upper95'])}%"
                    if not outcome["independent_question_bound"]:
                        bound += " (shared-page dependence leaves this assumption unmet)"
                lines.append(f"- Repeat {outcome['repeat']}: {outcome['rescue']} rescues, {outcome['harm']} harms, "
                             f"{outcome['both_correct']} pairs correct, {outcome['both_wrong']} pairs wrong{bound}.")
    lines += ["", "## Cost", "",
              "THINK and SEARCH share the generation-token cap. SEARCH can add input context and repeat prompt processing. "
              "The saved response token counts and request attempts quantify these costs separately. Saved execution cache hits "
              "describe the last execution of each branch, including cache reuse on resume. The gateway ledger records cumulative "
              "HTTP attempts and supplies the authoritative physical-request cost.", "",
              "| Model | Arm | Finished jobs | Input tokens | Completion tokens | Total tokens | Logical searches | Saved execution cache hits | Model attempts | Gateway attempts |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cost in summary["costs"]["by_model_arm"]:
        u = cost["usage"]
        lines.append(f"| {cost['model']} | {cost['arm']} | {cost['n_finished_jobs']} | {u['prompt_tokens']} | "
                     f"{u['completion_tokens']} | {u['total_tokens']} | {cost['logical_searches']} | "
                     f"{cost['cache_hits']} | {cost['model_http_attempts']} | {cost['search_gateway_attempts']} |")
    lines += ["", "## Interpretation and audit", "",
              "Equivalence uses the preregistered ±5 percentage-point margin and the conservative paired risk-difference "
              "interval. A degenerate task bootstrap alone supports no equivalence statement. Exact component intervals "
              "assume independent questions sampled within the frozen strata; connected subquestion or support-page overlap disables a formal equivalence label. "
              "Support-page title clusters provide an additional dependence sensitivity for HotpotQA, whose task IDs alone "
              "represent individual questions. A cluster interval requires at least two connected groups.", "",
              "Completed algorithmic failures receive zero scores. Infrastructure errors, global budget exhaustion, and checkpoint-paused jobs remain "
              "unevaluable in primary comparisons; the all-finished-as-zero sensitivity retains their observed zero scores. "
              "Outstanding jobs remain pending. Checkpoint-paused outcomes stay outside the completed-as-zero sensitivity. "
              "Every paired estimate reports its own complete-question count.", "",
              "Repetition/nonrepetition quotas define a diagnostic sample. Their contrast describes observed-state treatment "
              "heterogeneity. The second-model question scope follows its frozen manifest and supports a conditional robustness check. "
              "Secondary intervals remain unadjusted descriptive intervals; this report makes no multiplicity-adjusted significance claims.", "",
              "The experimental permission to search includes query generation, retrieved evidence, and additional model turns. "
              "SEARCH minus THINK estimates the value of that controlled continuation policy under the stated cap.", ""]
    for issue in summary["audit"]["issues"]:
        lines.append(f"- {issue}")
    return "\n".join(lines) + "\n"


def analyze(run_dir: str | Path, config: dict | None = None) -> dict:
    """Write reproducible scoring/statistics artifacts for a complete or partial run."""
    run_dir = Path(run_dir)
    config = config or json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    seed, repetitions = int(config.get("seed", 20261014)), int(config.get("bootstrap_repetitions", 10000))
    nodes = _read_jsonl(run_dir / "data" / "nodes.jsonl")
    labels = json.loads((run_dir / "data" / "labels.json").read_text(encoding="utf-8"))
    jobs = _read_jsonl(run_dir / "jobs.jsonl")
    if len({node["task_id"] for node in nodes}) != len(nodes):
        raise ValueError("The frozen manifest requires one node per original question")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("The frozen job list contains duplicate job identifiers")
    node_map = {node["node_id"]: node for node in nodes}
    job_map = {job["job_id"]: job for job in jobs}
    for node in nodes:
        if node["task_id"] not in labels or not labels[node["task_id"]].get("answers"):
            raise ValueError(f"Missing frozen answer labels for {node['task_id']}")
    results, unexpected = {}, []
    for path in sorted((run_dir / "results").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        job_id = result["job_id"]
        if job_id not in job_map:
            unexpected.append(job_id)
            continue
        if job_id in results:
            raise ValueError(f"Duplicate saved result for job {job_id}")
        for key in ("node_id", "task_id", "model", "arm", "repeat"):
            if result[key] != job_map[job_id][key]:
                raise ValueError(f"Result {job_id} differs from its frozen job field {key}")
        if "phase" in job_map[job_id] and result.get("phase") != job_map[job_id]["phase"]:
            raise ValueError(f"Result {job_id} differs from its frozen job field phase")
        results[job_id] = result
    scored = []
    for job in jobs:
        result = results.get(job["job_id"])
        if result is None:
            continue
        label = labels[result["task_id"]]
        node = node_map[result["node_id"]]
        answer = result.get("answer", "") or ""
        scored.append({key: result[key] for key in ("job_id", "node_id", "task_id", "model", "arm", "repeat", "status")}
                      | {"answer": answer, "corpus": node["corpus"], "group": node["group"], "phase": job.get("phase", "main"),
                         "dataset": node.get("dataset", "musique"), "producer": node.get("provenance", {}).get("kind", "historical")}
                      | score(answer, label["answers"], result["status"], label.get("metric", "squad")))
    records = aggregate_question_scores(scored, jobs, nodes)
    clusters = _task_clusters(nodes, labels)
    support_clusters = _task_clusters(nodes, labels, "supporting_page_titles")
    page_coverage = sum(bool(labels[node["task_id"]].get("supporting_page_titles")) for node in nodes)
    support_sensitivity = support_clusters if page_coverage else None
    models = {}
    for model in sorted({job["model"] for job in jobs}):
        selected = [row for row in records if row["model"] == model]
        contrasts = _contrast_rows(selected, clusters, seed, repetitions, support_sensitivity)
        sensitivities = {}
        for threshold in (0.5, 0.7):
            recategorized = [row | {"group": "R" if row["jaccard"] >= threshold else "N"}
                             for row in selected if row["jaccard"] is not None]
            sensitivities[f"jaccard_{threshold}"] = {
                group: paired_estimate([row for row in recategorized if row["group"] == group], "SEARCH", "THINK", clusters, seed, repetitions,
                                       support_clusters=support_sensitivity)
                for group in ("R", "N")}
        low_novelty = [row for row in selected if row["group"] == "R" and row["novelty"] is not None and row["novelty"] <= 0.3]
        sensitivities["repetition_low_novelty"] = paired_estimate(low_novelty, "SEARCH", "THINK", clusters, seed, repetitions, support_clusters=support_sensitivity)
        sensitivities["identical_query"] = paired_estimate([row for row in selected if row["identical_query"]], "SEARCH", "THINK", clusters, seed, repetitions,
                                                          support_clusters=support_sensitivity)
        all_finished = []
        for row in selected:
            arms = {arm: value | {"complete": value["all_finished"], "em": value["all_finished_em"]} for arm, value in row["arms"].items()}
            for value in arms.values():
                if value["all_finished"]:
                    value["f1"] = float(np.mean([value["by_repeat"][repeat]["f1"] for repeat in value["expected_repeats"]]))
            all_finished.append(row | {"arms": arms})
        sensitivities["all_finished_as_zero_R"] = paired_estimate([row for row in all_finished if row["group"] == "R"], "SEARCH", "THINK", clusters, seed, repetitions,
                                                                support_clusters=support_sensitivity)
        by_dataset = {}
        datasets = sorted({row["dataset"] for row in selected})
        for dataset in datasets:
            dataset_rows = [row for row in selected if row["dataset"] == dataset]
            by_dataset[dataset] = {
                "arms": {arm: _arm_summary(dataset_rows, arm) for arm in ARMS},
                "contrasts": contrasts if len(datasets) == 1 else _contrast_rows(dataset_rows, clusters, seed, repetitions, support_sensitivity),
                "heterogeneity_SEARCH_THINK_R_minus_N": _heterogeneity(dataset_rows, clusters, seed, repetitions, support_sensitivity),
                "heterogeneity_by_corpus": {corpus: _heterogeneity([row for row in dataset_rows if row["corpus"] == corpus],
                                                                  clusters, seed, repetitions, support_sensitivity)
                                            for corpus in sorted({row["corpus"] for row in dataset_rows})},
                "corpus_standardized_heterogeneity": corpus_standardized_heterogeneity(dataset_rows, clusters, seed, repetitions, support_sensitivity),
                "by_producer": _producer_summary(dataset_rows, clusters, seed, repetitions, support_sensitivity),
            }
        models[model] = {
            "arms": {arm: _arm_summary(selected, arm) for arm in ARMS}, "contrasts": contrasts,
            "heterogeneity_SEARCH_THINK_R_minus_N": _heterogeneity(selected, clusters, seed, repetitions, support_sensitivity),
            "sensitivity": sensitivities, "by_dataset": by_dataset,
            "heterogeneity_by_corpus": {corpus: _heterogeneity([row for row in selected if row["corpus"] == corpus],
                                                              clusters, seed, repetitions, support_sensitivity)
                                        for corpus in sorted({row["corpus"] for row in selected})},
            "corpus_standardized_heterogeneity": corpus_standardized_heterogeneity(selected, clusters, seed, repetitions, support_sensitivity),
            "raw_pooled_interaction_descriptive": True,
            "by_producer": _producer_summary(selected, clusters, seed, repetitions, support_sensitivity),
            "pooled_estimate_descriptive": len(datasets) > 1,
            "pooling": "Equal per-question weights under the frozen dataset/corpus/group allocation.",
        }
    ledger = Path(config.get("global_attempts", run_dir.parent / "attempts.jsonl"))
    attempts = _read_jsonl(ledger)
    issues = []
    pending = [job["job_id"] for job in jobs if job["job_id"] not in results or results[job["job_id"]]["status"] == "checkpoint_pending"]
    status_counts = dict(Counter(row["status"] for row in scored))
    if pending:
        issues.append(f"{len(pending)} planned branch jobs await saved outcomes.")
    if sum(status_counts.get(status, 0) for status in MISSING_STATUSES):
        issues.append("Infrastructure/global-budget outcomes reduce primary complete-pair coverage; inspect the per-status counts.")
    if status_counts.get("protocol_error", 0):
        issues.append(f"{status_counts['protocol_error']} protocol_error outcomes require an execution audit before scientific interpretation.")
    if unexpected:
        issues.append(f"{len(unexpected)} saved outcomes lie outside the frozen job list and were excluded.")
    if len(set(clusters.values())) < len(nodes):
        issues.append("Shared base subquestions create dependent task groups; use the connected-group sensitivity intervals.")
    if len(set(support_clusters.values())) < len(nodes):
        issues.append("Shared supporting pages connect questions; inspect the support-page cluster sensitivity before inference.")
    model_versions = defaultdict(set)
    for result in results.values():
        for call in result.get("calls", []):
            if call.get("actual_model"):
                model_versions[result["model"]].add(call["actual_model"])
    for model, versions in model_versions.items():
        if len(versions) > 1:
            issues.append(f"{model} has multiple actual response model identifiers: {sorted(versions)}; separate version strata before interpretation.")
    summary = {
        "design": {"primary_model": config.get("primary_model", "deepseek-flash"), "primary_comparison": "SEARCH-THINK",
                   "primary_group": "R", "n_questions": len(nodes), "n_subquestion_clusters": len(set(clusters.values())),
                   "n_support_page_clusters": len(set(support_clusters.values())), "support_page_label_coverage": page_coverage,
                   "questions_by_dataset": dict(Counter(node.get("dataset", "musique") for node in nodes)),
                   "dataset_cells": dict(Counter(node.get("dataset", "musique") + "/" + node["corpus"] + "/" + node["group"] for node in nodes)),
                   "producer_counts": dict(Counter(node.get("provenance", {}).get("kind", "historical") for node in nodes)),
                   "cells": dict(Counter(node["corpus"] + "/" + node["group"] for node in nodes)),
                   "bootstrap_seed": seed, "bootstrap_repetitions": repetitions, "equivalence_margin": 0.05,
                   "scores": "Frozen-label normalized EM; token F1 supplementary.",
                   "completion_scope": "Complete frozen repeat pairs, with algorithmic failures scored zero."},
        "coverage": {"expected_jobs": len(jobs), "saved_outcomes": len(scored),
                     "finished_jobs": sum(row["status"] != "checkpoint_pending" for row in scored), "pending_jobs": len(pending),
                     "primary_evaluable_jobs": sum(row["status"] not in MISSING_STATUSES for row in scored),
                     "pending_job_ids": pending, "status_counts": status_counts,
                     "finished_by_model_arm": dict(Counter(row["model"] + "/" + row["arm"] for row in scored if row["status"] != "checkpoint_pending")),
                     "planned_by_phase": dict(Counter(job.get("phase", "main") for job in jobs)),
                     "finished_by_phase": dict(Counter(row["phase"] for row in scored if row["status"] != "checkpoint_pending")),
                     "planned_by_dataset": dict(Counter(node_map[job["node_id"]].get("dataset", "musique") for job in jobs)),
                     "finished_by_dataset": dict(Counter(row["dataset"] for row in scored if row["status"] != "checkpoint_pending"))},
        "models": models, "costs": _cost_summary(scored, results, jobs, attempts),
        "audit": {"issues": issues, "unexpected_result_job_ids": unexpected,
                  "generation_length": generation_length_summary(list(results.values()), config),
                  "actual_model_identifiers": {key: sorted(values) for key, values in model_versions.items()},
                  "paired_repeat_noise": "Independent API calls; repeat identifiers fix output pairing.",
                  "multiplicity": "Secondary estimates are descriptive; unadjusted intervals carry no formal significance label."},
    }
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    (output / "scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8")
    _write_json(output / "summary.json", summary)
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    with (output / "task_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "phase", "dataset", "producer", "task_id", "node_id", "corpus", "group", "cluster_id", "support_page_cluster_id", "jaccard", "novelty"]
        fields += [f"{arm}_{key}" for arm in ARMS for key in ("em", "f1", "n_finished", "n_evaluable")]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            flat = {key: row[key] for key in fields if key in row}
            flat["cluster_id"] = clusters[row["task_id"]]
            flat["support_page_cluster_id"] = support_clusters[row["task_id"]]
            flat.update({f"{arm}_{key}": row["arms"][arm][key] for arm in ARMS for key in ("em", "f1", "n_finished", "n_evaluable")})
            writer.writerow(flat)
    with (output / "contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "dataset", "producer", "group", "corpus", "comparison", "n", "mean", "ci95_lower", "ci95_upper",
                  "cluster_lower", "cluster_upper", "n_clusters", "support_page_lower", "support_page_upper", "n_support_page_clusters",
                  "paired_lower", "paired_upper", "equivalence_supported"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, details in models.items():
            rows = [row | {"producer": "all"} for row in details["contrasts"]]
            if details["pooled_estimate_descriptive"]:
                rows += [row | {"producer": "all"} for dataset in details["by_dataset"].values() for row in dataset["contrasts"]]
            for dataset, dataset_details in details["by_dataset"].items():
                rows += [row | {"dataset": dataset, "producer": producer} for producer, producer_details in dataset_details["by_producer"].items()
                         for row in producer_details["contrasts"]]
            for row in rows:
                writer.writerow({"model": model} | {key: row[key] for key in ("dataset", "producer", "group", "corpus", "comparison", "n", "mean", "n_clusters", "n_support_page_clusters", "equivalence_supported")}
                                | {"ci95_lower": row["ci95"][0], "ci95_upper": row["ci95"][1],
                                   "cluster_lower": row["cluster_ci95"][0], "cluster_upper": row["cluster_ci95"][1],
                                   "support_page_lower": row["support_page_cluster_ci95"][0], "support_page_upper": row["support_page_cluster_ci95"][1],
                                   "paired_lower": row["conservative_paired_ci95"][0], "paired_upper": row["conservative_paired_ci95"][1]})
    return summary
