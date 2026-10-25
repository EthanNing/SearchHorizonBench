"""Score cost-threshold search decisions using standalone exported outcomes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search_horizon.io import rows, write
from scripts.score_value_forecasts import balanced_component_weights, evaluate, strata_have_replication


def evaluate_policy(draws, predictions, actor_model, mean_extra_tokens, penalty=.005,
                    bootstrap=10000, seed=2026101815, weighting='equal_strata'):
    if not np.isfinite(mean_extra_tokens) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError('Cost threshold and nonnegative token valuation must be finite.')
    if weighting not in ('equal_strata', 'questions'):
        raise ValueError('Weighting must be questions or equal_strata.')
    # The common scorer checks complete IDs, finite forecasts and shared rewards.
    evaluate(draws, predictions, actor_model, bootstrap=1, seed=seed)
    frame = [u for u in draws if u['actor_model'] == actor_model]
    lookup = {p['unit_id']: p for p in predictions}; selected = []; overhead = []
    rewards = []; costs = []
    for unit in frame:
        pred = lookup[unit['unit_id']]
        delta = float(pred['delta']) if pred.get('status', 'ok') == 'ok' else 0.
        selected.append(int(delta > penalty * mean_extra_tokens / 1000))
        value = float(pred['inference_API_tokens'])
        if not np.isfinite(value) or value < 0: raise ValueError('Prediction API tokens must be finite and nonnegative.')
        overhead.append(value)
        reward = []; cost = []
        for arm in ('reuse', 'new'):
            observed = unit['rewards'][arm]; resources = unit['reported_resources'][arm]
            if len(observed) != len(resources): raise ValueError('Each draw needs a corresponding resource record.')
            reward.append(np.mean([0. if v is None else v for v in observed]))
            tokens = [float(r['usage']['total_tokens']) for r in resources]
            if not all(np.isfinite(v) and v >= 0 for v in tokens): raise ValueError('Actor tokens must be finite and nonnegative.')
            cost.append(np.mean(tokens))
        rewards.append(reward); costs.append(cost)
    choice = np.array(selected); overhead = np.array(overhead)
    rewards = np.array(rewards); costs = np.array(costs); n = len(frame)
    if weighting == 'equal_strata':
        point, weights, redrawn = balanced_component_weights(frame, bootstrap, seed)
        intervals_estimable = strata_have_replication(frame)
    else:
        components = sorted({u['support_component'] for u in frame}); component_index = {c: i for i, c in enumerate(components)}
        counts = np.random.default_rng(seed).multinomial(len(components), np.full(len(components), 1 / len(components)), bootstrap)
        weights = counts[:, [component_index[u['support_component']] for u in frame]].astype(float)
        weights /= weights.sum(axis=1, keepdims=True); point = np.full(n, 1 / n); redrawn = 0
        intervals_estimable = len(components) >= 2
    def describe(values):
        if not intervals_estimable: return {'mean': float(point @ values)}
        samples = weights @ values
        lower, upper = np.quantile(samples, [.025, .975])
        return {'mean': float(point @ values), 'bootstrap95': [float(lower), float(upper)],
                'one_sided97_5_lower': float(lower), 'one_sided97_5_upper': float(upper)}
    chosen_reward = rewards[np.arange(n), choice]; chosen_cost = costs[np.arange(n), choice]
    utility = rewards - penalty * costs / 1000
    chosen_utility = chosen_reward - penalty * (chosen_cost + overhead) / 1000
    rate = float(point @ choice); boot_rate = weights @ choice
    random_samples = weights @ chosen_utility - ((1 - boot_rate) * (weights @ utility[:, 0]) + boot_rate * (weights @ utility[:, 1]))
    result = {'SEARCH_fraction': describe(choice), 'success': describe(chosen_reward),
        'actor_tokens': describe(chosen_cost), 'predictor_API_tokens': describe(overhead),
        'total_API_tokens': describe(chosen_cost + overhead), 'utility': describe(chosen_utility),
        'utility_vs_matched_rate_random': {
            'mean': float(point @ chosen_utility - ((1 - rate) * (point @ utility[:, 0]) + rate * (point @ utility[:, 1]))),
            **({'bootstrap95': np.quantile(random_samples, [.025, .975]).tolist()} if intervals_estimable else {})}}
    for baseline, action in enumerate(('THINK', 'SEARCH')):
        result['utility_vs_' + action] = describe(chosen_utility - utility[:, baseline])
        result['success_vs_' + action] = describe(chosen_reward - rewards[:, baseline])
        bounds = []
        for i, unit in enumerate(frame):
            known = -penalty * (chosen_cost[i] + overhead[i] - costs[i, baseline]) / 1000
            unknown = defaultdict(float)
            for arm_index, arm in enumerate(('reuse', 'new')):
                coefficient = (int(choice[i] == arm_index) - int(baseline == arm_index)) / len(unit['rewards'][arm])
                for reward, case, resource in zip(unit['rewards'][arm], unit['reward_cases'][arm], unit['reported_resources'][arm]):
                    if resource['status'] != 'ok' or resource.get('known_nonanswer', False): continue
                    if reward is None: unknown[case] += coefficient
                    else: known += coefficient * reward
            bounds.append([known + sum(min(0., v) for v in unknown.values()),
                           known + sum(max(0., v) for v in unknown.values())])
        bounds = np.array(bounds)
        result['shared_judge_bounds_vs_' + action] = {'lower': describe(bounds[:, 0]), 'upper': describe(bounds[:, 1])}
    return {'actor_model': actor_model, 'questions': n, 'weighting': weighting,
        'intervals_estimable': intervals_estimable,
        'mean_extra_actor_tokens': mean_extra_tokens, 'token_penalty_per1000': penalty,
        'empty_stratum_bootstrap_draws_redrawn': redrawn, 'metrics': result,
        'scope': 'One frozen threshold decision on saved branches. API-token utility charges supplied prediction and actor tokens; CPU time and acquisition expenses remain separate.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draws', required=True); parser.add_argument('--predictions', required=True)
    parser.add_argument('--actor-model', required=True); parser.add_argument('--mean-extra-tokens', type=float, required=True)
    parser.add_argument('--penalty', type=float, default=.005); parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026101815)
    parser.add_argument('--weighting', choices=('questions', 'equal_strata'), default='equal_strata')
    parser.add_argument('--output'); args = parser.parse_args()
    result = evaluate_policy(rows(args.draws), rows(args.predictions), args.actor_model, args.mean_extra_tokens,
        args.penalty, args.bootstrap, args.seed, args.weighting)
    if args.output: write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
