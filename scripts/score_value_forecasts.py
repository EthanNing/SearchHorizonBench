"""Score external signed-value forecasts against exported repeated outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from search_horizon.effect_prediction import effect_moments
from search_horizon.io import read,rows,write


def balanced_component_weights(frame, bootstrap, seed):
    """Keep each observed stratum equally weighted while resampling components."""
    if not all('stratum' in u and 'support_component' in u for u in frame):
        raise ValueError('Equal-stratum scoring requires stratum and support_component on every unit.')
    groups = sorted({u['stratum'] for u in frame})
    components = sorted({u['support_component'] for u in frame})
    component_index = {c: i for i, c in enumerate(components)}
    ids = [component_index[u['support_component']] for u in frame]
    masks = [np.array([u['stratum'] == g for u in frame]) for g in groups]
    point = np.zeros(len(frame))
    for mask in masks: point[mask] = 1 / (len(groups) * mask.sum())
    rng = np.random.default_rng(seed); batches = []; remaining = bootstrap; rejected = 0
    while remaining:
        counts = rng.multinomial(len(components), np.full(len(components), 1 / len(components)), remaining)
        weights = counts[:, ids].astype(float)
        totals = np.column_stack([weights[:, mask].sum(axis=1) for mask in masks])
        valid = np.all(totals > 0, axis=1); rejected += int((~valid).sum())
        weights = weights[valid]; totals = totals[valid]
        for j, mask in enumerate(masks): weights[:, mask] /= len(groups) * totals[:, j, None]
        batches.append(weights); remaining -= len(weights)
    return point, np.concatenate(batches), rejected


def strata_have_replication(frame):
    return all(len({u['support_component'] for u in frame if u['stratum'] == group}) >= 2
               for group in {u['stratum'] for u in frame})


def evaluate(draws,predictions,actor_model,constant=0.0,bootstrap=10000,seed=2026101511,probability_priors=None,weighting='questions'):
    if weighting not in ('questions', 'equal_strata'):
        raise ValueError('Weighting must be questions or equal_strata.')
    all_ids={u['unit_id'] for u in draws}
    frame=[u for u in draws if u['actor_model']==actor_model]
    if not frame or len(all_ids)!=len(draws):
        raise ValueError('Select an existing actor frame with unique unit IDs.')
    by_id={p['unit_id']:p for p in predictions}
    if len(by_id)!=len(predictions) or not set(by_id)<=all_ids:
        raise ValueError('Forecast IDs must be unique and belong to the exported data.')
    if not {u['unit_id'] for u in frame}<=by_id.keys():
        raise ValueError('Every unit in the selected actor frame needs a forecast record.')
    probability_method=any('p_reuse_correct' in by_id[u['unit_id']] or 'p_new_correct' in by_id[u['unit_id']] for u in frame)
    values=[]
    for u in frame:
        p=by_id[u['unit_id']]
        if p.get('status') in ('infra_error','budget_exhausted','checkpoint_pending'):
            raise ValueError('Transport-pending forecasts need repair or an explicitly incomplete report.')
        f=float(p['delta']) if p.get('status','ok')=='ok' else constant
        if not np.isfinite(f) or not -1<=f<=1:
            raise ValueError('Signed probability forecasts must be finite and lie in[-1,1].')
        outcomes=u['rewards'];case_ids=u.get('reward_cases',{})
        unknown=set()
        for arm in ('new','reuse'):
            for i,y in enumerate(outcomes[arm]):
                if y is None:
                    case=case_ids.get(arm,[])[i] if len(case_ids.get(arm,[]))==len(outcomes[arm]) else None
                    if case is None:raise ValueError('Unresolved rewards need shared case IDs.')
                    unknown.add(case)
                elif not np.isfinite(float(y)) or not 0<=float(y)<=1:
                    raise ValueError('Endpoint rewards must lie in[0,1].')
        def moment(assignment):
            def scores(arm):
                return [float(assignment.get(case_ids[arm][i],0)) if y is None else float(y)
                        for i,y in enumerate(outcomes[arm])]
            return effect_moments(scores('new'),scores('reuse'),forecast=f,
                                  identical_requests=u.get('identical_requests',False))
        point=moment({});alternatives=[]
        ordered=sorted(unknown)
        for assignment in itertools.product((0,1),repeat=len(ordered)):
            m=moment(dict(zip(ordered,assignment)))
            alternatives.append(m['risk_difference_vs_zero_forecast']-(constant*constant-2*constant*m['mean_effect']))
        d=point['mean_effect']
        values.append({'unit_id':u['unit_id'],'effect':d,'squared_effect':point['squared_effect'],
            'corrected_effect_MSE':point['noise_corrected_squared_error'],
            'excess_risk_vs_constant':point['risk_difference_vs_zero_forecast']-(constant*constant-2*constant*d),
            'shared_assignment_risk_lower':min(alternatives),'shared_assignment_risk_upper':max(alternatives),
            'route_gain_vs_reuse':d if f>0 else 0.0})
        if probability_method:
            for arm,key in [('reuse','p_reuse_correct'),('new','p_new_correct')]:
                if key not in p:raise ValueError('A probability forecast needs both action probabilities on every unit.')
                probability=float(probability_priors[arm]) if p.get('status','ok')!='ok' and probability_priors is not None else float(p[key])
                if not np.isfinite(probability) or not 0<=probability<=1:
                    raise ValueError('Action probabilities must lie in[0,1].')
                target=float(np.mean([0.0 if y is None else float(y) for y in outcomes[arm]]))
                values[-1]['Brier_'+arm]=probability**2-2*probability*target+target
                if probability_priors is not None:
                    prior=probability_priors[arm]
                    values[-1]['Brier_difference_vs_prior_'+arm]=probability**2-prior**2-2*(probability-prior)*target
    strata=sorted({u.get('stratum','all') for u in frame});rng=np.random.default_rng(seed)
    cluster_weights=None
    if weighting == 'equal_strata':
        point_weights, cluster_weights, redrawn = balanced_component_weights(frame, bootstrap, seed)
        intervals_estimable = strata_have_replication(frame)
    elif any('support_component' in u for u in frame):
        if not all('support_component' in u for u in frame):
            raise ValueError('Every unit needs a support component when using component inference.')
        components=sorted({u['support_component'] for u in frame})
        component_index={c:i for i,c in enumerate(components)}
        counts=rng.multinomial(len(components),np.full(len(components),1/len(components)),size=bootstrap)
        cluster_weights=counts[:,[component_index[u['support_component']] for u in frame]].astype(float)
    else:
        groups=[np.array([i for i,u in enumerate(frame) if u.get('stratum','all')==s]) for s in strata]
        indices=np.hstack([rng.choice(g,size=(bootstrap,len(g)),replace=True) for g in groups])
    def describe(key):
        a=np.array([u[key] for u in values])
        if weighting == 'equal_strata' and not intervals_estimable:
            return {'mean':float(point_weights @ a)}
        if cluster_weights is not None:
            draws=(cluster_weights@a)/cluster_weights.sum(axis=1)
            return {'mean':float(point_weights @ a) if weighting == 'equal_strata' else float(a.mean()),
                    'support_component_bootstrap95':np.quantile(draws,[.025,.975]).tolist()}
        return {'mean':float(a.mean()),
            'stratified_question_bootstrap95':np.quantile(a[indices].mean(axis=1),[.025,.975]).tolist()}
    return {'actor_model':actor_model,'questions':len(frame),'constant_effect_baseline':constant,
            **({'weighting':weighting,'empty_stratum_bootstrap_draws_redrawn':redrawn,
                'intervals_estimable':intervals_estimable} if weighting=='equal_strata' else {}),
            'forecast_statuses':dict(Counter(by_id[u['unit_id']].get('status','ok') for u in frame)),
            'metrics':{k:describe(k) for k in values[0] if k!='unit_id'},'units':values,
            'interpretation':'Forecasts must be fixed independently of evaluation outcomes. Unresolved rewards usezero forpoint estimates andshared case assignments forriskbounds. Negative finite-sample moment estimates remainunclipped. Saved-branch routing excludes predictor overhead anddoesnot establishdeploymentbenefit.'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--draws',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--actor-model',required=True);p.add_argument('--constant-effect',type=float)
    p.add_argument('--baseline',help='Frozen baseline JSON with constant_effect and optional probability_priors.')
    p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--seed',type=int,default=2026101511)
    p.add_argument('--weighting',choices=('questions','equal_strata'),default='questions')
    p.add_argument('--output');a=p.parse_args()
    baseline=read(a.baseline) if a.baseline else {}
    constant=a.constant_effect if a.constant_effect is not None else baseline.get('constant_effect',0.)
    priors=baseline.get('probability_priors')
    priors={'reuse':priors['THINK'],'new':priors['SEARCH']} if priors else None
    result=evaluate(rows(a.draws),rows(a.predictions),a.actor_model,constant,a.bootstrap,a.seed,priors,a.weighting)
    if a.output:write(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='units'},ensure_ascii=False,indent=2))
