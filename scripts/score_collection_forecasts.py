"""Score arbitrary three-action value forecasts using saved bounded rewards."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from search_horizon.io import rows,write


def evaluate(frame,predictions,bootstrap=10000,seed=2026102302):
    if not frame or bootstrap<1:raise ValueError('Require a nonempty frame and positive bootstrap count')
    lookup={p['task_id']:p for p in predictions}
    if len(lookup)!=len(predictions) or {r['task_id'] for r in frame}!=set(lookup):raise ValueError('Unique forecast and outcome IDs must match')
    names=sorted({r['component'] for r in frame});columns=[names.index(r['component']) for r in frame]
    if len(names)<2:raise ValueError('Component intervals require at least two independent components')
    if any(p.get('status') in ('infra_error','budget_exhausted','checkpoint_pending') for p in predictions):raise ValueError('Repair incomplete forecasts before scoring')
    w=np.random.default_rng(seed).multinomial(len(names),np.full(len(names),1/len(names)),bootstrap)[:,columns].astype(float);w/=w.sum(axis=1,keepdims=True)
    reward=np.array([r['rewards'] for r in frame],float)
    if reward.ndim!=3 or reward.shape[1]!=3 or reward.shape[2]<2 or not np.isfinite(reward).all() or np.any((reward<0)|(reward>1)):raise ValueError('Require three actions with repeated bounded rewards')
    prediction=np.array([lookup[r['task_id']]['values'] for r in frame],float)
    if prediction.shape!=(len(frame),3) or not np.isfinite(prediction).all():raise ValueError('Every forecast needs three finite values')
    kinds={lookup[r['task_id']].get('kind','expected_rewards') for r in frame}
    if len(kinds)!=1:raise ValueError('Choose one prediction kind per scoring run')
    kind=next(iter(kinds))
    if kind not in ('expected_rewards','relative_gains'):raise ValueError('Unknown prediction kind')
    if kind=='expected_rewards' and np.any((prediction<0)|(prediction>1)):raise ValueError('Expected bounded rewards must be in[0,1]')
    f=prediction[:,1:]-prediction[:,[0]]
    if np.any(np.abs(f)>1):raise ValueError('Expected action gains must lie in[-1,1]')
    mean=reward.mean(axis=2);m=reward.shape[2]
    second=(reward.sum(axis=2)**2-(reward**2).sum(axis=2))/(m*(m-1))
    theta=second[:,1:]+second[:,[0]]-2*mean[:,1:]*mean[:,[0]]
    risk=(f*f-2*f*(mean[:,1:]-mean[:,[0]])+theta).mean(axis=1)
    baseline=np.array([r['development_action_means'] for r in frame])
    b=baseline[:,1:]-baseline[:,[0]]
    base_risk=(b*b-2*b*(mean[:,1:]-mean[:,[0]])+theta).mean(axis=1)
    def stat(a):return {'mean':float(a.mean()),'bootstrap95':np.quantile(w@a,[.025,.975]).tolist()}
    result={'tasks':len(frame),'components':len(names),'kind':kind,'gain_MSE':stat(risk),
        'gain_MSE_vs_development_mean':stat(risk-base_risk),
        'outcome_squared_score':stat(((prediction[:,:,None]-reward)**2).mean(axis=(1,2))) if kind=='expected_rewards' else None,
        'scope':'Offline scoring of frozen bounded set rewards. Supporting components are the independent sampling clusters; repeated draws are not additional tasks.'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--draws',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--seed',type=int,default=2026102302);p.add_argument('--output',type=Path)
    a=p.parse_args();result=evaluate(rows(a.draws),rows(a.predictions),a.bootstrap,a.seed)
    if a.output:write(a.output,result)
    else:
        import json
        print(json.dumps(result,indent=2))
