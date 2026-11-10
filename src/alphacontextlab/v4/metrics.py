"""Signed cross-sectional IC, dependence-aware intervals and explicit stage gates."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def corr(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
    if len(a)<3 or np.std(a)<1e-14 or np.std(b)<1e-14:return np.nan
    return float(np.corrcoef(a,b)[0,1])


def daily_ic(frame,prediction,min_count=30):
    d=frame[['date','symbol','label']].copy();d['prediction']=prediction;rows=[]
    for day,g in d.groupby('date',sort=True):
        v=g.dropna(subset=['label','prediction'])
        if len(v)<min_count:continue
        rows.append({'date':day,'n':len(v),'ic':corr(v.prediction,v.label),
                     'rank_ic':corr(rankdata(v.prediction),rankdata(v.label))})
    return pd.DataFrame(rows,columns=['date','n','ic','rank_ic'])


def block_mean_interval(values,seed=17,repetitions=1000,block=20):
    x=np.asarray(values,float);x=x[np.isfinite(x)];n=len(x)
    if n<block:return [None,None]
    rng=np.random.default_rng(seed);means=[]
    for _ in range(repetitions):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        ids=((starts[:,None]+np.arange(block))%n).ravel()[:n];means.append(x[ids].mean())
    return [float(v) for v in np.quantile(means,[.025,.975])]


def ic_summary(d,config):
    x=d.rank_ic.dropna().to_numpy();spec=config['signal_gate'];sd=float(np.std(x,ddof=1)) if len(x)>1 else 0.
    return {'days':len(x),'mean_ic':float(d.ic.mean()) if len(x) else None,
            'mean_rank_ic':float(np.mean(x)) if len(x) else None,'daily_ic_ir':float(np.mean(x)/sd) if sd>0 else None,
            'positive_day_fraction':float(np.mean(x>0)) if len(x) else None,
            'mean_rank_ic_ci95':block_mean_interval(x,config['seed'],spec['bootstrap_repetitions'],spec['block_sessions']),
            'ir_definition':'mean daily RankIC / sample std; no sqrt252 inflation for overlapping H=5 labels'}


def signal_gate(d,config):
    s=ic_summary(d,config);spec=config['signal_gate'];folds=d.groupby('fold').rank_ic.mean()
    threshold=spec['mean_rank_ic'];fraction=float((folds>=threshold).mean()) if len(folds) else 0.
    reasons=[]
    if s['mean_rank_ic'] is None or s['mean_rank_ic']<threshold:reasons.append('mean_rank_ic_below_0.02')
    if fraction<spec['min_fold_pass_fraction']:reasons.append('less_than_75pct_folds_at_0.02')
    if s['mean_rank_ic_ci95'][0] is None or s['mean_rank_ic_ci95'][0]<=spec['bootstrap_lower']:reasons.append('block_mean_interval_not_strictly_positive')
    return {**s,'fold_pass_fraction':fraction,'fold_mean_rank_ic':{str(k):float(v) for k,v in folds.items()},
            'passed':not reasons,'reasons':reasons,'note':'Signed OOF RankIC after validation-only orientation; no absolute daily IC averaging'}


def path_statistics(d):
    r=d.net_return.to_numpy(float);nav=np.cumprod(1+r);sd=np.std(r,ddof=1) if len(r)>1 else 0.
    return {'sessions':len(r),'cumulative_return':float(nav[-1]-1) if len(r) else 0.,
            'annualized_return':float(nav[-1]**(252/len(r))-1) if len(r) else 0.,
            'net_sharpe':float(np.mean(r)/sd*np.sqrt(252)) if sd>1e-14 else None,
            'max_drawdown':float(np.min(np.r_[1,nav]/np.maximum.accumulate(np.r_[1,nav])-1)),
            'annual_turnover':float(d.turnover.sum()/len(r)*252) if len(r) else 0.}


def paired_ic_increment(base,other,config):
    d=base[['date','rank_ic']].merge(other[['date','rank_ic']],on='date',suffixes=('_base','_other'))
    delta=d.rank_ic_other-d.rank_ic_base
    return {'paired_days':len(d),'mean_delta_rank_ic':float(delta.mean()),
            'block_ci95':block_mean_interval(delta,config['seed'],config['signal_gate']['bootstrap_repetitions'],20)}


def paper_gate(linear_pass,model_signal,performance,annual,quality,config):
    reasons=[];spec=config['paper_gate']
    if not linear_pass:reasons.append('linear_signal_stability_gate_failed')
    if not model_signal['passed']:reasons.append('selected_model_signal_gate_failed')
    if performance['net_sharpe'] is None or performance['net_sharpe']<=spec['minimum_net_sharpe']:reasons.append('net_sharpe_not_above_1')
    if spec['all_years_net_positive'] and any(x['cumulative_return']<=0 for x in annual.values()):reasons.append('nonpositive_evaluation_year')
    if not quality['pit_universe_certified']:reasons.append('pit_universe_not_certified')
    if not quality['cash_event_ledger_certified']:reasons.append('executable_cash_event_ledger_not_certified')
    return {'status':'eligible_for_local_paper' if not reasons else 'blocked','reasons':reasons,
            'live_orders_sent':0,'paper_orders_sent':0,'historical_winner_is_not_prospectively_confirmed':True}
