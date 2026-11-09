"""X-inspired candidate/value stages, mapped to return/risk/cost, not clicks."""
from itertools import combinations

import numpy as np
import pandas as pd

from ..backtest import execute,limit_fraction


def candidate_weights(n):
    pool=[np.zeros(n)]
    pool += [np.eye(n)[j] for j in range(n)]
    for a,b in combinations(range(n),2):
        value=np.zeros(n);value[[a,b]]=.5;pool.append(value)
    return np.stack(pool)


def choose(mu,downside,corr,pre,profile,assumed_cost):
    # Current position is an explicit no-churn candidate, kept first on ties.
    candidates=np.vstack([pre,candidate_weights(len(pre))])
    risk_cov=downside[:,None]*corr*downside[None,:]
    risk=np.sqrt(np.maximum(np.einsum('bi,ij,bj->b',candidates,risk_cov,candidates),0))
    utility=candidates@mu-profile['risk_penalty']*risk
    if profile['cost_aware']:
        utility-=assumed_cost*(np.abs(candidates-pre).sum(1)+candidates.sum(1))
    if not np.isfinite(utility).all():raise ValueError('Invalid candidate utility')
    winner=int(np.argmax(utility))
    return candidates[winner].copy(),float(utility[winner])


def arrays(meta,prediction,symbols):
    data=meta.copy();data['mu']=prediction[:,0];data['downside']=np.maximum(prediction[:,1],0.)
    if data.duplicated(['entry_date','symbol']).any():raise ValueError('Duplicate candidates')
    dates=pd.DatetimeIndex(sorted(data.entry_date.unique()));n=len(symbols)
    if len(data)!=len(dates)*n:raise ValueError('Incomplete basket; no intersection filtering')
    order=data.set_index(['entry_date','symbol']).reindex(pd.MultiIndex.from_product([dates,symbols]))
    def matrix(name,dtype=float):
        value=order[name].to_numpy().reshape(len(dates),n).astype(dtype)
        if dtype is float and not np.isfinite(value).all():raise ValueError('Nonfinite daily execution input '+name)
        return value
    fields={k:matrix(k) for k in ['mu','downside','target','entry_gap','exit_gap','momentum60_signal','vol20']}
    fields['tradable']=matrix('entry_tradable',bool);fields['exit_tradable']=matrix('exit_tradable',bool)
    fields['session']=matrix('session',int)[:,0]
    fields['exit_dates']=pd.DatetimeIndex(order.exit_date.to_numpy().reshape(len(dates),n)[:,0])
    if not dates[1:].equals(fields['exit_dates'][:-1]):raise ValueError('Noncontiguous market sessions')
    return dates,fields


def simulate(meta,prediction,corrs,config,profile_name,cost_bps,policy='model'):
    symbols=config['symbols'];dates,f=arrays(meta,prediction,symbols);n=len(symbols)
    profile=config['profiles'][profile_name];pre=np.zeros(n);records=[];fee=cost_bps/10000
    assumed=config['assumed_selection_cost_bps']/10000
    for j,date in enumerate(dates):
        corr=corrs[f['session'][j]];rebalance=j==0 or f['session'][j]%profile['rebalance']==0
        utility=0.;target=pre.copy()
        if rebalance:
            if policy=='model':target,utility=choose(f['mu'][j],f['downside'][j],corr,pre,profile,assumed)
            elif policy=='cash':target=np.zeros(n)
            elif policy=='csi300':
                if j==0:target=np.eye(n)[symbols.index('510300')]
            elif policy=='equal_buy_hold':
                if j==0:target=np.ones(n)/n
            elif policy in ['inverse_volatility','momentum60']:
                vol=np.maximum(f['vol20'][j],1e-5)
                if policy=='inverse_volatility':target=1/vol;target/=target.sum()
                else:
                    score=f['momentum60_signal'][j]/vol;best=np.argmax(score);target=np.zeros(n)
                    if score[best]>0:target[best]=1.
                covariance=vol[:,None]*corr*vol[None,:]
                annual_risk=np.sqrt(max(target@covariance@target,0)*252)
                target*=min(1.,config['risk_target_annual']/max(annual_risk,1e-8))
            else:raise ValueError(policy)
        limits=np.array([limit_fraction(s,date) for s in symbols])
        buy=(f['entry_gap'][j]<limits-.001)&f['tradable'][j]
        sell=(f['entry_gap'][j]>-(limits-.001))&f['tradable'][j]
        holdings,cash,traded=execute(pre,target,buy,sell,fee)
        asset_end=holdings*(1+f['target'][j]);nav_end=cash+asset_end.sum();liquidation=0.
        if j==len(dates)-1:
            exit_limits=np.array([limit_fraction(s,f['exit_dates'][j]) for s in symbols])
            if np.any((asset_end>1e-12)&((f['exit_gap'][j]<=-(exit_limits-.001))|~f['exit_tradable'][j])):
                raise ValueError('Blocked terminal liquidation; cannot manufacture fill')
            liquidation=asset_end.sum();nav_end-=fee*liquidation
        if nav_end<=0 or cash< -1e-8:raise ValueError('Invalid self-financing account')
        records.append({'entry_date':str(date.date()),'exit_date':str(f['exit_dates'][j].date()),
                        'net_return':float(nav_end-1),'market_return':float(f['target'][j,symbols.index('510300')]),
                        'turnover':float(traded+liquidation),'cost_return':float(fee*(traded+liquidation)),
                        'gross_exposure':float(holdings.sum()),'rebalanced':bool(rebalance),
                        'blocked_buys':int(np.sum((target>pre+1e-10)&~buy)),
                        'blocked_sells':int(np.sum((target<pre-1e-10)&~sell)),
                        'utility':utility})
        pre=asset_end/(cash+asset_end.sum())
    return pd.DataFrame(records)


def certainty_equivalent(path,gamma):
    r=path.net_return.to_numpy()
    return float(252*(r.mean()-.5*gamma*np.var(r,ddof=1)))
