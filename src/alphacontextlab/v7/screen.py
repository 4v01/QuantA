"""Fractional adjusted-return screen, including cash on flat scores and rank buffer."""
import numpy as np
import pandas as pd


def choose_target(scores,weights,config,buffer=False,benchmark=False):
    x=np.asarray(scores,float);eligible=np.isfinite(x);target=np.zeros(len(x))
    if eligible.sum()<config['min_cross_section']:return weights.copy(),True
    ids=np.flatnonzero(eligible)
    if benchmark:target[ids]=1/len(ids);return target,False
    if np.std(x[ids])<1e-12:return target,False
    order=ids[np.argsort(-x[ids],kind='stable')];k=max(1,int(np.ceil(len(ids)*config['top_fraction'])))
    keep=[]
    if buffer:
        limit=int(np.ceil(len(ids)*config['buffer_exit_fraction']));keep=[j for j in order[:limit] if weights[j]>1e-12][:k]
    selected=keep+[j for j in order if j not in keep][:k-len(keep)]
    target[selected]=1/len(selected);return target,False


def simulate(pred,adj,calendar,symbols,config,cost,buffer=False,benchmark=False):
    scores=pred.pivot(index='date',columns='symbol',values='prediction').reindex(index=calendar,columns=symbols)
    prices=pd.DataFrame({s:adj[s].set_index('date').open for s in symbols}).reindex(calendar)
    observed=prices.notna().to_numpy();mark=prices.ffill().to_numpy();first=calendar.get_loc(pred.date.min())+1
    weights=np.zeros(len(symbols));nav=1.;rows=[]
    for j,day in enumerate(range(first,len(calendar)-1)):
        target=weights.copy();turn=0.;missing=False;scheduled=j%config['rebalance_sessions']==0
        if scheduled:
            target,missing=choose_target(scores.iloc[day-1].to_numpy(),weights,config,buffer,benchmark);turn=float(np.abs(target-weights).sum())
        asset_return=mark[day+1]/mark[day]-1;gross=float(target@asset_return);fee=turn*cost/10000
        net=(1-fee)*(1+gross)-1;nav*=1+net;exposure=float(target.sum());weights=target*(1+asset_return)/(1+gross)
        terminal=0.
        if day==len(calendar)-2:
            t=float(weights.sum());terminal=t*cost/10000;net=(1+net)*(1-terminal)-1;nav*=1-terminal;turn+=t;weights[:]=0
        rows.append({'date':str(calendar[day].date()),'signal_date':str(calendar[day-1].date()),'exit_date':str(calendar[day+1].date()),
          'gross_return':gross,'net_return':net,'nav':nav,'turnover':turn,'cost_fraction':fee+terminal,'gross_exposure':exposure,
          'stale_held_marks':int(((target>0)&(~observed[day]|~observed[day+1])).sum()),'missing_prediction_rebalance':missing})
    return pd.DataFrame(rows)
