import numpy as np
import pandas as pd
from ..v7.screen import choose_target
from ..v8.portfolio import risk_state,net_edge_target


def target(scores,expected,old,state,method,policy,cost,base,config):
    good=np.isfinite(scores)
    if good.sum()<base['min_cross_section']:return np.zeros_like(old)
    if method=='equal_weight':
        w=np.zeros_like(old); w[good]=1/good.sum()
        if policy=='equal_slow_vol':
            if state is None:return np.zeros_like(old)
            vol=float(np.sqrt(w@state['covariance']@w*252));w*=min(1.,config['target_annual_volatility']/max(vol,1e-12))
        return w
    if np.std(scores[good])<1e-12:return np.zeros_like(old)
    if policy=='slow_net_edge':
        if state is None:return np.zeros_like(old)
        # V8 solver's fixed factor 5 is converted algebraically to H20.
        w,_=net_edge_target(expected,old,state['covariance']*(config['allocation_horizon']/5),1.,cost,config)
        return w
    w,missing=choose_target(scores,old,base,buffer=True)
    return np.zeros_like(old) if missing else w


def simulate(pred,tape,symbols,base,config,method,policy,cost):
    days=pd.DatetimeIndex(sorted(tape.date.unique()))
    def matrix(frame,column):return frame.pivot(index='date',columns='symbol',values=column).reindex(index=days,columns=symbols).to_numpy()
    close=matrix(tape,'close_return'); future=matrix(tape,'next_open_return'); observed=matrix(tape,'open_observed').astype(bool)
    scores=matrix(pred,'prediction'); expected=matrix(pred,'expected_excess20'); first=days.get_loc(pred.date.min())+1
    interval=5 if policy in ['weekly_buffer','equal_weekly'] else config['slow_rebalance_sessions']
    old=np.zeros(len(symbols));nav=1.;paths=[];holdings=[]
    for j,day in enumerate(range(first,len(days)-1)):
        scheduled=j%interval==0; w=old.copy(); turn=0.
        if scheduled:
            state=risk_state(close[:day],config)
            w=target(scores[day-1],expected[day-1],old,state,method,policy,cost,base,config)
            turn=float(np.abs(w-old).sum())
        returns=np.nan_to_num(future[day],nan=0.);gross=float(w@returns);entry=turn*cost/10000
        old=w*(1+returns)/(1+gross);terminal=0.
        if day==len(days)-2:
            exit_turn=float(old.sum());terminal=exit_turn*cost/10000;turn+=exit_turn;old[:]=0.
        net=(1-entry)*(1+gross)*(1-terminal)-1;nav*=1+net
        paths.append({'date':days[day],'signal_date':days[day-1],'exit_date':days[day+1],'gross_return':gross,'net_return':net,
                      'nav':nav,'turnover':turn,'entry_fee':entry,'terminal_fee':terminal,'cost_fraction':entry+terminal,
                      'gross_exposure':float(w.sum()),'scheduled':scheduled,
                      'stale_held_marks':int(((w>0)&(~observed[day]|~observed[day+1])).sum())})
        for i in np.flatnonzero(w>0):holdings.append({'date':days[day],'symbol':symbols[i],'weight':w[i]})
    return pd.DataFrame(paths),pd.DataFrame(holdings,columns=['date','symbol','weight'])
