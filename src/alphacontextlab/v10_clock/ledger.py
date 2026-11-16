"""Pooled fractional-return screen with funded virtual sleeves.

Costs retain V9's pro-rata before-return convention for a fair comparison.
They are charged once on aggregate external weight changes, not sleeve sums.
This is not a stock/share/corporate-action cash ledger.
"""
import numpy as np
import pandas as pd
from ..v7.screen import choose_target


def matrices(pred,tape,symbols):
    days=pd.DatetimeIndex(sorted(tape.date.unique()))
    def pivot(frame,column):
        return frame.pivot(index='date',columns='symbol',values=column).reindex(index=days,columns=symbols).to_numpy()
    return days,pivot(pred,'prediction'),pivot(tape,'next_open_return'),pivot(tape,'open_observed').astype(bool)


def pool_turnover(previous,target):
    changes=target-previous
    return float(np.abs(changes.sum(axis=0)).sum()),float(np.abs(changes).sum())


def simulate(pred,tape,symbols,base,config,clock,benchmark=False):
    days,scores,future,observed=matrices(pred,tape,symbols)
    stagger=clock=='stagger5';count=5 if stagger else 1;offset=0 if stagger else int(clock[-1])
    budgets=np.asarray(config['initial_sleeve_budgets'],float) if stagger else np.ones(1)
    if len(budgets)!=count or abs(budgets.sum()-1)>1e-12:raise ValueError('Unfunded or leveraged initial sleeve budget')
    positions=np.zeros((count,len(symbols)));first=days.get_loc(pred.date.min())+1
    rows=[];holdings=[];sleeve_positions=[];accounts=[]
    for j,day in enumerate(range(first,len(days)-1)):
        previous=positions.copy();before_budget=budgets.copy()
        active=j%5 if stagger else (0 if j>=offset and (j-offset)%5==0 else None)
        missing=False
        if active is not None:
            if budgets[active]<=1e-14:raise ValueError('Sleeve insolvent')
            relative=positions[active]/budgets[active]
            desired,missing=choose_target(scores[day-1],relative,base,buffer=True,benchmark=benchmark)
            positions[active]=budgets[active]*desired
        net_turn,gross_turn=pool_turnover(previous,positions)
        if net_turn>gross_turn+1e-10:raise AssertionError('Net turnover exceeds gross intentions')
        target=positions.sum(axis=0);r=np.nan_to_num(future[day],nan=0.)
        sleeve_gross=positions@r;gross=float(sleeve_gross.sum())
        if gross<=-1:raise ValueError('Portfolio insolvency')
        after_positions=positions*(1+r)/(1+gross)
        after_budget=(budgets+sleeve_gross)/(1+gross)
        if np.any(positions.sum(axis=1)>budgets+1e-10) or np.any(positions<-1e-12):raise AssertionError('Sleeve spent unowned capital')
        for i in np.flatnonzero(target>0):holdings.append({'date':days[day],'symbol':symbols[i],'weight':target[i]})
        for s in range(count):
            accounts.append({'date':days[day],'sleeve':s,'budget_before':before_budget[s],
                'cash_before':before_budget[s]-previous[s].sum(),'target_exposure':positions[s].sum(),
                'budget_after_return':after_budget[s],'cash_after_return':after_budget[s]-after_positions[s].sum(),
                'cash_after_terminal':after_budget[s] if day==len(days)-2 else after_budget[s]-after_positions[s].sum(),
                'end_stock_exposure':0. if day==len(days)-2 else after_positions[s].sum(),
                'sleeve_gross_contribution':sleeve_gross[s],'active':active==s})
            for i in np.flatnonzero(positions[s]>0):sleeve_positions.append({'date':days[day],'sleeve':s,'symbol':symbols[i],'weight':positions[s,i]})
        exit_turn=float(after_positions.sum()) if day==len(days)-2 else 0.
        rows.append({'date':days[day],'signal_date':days[day-1],'exit_date':days[day+1],'session_index':j,
            'gross_return':gross,'entry_turnover':net_turn,'gross_sleeve_turnover':gross_turn,
            'netting_reduction':max(0.,gross_turn-net_turn),'terminal_turnover':exit_turn,
            'turnover':net_turn+exit_turn,'gross_exposure':float(target.sum()),'scheduled':active is not None,
            'active_sleeve':active if active is not None else -1,'missing_prediction_rebalance':missing,
            'stale_held_marks':int(((target>0)&(~observed[day]|~observed[day+1])).sum())})
        positions=after_positions;budgets=after_budget
        if exit_turn:positions[:]=0.
    return (pd.DataFrame(rows),pd.DataFrame(holdings,columns=['date','symbol','weight']),
            pd.DataFrame(accounts),pd.DataFrame(sleeve_positions,columns=['date','sleeve','symbol','weight']))


def apply_cost(gross_path,cost):
    p=gross_path.copy();p['entry_fee']=p.entry_turnover*cost/10000;p['terminal_fee']=p.terminal_turnover*cost/10000
    p['cost_fraction']=p.entry_fee+p.terminal_fee
    p['net_return']=(1-p.entry_fee)*(1+p.gross_return)*(1-p.terminal_fee)-1
    p['nav']=(1+p.net_return).cumprod();p['cost_bps']=cost
    return p
