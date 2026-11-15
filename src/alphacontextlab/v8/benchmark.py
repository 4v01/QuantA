"""Independent equal-weight timing control, without alpha-strength gating.

The originally frozen controller incorrectly applied its flat-score cash rule
to the benchmark too. Preserve that evidence; publish this correction separately.
No model or candidate policy is changed, and the benchmark never selects stocks
by score or ticker rank.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..study import write_json
from .portfolio import risk_state
from .study import path_metrics


def simulate_benchmark(pred,tape,symbols,base,config,cost):
    days=pd.DatetimeIndex(sorted(tape.date.unique()))
    close=tape.pivot(index='date',columns='symbol',values='close_return').reindex(index=days,columns=symbols).to_numpy()
    future=tape.pivot(index='date',columns='symbol',values='next_open_return').reindex(index=days,columns=symbols).to_numpy()
    observed=tape.pivot(index='date',columns='symbol',values='open_observed').reindex(index=days,columns=symbols).to_numpy(bool)
    eligible=pred.assign(eligible=True).pivot(index='date',columns='symbol',values='eligible').reindex(index=days,columns=symbols).fillna(False).to_numpy(bool)
    first=days.get_loc(pred.date.min())+1; old=np.zeros(len(symbols)); nav=1.; paths=[]; holdings=[]
    for j,day in enumerate(range(first,len(days)-1)):
        target=old.copy(); turn=0.; scheduled=j%config['rebalance_sessions']==0
        if scheduled:
            state=risk_state(close[:day],config); good=eligible[day-1]; target=np.zeros(len(symbols))
            if state is not None and state['trend'] and good.sum()>=base['min_cross_section']:
                target[good]=1/good.sum()
                vol=float(np.sqrt(target@state['covariance']@target*252))
                target*=min(1.,config['target_annual_volatility']/max(vol,1e-12))
            turn=float(np.abs(target-old).sum())
        returns=np.nan_to_num(future[day],nan=0.); gross=float(target@returns); fee=turn*cost/10000
        net=(1-fee)*(1+gross)-1; old=target*(1+returns)/(1+gross); terminal=0.
        if day==len(days)-2:
            liquidation=float(old.sum()); terminal=liquidation*cost/10000; turn+=liquidation
            net=(1+net)*(1-terminal)-1; old[:]=0.
        nav*=1+net
        paths.append({'date':days[day],'signal_date':days[day-1],'exit_date':days[day+1],
            'gross_return':gross,'net_return':net,'nav':nav,'turnover':turn,'cost_fraction':fee+terminal,
            'entry_fee':fee,'terminal_fee':terminal,'gross_exposure':float(target.sum()),'scheduled':scheduled,
            'stale_held_marks':int(((target>0)&(~observed[day]|~observed[day+1])).sum())})
        for i in np.flatnonzero(target>0):holdings.append({'date':days[day],'symbol':symbols[i],'weight':target[i]})
    return pd.DataFrame(paths),pd.DataFrame(holdings,columns=['date','symbol','weight'])


def run():
    config=json.loads(Path('configs/v8.json').read_text()); base=json.loads(Path(config['base_config']).read_text())
    root=Path('reports/v8/benchmark_correction')
    if root.exists():raise ValueError('Preserve corrected evidence too; no overwrite')
    root.mkdir(); rows=[]
    for phase in ['development','evaluation']:
        src=Path('reports/v8')/phase
        pred=pd.read_csv(src/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
        tape=pd.read_csv(src/'market_tape.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
        symbols=json.loads((src/'provenance.json').read_text())['symbols']; paths=[]; weights=[]
        for cost in config['cost_bps']:
            p,w=simulate_benchmark(pred,tape,symbols,base,config,cost)
            rows.append({'phase':phase,'policy':'equal_trend_vol_corrected','cost_bps':cost,**path_metrics(p)})
            p['cost_bps']=cost; w['cost_bps']=cost; paths.append(p); weights.append(w)
        pd.concat(paths).to_csv(root/(phase+'_paths.csv.gz'),index=False,float_format='%.17g')
        pd.concat(weights).to_csv(root/(phase+'_holdings.csv.gz'),index=False,float_format='%.17g')
    pd.DataFrame(rows).to_csv(root/'summary.csv',index=False,float_format='%.17g')
    write_json(root/'reason.json',{'error':'Original benchmark inherited flat-alpha cash gate; use corrected benchmark for attribution.',
        'model_and_candidate_policies_changed':False,'candidate_selection_changed':False,'original_evidence_preserved':True,
        'corrected_rule':'Equal weight all common eligible names, then apply the frozen trend and volatility rules, regardless of score dispersion.'})


if __name__=='__main__':run()
