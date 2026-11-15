"""Public arithmetic/allocation replay; optional trusted local model replay."""
import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.metrics import daily_ic
from ..v6.models import percentile
from ..v7.data import load
from ..v7.study import ic_summary
from .models import calibrate_slope, nested_split
from .portfolio import make_tape, risk_state, simulate
from .study import gate, path_metrics, return_metrics, select_policy
from .benchmark import simulate_benchmark


def read(path, dates=()):
    return pd.read_csv(path,dtype={'symbol':str},parse_dates=list(dates),float_precision='round_trip')


def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a: equal(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-10,equal_nan=True)
    else: assert a==b,(a,b)


def frame_equal(a,b,keys,cols):
    x=a.sort_values(keys).reset_index(drop=True); y=b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(x[keys],y[keys],check_dtype=False)
    np.testing.assert_allclose(x[cols].to_numpy(dtype=float),y[cols].to_numpy(dtype=float),rtol=1e-8,atol=1e-10,equal_nan=True)


def public_phase(root,config,base):
    root=Path(root); prov=json.loads((root/'provenance.json').read_text()); symbols=prov['symbols']
    assert prov['strict_pit'] is False and prov['seed']==17
    assert prov['config_sha256']==sha256('configs/v8.json') and prov['base_config_sha256']==sha256('configs/v7.json')
    for name,digest in [('configs/v8.json',prov['config_sha256']),('configs/v7.json',prov['base_config_sha256'])]:
        frozen=subprocess.check_output(['git','show',prov['source_commit']+':'+name]); assert hashlib.sha256(frozen).hexdigest()==digest
    pred=read(root/'predictions.csv.gz',['date','exit_date']); cal=read(root/'calibration_predictions.csv.gz',['date','exit_date'])
    tape=read(root/'market_tape.csv.gz',['date']); paths=read(root/'screen_paths.csv.gz',['date','signal_date','exit_date'])
    weights=read(root/'holdings.csv.gz',['date']); summary=read(root/'screen_summary.csv')
    assert not pred.duplicated(['date','symbol']).any() and sorted(pred.symbol.unique())==symbols
    computed_ic=[]
    for fold,q in pred.groupby('fold'):
        evidence=root/'evidence'/fold; selection=json.loads((evidence/'selection.json').read_text()); s=selection['split']
        boundary=pd.Timestamp(s['calibration_start']); begin=pd.Timestamp(s['test_start'])
        assert pd.Timestamp(s['training_exit20_max'])<pd.Timestamp(s['validation_start'])
        assert pd.Timestamp(s['selection_exit20_max'])<boundary and pd.Timestamp(s['earlier_refit_exit20_max'])<boundary
        assert pd.Timestamp(s['calibration_exit20_max'])<begin and pd.Timestamp(s['refit_exit20_max'])<begin
        cq=cal[cal.fold==fold]; assert cq.date.min()>=boundary and cq.exit_date.max()<begin
        equal(calibrate_slope(cq,cq.prediction,config),selection['return_calibration'])
        slope=selection['return_calibration']['slope']
        np.testing.assert_allclose(q.expected_excess,slope*(q.prediction-q.groupby('date').prediction.transform('mean')),atol=1e-10)
        for group in selection['groups'].values():
            assert group['selected']['direction']==1 and 0<=group['confidence']<=1
        checkpoint=json.loads((evidence/'checkpoint.json').read_text()); equal(checkpoint['provenance'],prov)
        d=daily_ic(q,q.prediction,base['min_cross_section']); d['fold']=fold; computed_ic.append(d)
    ic=pd.concat(computed_ic,ignore_index=True)
    frame_equal(ic,read(root/'daily_ic.csv',['date']),['date','fold'],['n','ic','rank_ic'])
    equal(ic_summary(ic,base),json.loads((root/'ic_summary.json').read_text()))
    equal(return_metrics(pred),json.loads((root/'return_metrics.json').read_text()))
    days=pd.DatetimeIndex(sorted(tape.date.unique())); closes=tape.pivot(index='date',columns='symbol',values='close_return').reindex(index=days,columns=symbols).to_numpy()
    dates=set(pred.date); expected={}
    for i,day in enumerate(days):
        if day in dates:
            state=risk_state(closes[:i+1],config); expected[day]=state['market_expected'] if state else 0.
    np.testing.assert_allclose(pred.expected_return,pred.expected_excess+pred.date.map(expected),atol=1e-10)
    count=0
    for policy in config['policies']+[config['benchmark']]:
        for cost in config['cost_bps']:
            actual=paths[(paths.policy==policy)&(paths.cost_bps==cost)]
            holdings=weights[(weights.policy==policy)&(weights.cost_bps==cost)]
            regenerated,generated_weights=simulate(pred,tape,symbols,base,config,policy,cost)
            frame_equal(regenerated,actual,['date'],['gross_return','net_return','nav','turnover','cost_fraction','entry_fee','terminal_fee','gross_exposure','stale_held_marks'])
            frame_equal(generated_weights,holdings,['date','symbol'],['weight'])
            reported=summary[(summary.policy==policy)&(summary.cost_bps==cost)].iloc[0]
            for key,value in path_metrics(actual).items():
                if value is None: assert pd.isna(reported[key])
                else: equal(value,float(reported[key]))
            # Independent accounting from public daily holdings and next-open tape.
            daily=holdings.merge(tape[['date','symbol','next_open_return']],on=['date','symbol'],validate='many_to_one')
            gross=(daily.weight*daily.next_open_return).groupby(daily.date).sum().reindex(actual.date,fill_value=0.).to_numpy()
            np.testing.assert_allclose(gross,actual.gross_return,atol=1e-10)
            np.testing.assert_allclose((1-actual.entry_fee)*(1+gross)*(1-actual.terminal_fee)-1,actual.net_return,atol=1e-10)
            wide=holdings.pivot(index='date',columns='symbol',values='weight').reindex(index=actual.date,columns=symbols).fillna(0.).to_numpy()
            realized=tape.pivot(index='date',columns='symbol',values='next_open_return').reindex(index=actual.date,columns=symbols).fillna(0.).to_numpy()
            previous=np.zeros(len(symbols))
            for i,(_,day) in enumerate(actual.iterrows()):
                delta=float(np.abs(wide[i]-previous).sum())
                if not day.scheduled: assert delta<1e-8
                entry=delta*cost/10000 if day.scheduled else 0.
                drift=wide[i]*(1+realized[i])/(1+gross[i])
                exit_turn=float(drift.sum()) if i==len(actual)-1 else 0.
                equal(float(day.turnover),(delta if day.scheduled else 0.)+exit_turn)
                equal(float(day.entry_fee),entry); equal(float(day.terminal_fee),exit_turn*cost/10000)
                previous=drift
            assert (actual.gross_exposure>=-1e-8).all() and (actual.gross_exposure<=1+1e-7).all()
            if policy in ['net_edge','trend_net_edge']:
                # Weight cap applies when allocating; subsequent price drift is allowed.
                rebalance=set(actual.loc[actual.scheduled,'date'])
                assert holdings.loc[holdings.date.isin(rebalance),'weight'].max()<=config['maximum_stock_weight']+1e-7 if len(holdings) else True
            count+=1
    if prov['phase']=='development': equal(select_policy(summary,config),json.loads((root/'champion_candidate.json').read_text()))
    else:
        seal=json.loads(Path('reports/v8/champion_seal.json').read_text())
        assert prov['seal_sha256']==sha256('reports/v8/champion_seal.json')
        equal(gate(ic_summary(ic,base),summary,seal['policy'],base,config,seal['selection_qualified']),json.loads((root/'financial_gate.json').read_text()))
    return {'phase':prov['phase'],'predictions':len(pred),'curves':count,'checkpoint_sets':len(list((root/'evidence').glob('*')))}


def replay_phase(root,config,base):
    root=Path(root); prov=json.loads((root/'provenance.json').read_text()); out=Path('outputs/v8')/root.name
    panel,adj,calendar,symbols,_,_=load(); years=config['development_years'] if root.name=='development' else config['evaluation_years']
    assert sha256('data/v7/manifest.json')==prov['data_manifest_sha256'] and sha256('data/v4_akshare/manifest.json')==prov['price_manifest_sha256']
    tape=make_tape(adj,calendar,symbols,f'{max(years)}-12-31')
    frame_equal(tape,read(root/'market_tape.csv.gz',['date']),['date','symbol'],['close_return','next_open_return','open_observed'])
    pred=read(root/'predictions.csv.gz',['date','exit_date']); cal=read(root/'calibration_predictions.csv.gz',['date','exit_date'])
    for year in years:
        for half in [1,2]:
            fold=f'{year}H{half}'; evidence=root/'evidence'/fold; p=out/'evidence'/fold/'models.pkl'
            assert sha256(p)==json.loads((evidence/'checkpoint.json').read_text())['sha256']
            with p.open('rb') as f: models=pickle.load(f)
            _,_,_,cq,_,q,_=nested_split(panel,year,half,base,config)
            for phase,frame,public in [('calibration',cq,cal[cal.fold==fold]),('final',q,pred[pred.fold==fold])]:
                prediction=np.zeros(len(frame))
                for group,state in models[phase].items():
                    raw=state['model'].predict(frame[state['features']].to_numpy())*state['direction']
                    rank=percentile(frame,raw); c=state['calibration']
                    prediction+=base['weights'][group]*np.clip((rank-c['mean'])/c['std'],-base['score_clip'],base['score_clip'])*state['confidence']
                z=frame[['date','symbol']].copy(); z['prediction']=prediction
                frame_equal(z,public,['date','symbol'],['prediction'])
    return {'phase':prov['phase'],'local_predictions':len(pred)+len(cal),'checkpoint_sets':len(years)*2}


def corrected_benchmark(config,base):
    root=Path('reports/v8/benchmark_correction'); summary=read(root/'summary.csv'); count=0
    for phase in ['development','evaluation']:
        original=Path('reports/v8')/phase
        pred=read(original/'predictions.csv.gz',['date']); tape=read(original/'market_tape.csv.gz',['date'])
        symbols=json.loads((original/'provenance.json').read_text())['symbols']
        paths=read(root/(phase+'_paths.csv.gz'),['date']); weights=read(root/(phase+'_holdings.csv.gz'),['date'])
        for cost in config['cost_bps']:
            regenerated,w=simulate_benchmark(pred,tape,symbols,base,config,cost)
            actual=paths[paths.cost_bps==cost]; held=weights[weights.cost_bps==cost]
            frame_equal(regenerated,actual,['date'],['gross_return','net_return','nav','turnover','entry_fee','terminal_fee','gross_exposure'])
            frame_equal(w,held,['date','symbol'],['weight'])
            row=summary[(summary.phase==phase)&(summary.cost_bps==cost)].iloc[0]
            for key,value in path_metrics(actual).items():
                if value is None: assert pd.isna(row[key])
                else: equal(value,float(row[key]))
            count+=1
    return count


def main():
    p=argparse.ArgumentParser(); p.add_argument('--local-replay',action='store_true'); p.add_argument('--receipt'); a=p.parse_args()
    config=json.loads(Path('configs/v8.json').read_text()); base=json.loads(Path(config['base_config']).read_text()); result=[]
    for phase in ['development','evaluation']:
        root=Path('reports/v8')/phase
        if not root.exists(): continue
        result.append(replay_phase(root,config,base) if a.local_replay else public_phase(root,config,base))
    if Path('reports/v8/champion_seal.json').exists():
        seal=json.loads(Path('reports/v8/champion_seal.json').read_text()); candidate=Path('reports/v8/development/champion_candidate.json')
        assert seal['development_candidate_sha256']==sha256(candidate)
        assert seal['development_summary_sha256']==sha256('reports/v8/development/screen_summary.csv')
        assert seal['development_signal_sha256']==sha256('reports/v8/development/ic_summary.json')
        assert seal['config_sha256']==sha256('configs/v8.json') and seal['base_config_sha256']==sha256('configs/v7.json')
        for path,digest in seal['core_sources'].items(): assert sha256(path)==digest
        equal(seal['policy'],json.loads(candidate.read_text())['policy'])
        inventory=json.loads(Path('reports/v8/old_reports_integrity.json').read_text())
        paths=list(inventory['files']); base_commit=inventory['base_commit']
        raw=subprocess.check_output(['git','cat-file','--batch'],input=''.join(base_commit+':'+path+'\n' for path in paths).encode())
        offset=0
        for path in paths:
            end=raw.index(b'\n',offset); header=raw[offset:end].decode().split(); size=int(header[2]); offset=end+1
            frozen=raw[offset:offset+size]; offset+=size+1
            assert hashlib.sha256(frozen).hexdigest()==inventory['files'][path]==sha256(path)
    receipt={'passed':True,'mode':'local_replay' if a.local_replay else 'public','phases':result}
    if not a.local_replay and Path('reports/v8/benchmark_correction').exists():
        receipt['corrected_benchmark_curves']=corrected_benchmark(config,base)
    if a.receipt: write_json(a.receipt,receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
