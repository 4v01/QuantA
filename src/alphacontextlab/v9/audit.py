"""Replay public calibration, selection, portfolios and trusted local checkpoints."""
import argparse,hashlib,json,pickle,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.metrics import daily_ic
from ..v7.study import ic_summary,split
from ..v8.audit import read,equal,frame_equal
from ..v8.study import path_metrics
from ..v8.portfolio import make_tape
from .models import calibrate,apply_calibration,predict
from .portfolio import simulate
from .study import select_candidate,financial_gate,excess_r2


def independent_accounting(path,holdings,tape,symbols,cost):
    w=holdings.pivot(index='date',columns='symbol',values='weight').reindex(index=path.date,columns=symbols).fillna(0.).to_numpy()
    r=tape.pivot(index='date',columns='symbol',values='next_open_return').reindex(index=path.date,columns=symbols).fillna(0.).to_numpy()
    gross=np.einsum('ij,ij->i',w,r);np.testing.assert_allclose(gross,path.gross_return,atol=1e-10)
    old=np.zeros(len(symbols))
    for i,(_,day) in enumerate(path.iterrows()):
        turn=float(abs(w[i]-old).sum())
        if not day.scheduled:assert turn<1e-8
        entry=(turn if day.scheduled else 0.)*cost/10000;drift=w[i]*(1+r[i])/(1+gross[i])
        exit_turn=float(drift.sum()) if i==len(path)-1 else 0.
        equal(float(day.turnover),(turn if day.scheduled else 0.)+exit_turn)
        equal(float(day.entry_fee),entry);equal(float(day.terminal_fee),exit_turn*cost/10000)
        old=drift
    np.testing.assert_allclose((1-path.entry_fee)*(1+gross)*(1-path.terminal_fee)-1,path.net_return,atol=1e-10)
    assert (path.gross_exposure>=-1e-8).all() and (path.gross_exposure<=1+1e-7).all()


def public_phase(root,config,base):
    root=Path(root);prov=json.loads((root/'provenance.json').read_text());symbols=prov['symbols']
    assert not prov['strict_pit'] and prov['seed']==17
    for p,key in [('configs/v9.json','config_sha256'),('configs/v7.json','base_config_sha256')]:
        assert sha256(p)==prov[key]==hashlib.sha256(subprocess.check_output(['git','show',prov['source_commit']+':'+p])).hexdigest()
    all_pred=read(root/'predictions.csv.gz',['date','exit_date','exit20']);pred=all_pred[all_pred.evaluation_period]
    assert not all_pred.duplicated(['date','symbol','method']).any()
    for method in config['methods']:
        frame_equal(all_pred[all_pred.method==method],all_pred[all_pred.method=='matched_control'],['date','symbol','fold'],['label','target','label20','target20'])
    prior=read('reports/v9/development/predictions.csv.gz',['date','exit_date','exit20']) if prov['phase']=='evaluation' else all_pred.iloc[:0].copy()
    if prov['phase']=='evaluation':assert prov['development_forecasts_sha256']==sha256('reports/v9/development/predictions.csv.gz')
    for fold in sorted(all_pred.fold.unique()):
        for method in config['methods']:
            q=all_pred[(all_pred.fold==fold)&(all_pred.method==method)];evidence=root/'evidence'/(fold+'-'+method)
            selection=json.loads((evidence/'selection.json').read_text());s=selection['split'];begin=pd.Timestamp(s['test_start'])
            assert pd.Timestamp(s['training_exit20_max'])<pd.Timestamp(s['validation_start']) and pd.Timestamp(s['refit_exit20_max'])<begin
            cal=calibrate(prior,begin,method,config);equal(cal,selection['return_calibration'])
            if cal['rows']:assert pd.Timestamp(cal['exit20_max'])<begin
            np.testing.assert_allclose(apply_calibration(q,cal),q.expected_excess20,atol=1e-10)
            if method!='classic_value_quality':
                for group,state in selection['groups'].items():
                    assert state['direction']==1
                    assert state['selected']==max(state['candidates'],key=lambda z:z['score'])
                    confidence=state['selected']['diagnostics']['reliability'] if method=='matched_control' else 1.
                    if state['selected']['diagnostics']['mean'] is not None:equal(confidence,state['confidence'])
            equal(json.loads((evidence/'checkpoint.json').read_text())['provenance'],prov)
            prior=pd.concat([prior,q],ignore_index=True)
    daily=read(root/'daily_ic.csv',['date']);signals={};secondary={}
    for method,g in pred.groupby('method'):
        for horizon in [5,20]:
            parts=[]
            for fold,q in g.groupby('fold'):
                q=q.copy()
                if horizon==20:q['label']=q.label20
                d=daily_ic(q,q.prediction,base['min_cross_section']);d['fold']=fold;parts.append(d)
            d=pd.concat(parts,ignore_index=True);saved=daily[(daily.method==method)&(daily.horizon==horizon)]
            frame_equal(d,saved,['date','fold'],['n','ic','rank_ic'])
            (signals if horizon==5 else secondary)[method]=ic_summary(d,base)
    equal(signals,json.loads((root/'ic_summary.json').read_text()));equal(secondary,json.loads((root/'secondary_H20_ic.json').read_text()))
    equal(excess_r2(pred),json.loads((root/'return_metrics.json').read_text()))
    # V9 matched_control must reproduce V7 long_history, not a weaker new baseline.
    old_phase='development' if prov['phase']=='development' else 'holdout'
    old=read(Path('reports/v7')/old_phase/'predictions.csv.gz',['date'])
    old=old[(old.method=='long_history')&(old.group=='fusion')]
    frame_equal(pred[pred.method=='matched_control'],old,['date','symbol','fold'],['prediction','label'])
    tape=read(root/'market_tape.csv.gz',['date']);paths=read(root/'screen_paths.csv.gz',['date','signal_date','exit_date']);weights=read(root/'holdings.csv.gz',['date']);summary=read(root/'screen_summary.csv');count=0
    pairs=[(m,p) for m in config['methods'] for p in config['policies']]+[('equal_weight',p) for p in config['benchmarks']]
    for method,policy in pairs:
        q=pred[pred.method==('matched_control' if method=='equal_weight' else method)]
        for cost in config['cost_bps']:
            actual=paths[(paths.method==method)&(paths.policy==policy)&(paths.cost_bps==cost)]
            held=weights[(weights.method==method)&(weights.policy==policy)&(weights.cost_bps==cost)]
            generated,w=simulate(q,tape,symbols,base,config,method,policy,cost)
            frame_equal(generated,actual,['date'],['gross_return','net_return','nav','turnover','entry_fee','terminal_fee','gross_exposure','stale_held_marks'])
            frame_equal(w,held,['date','symbol'],['weight']);independent_accounting(actual,held,tape,symbols,cost)
            if method=='matched_control' and policy=='weekly_buffer':
                old_paths=read(Path('reports/v7')/old_phase/'screen_paths.csv.gz',['date'])
                old_paths=old_paths[(old_paths.method=='long_history')&(old_paths.policy=='buffer')&(old_paths.cost_bps==cost)]
                frame_equal(actual,old_paths,['date'],['gross_return','net_return','nav','turnover'])
            row=summary[(summary.method==method)&(summary.policy==policy)&(summary.cost_bps==cost)].iloc[0]
            for key,value in path_metrics(actual).items():
                if value is None:assert pd.isna(row[key])
                else:equal(value,float(row[key]))
            count+=1
    if prov['phase']=='development':equal(select_candidate(summary,signals,base,config),json.loads((root/'champion_candidate.json').read_text()))
    else:
        seal=json.loads(Path('reports/v9/champion_seal.json').read_text());assert prov['seal_sha256']==sha256('reports/v9/champion_seal.json')
        equal(financial_gate(signals,summary,seal,base,config),json.loads((root/'financial_gate.json').read_text()))
    return {'phase':prov['phase'],'forecast_rows_including_warmup':len(all_pred),'evaluation_forecast_rows':len(pred),'curves':count,'checkpoint_sets':len(list((root/'evidence').glob('*'))),'V7_control_equivalence':True}


def local_phase(root,config,base):
    from .data import load
    root=Path(root);prov=json.loads((root/'provenance.json').read_text());panel,adj,calendar,symbols,manifest,quality=load()
    assert sha256('data/v7/manifest.json')==prov['data_manifest_sha256'] and sha256('data/v4_akshare/manifest.json')==prov['price_manifest_sha256']
    equal(quality,json.loads((root/'coverage.json').read_text()))
    years=config['development_years'] if prov['phase']=='development' else config['evaluation_years'];tape=make_tape(adj,calendar,symbols,f'{max(years)}-12-31')
    frame_equal(tape,read(root/'market_tape.csv.gz',['date']),['date','symbol'],['close_return','next_open_return','open_observed'])
    pred=read(root/'predictions.csv.gz',['date']);count=0
    for name,g in pred.groupby(['fold','method']):
        fold,method=name;folder=fold+'-'+method;e=root/'evidence'/folder;p=Path('outputs/v9')/root.name/'evidence'/folder/'models.pkl'
        assert sha256(p)==json.loads((e/'checkpoint.json').read_text())['sha256']
        with p.open('rb') as f:model=pickle.load(f)
        _,_,_,q,_=split(panel,int(fold[:4]),int(fold[-1]),config['train_months'],base)
        q.loc[q.exit_date>pd.Timestamp(f'{max(years)}-12-31'),['label','target']]=np.nan
        q.loc[q.exit20>pd.Timestamp(f'{max(years)}-12-31'),['label20','target20']]=np.nan
        z=q[['date','symbol','label','target','label20','target20']].copy();z['prediction']=predict(q,model,base)['fusion']
        frame_equal(z,g,['date','symbol'],['prediction','label','target','label20','target20']);count+=1
    return {'phase':prov['phase'],'forecast_rows':len(pred),'checkpoint_sets':count,'input_tape_and_economic_features_verified':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--local-replay',action='store_true');p.add_argument('--receipt');a=p.parse_args()
    config=json.loads(Path('configs/v9.json').read_text());base=json.loads(Path(config['base_config']).read_text());results=[]
    for phase in ['development','evaluation']:
        root=Path('reports/v9')/phase
        if root.exists():results.append(local_phase(root,config,base) if a.local_replay else public_phase(root,config,base))
    sealpath=Path('reports/v9/champion_seal.json')
    if sealpath.exists():
        seal=json.loads(sealpath.read_text());assert seal['config_sha256']==sha256('configs/v9.json')
        candidate=Path('reports/v9/development/champion_candidate.json');assert seal['development_candidate_sha256']==sha256(candidate)
        cand=json.loads(candidate.read_text());assert (seal['method'],seal['policy'])==(cand['method'],cand['policy'])
        for name,digest in seal['core_sources'].items():assert sha256(name)==digest
    inventory=Path('reports/v9/old_reports_integrity.json')
    if inventory.exists():
        inv=json.loads(inventory.read_text());names=list(inv['files']);raw=subprocess.check_output(['git','cat-file','--batch'],input=''.join(inv['base_commit']+':'+p+'\n' for p in names).encode());offset=0
        for name in names:
            end=raw.index(b'\n',offset);size=int(raw[offset:end].decode().split()[2]);offset=end+1
            assert hashlib.sha256(raw[offset:offset+size]).hexdigest()==inv['files'][name]==sha256(name);offset+=size+1
    receipt={'passed':True,'mode':'local_replay' if a.local_replay else 'public','phases':results}
    if a.receipt:write_json(a.receipt,receipt)
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
