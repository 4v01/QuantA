"""Recompute all public fusion diagnostics; optional trusted local checkpoint replay."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..v4.metrics import daily_ic,path_statistics,paired_ic_increment
from .models import GROUPS,percentile
from .study import summary,splits


def close(a,b,path='root'):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Keys differ '+path)
        for k in a:close(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('Lengths differ '+path)
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not np.isclose(a,b,atol=1e-9,rtol=1e-8):raise ValueError(f'Number differs {path}: {a} != {b}')
    elif a!=b:raise ValueError('Mismatch '+path)


def audit_case(root,config):
    r=Path(root);prov=json.loads((r/'provenance.json').read_text())
    assert prov['strict_pit'] is False and prov['availability_mode']==config['availability_mode']
    assert sha256('configs/v6.json')==prov['config_sha256']
    frozen=subprocess.check_output(['git','show',prov['source_commit']+':configs/v6.json'])
    assert hashlib.sha256(frozen).hexdigest()==prov['config_sha256']
    assert sha256(r/'data_manifest.json')==prov['data_manifest_sha256']
    if prov['phase']=='followup':
        assert sha256('configs/v6_improvement.json')==prov['improvement_sha256']
        frozen=subprocess.check_output(['git','show',prov['source_commit']+':configs/v6_improvement.json'])
        assert hashlib.sha256(frozen).hexdigest()==prov['improvement_sha256']
    pred=pd.read_csv(r/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
    assert not pred.duplicated(['date','symbol','model']).any()
    # No source group is silently dropped or reweighted.
    wide=pred.pivot(index=['date','symbol'],columns='model',values='prediction')
    assert wide.notna().all().all();expected=sum(config['weights'][g]*wide[g] for g in config['weights'])
    np.testing.assert_allclose(wide.fusion,expected,atol=1e-12,rtol=1e-12)
    d=pd.read_csv(r/'daily_ic.csv',parse_dates=['date'],float_precision='round_trip');actual=[]
    for (model,fold),g in pred.groupby(['model','fold']):
        z=daily_ic(g,g.prediction,config['min_cross_section']);z['model']=model;z['fold']=fold;actual.append(z)
    a=pd.concat(actual).sort_values(['model','fold','date']).reset_index(drop=True)
    b=d.sort_values(['model','fold','date']).reset_index(drop=True)
    assert a[['date','model','fold']].equals(b[['date','model','fold']]);np.testing.assert_allclose(a[['n','ic','rank_ic']],b[['n','ic','rank_ic']],atol=1e-9,rtol=1e-8)
    measures={arm:summary(z,config) for arm,z in d.groupby('model')};close(measures,json.loads((r/'ic_summary.json').read_text()))
    increments={arm:paired_ic_increment(d[d.model=='price_volume'],d[d.model==arm],{'seed':config['seed'],'signal_gate':{'bootstrap_repetitions':config['bootstrap_repetitions']}}) for arm in ['fundamental','alternative','fusion']}
    close(increments,json.loads((r/'incremental_ic.json').read_text()))
    val=pd.read_csv(r/'validation_ic.csv',parse_dates=['date'])
    for p in sorted((r/'evidence').glob('*-selection.json')):
        name=p.name.replace('-selection.json','');record=json.loads(p.read_text());s=record['split']
        assert pd.Timestamp(s['training_label_exit_max'])<pd.Timestamp(s['validation_start'])
        assert pd.Timestamp(s['refit_label_exit_max'])<pd.Timestamp(s['test_start'])
        assert record['provenance']==prov and record['selection']['weights']==config['weights']
        for group,e in record['selection']['groups'].items():
            chosen=max(e['candidates'],key=lambda x:x['selection_score']);close(chosen,e['selected'])
            assert chosen['direction']==(1 if chosen['raw_validation_ic']>=0 else -1)
            assert np.isclose(val[(val.model==group)&(val.fold==name)].rank_ic.mean(),chosen['selection_score'],atol=1e-9)
    paths=pd.read_csv(r/'screen_paths.csv.gz');stats=pd.read_csv(r/'screen_summary.csv');curves=0
    for (arm,cost),g in paths.groupby(['model','cost_bps']):
        assert g.date.is_monotonic_increasing and not g.date.duplicated().any()
        assert np.all(pd.to_datetime(g.signal_date)<pd.to_datetime(g.date))
        assert np.all(pd.to_datetime(g.exit_date)>pd.to_datetime(g.date))
        np.testing.assert_allclose(g.nav,np.cumprod(1+g.net_return),atol=1e-9,rtol=1e-8)
        v=path_statistics(g);row=stats[(stats.model==arm)&(stats.cost_bps==cost)].iloc[0];close(v,{k:row[k].item() if hasattr(row[k],'item') else row[k] for k in v});curves+=1
    if prov['phase']=='pilot':assert pred.date.dt.year.eq(2020).all()
    else:assert pred.date.dt.year.eq(2021).all()
    status=json.loads((r/'status.json').read_text());assert not status['strict_pit'] and not status['paper_admitted'] and status['orders_sent']==0
    return {'status':'passed','case':r.name,'prediction_rows':len(pred),'daily_ic_rows':len(d),'curves':curves,'source_commit':prov['source_commit']}


def replay_case(root,outputs,config):
    import pickle
    from .data import load
    from ..v4.screen import screen_path
    r=Path(root);prov=json.loads((r/'provenance.json').read_text());out=Path(outputs)
    assert sha256('data/v6/manifest.json')==prov['data_manifest_sha256']
    assert sha256(Path(config['data_prices'])/'manifest.json')==prov['price_manifest_sha256']
    variant=prov.get('variant','baseline');p,raw,adj,calendar,symbols,manifest,prices,coverage=load(entity_only=prov['entity_only'],include_text=variant=='entity_tfidf')
    public=pd.read_csv(r/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip');generated=[];checkpoints=0
    years=config['pilot_years'] if prov['phase']=='pilot' else config['followup_years']
    for year in years:
        for half in [1,2]:
            fold=f'{year}H{half}';receipt=json.loads((r/'evidence'/(fold+'-checkpoint.json')).read_text());fp=out/fold/'models.pkl'
            assert sha256(fp)==receipt['sha256'] and receipt['provenance']==prov
            with fp.open('rb') as f:models=pickle.load(f)
            _,_,_,q,_=splits(p,year,half,config);q.loc[q.exit_date>pd.Timestamp(f'{max(years)}-12-31'),['label','target']]=np.nan
            scores={}
            for arm,m in models.items():
                x=q[m['features']] if m.get('text_input',False) else q[m['features']].to_numpy()
                values=percentile(q,m['model'].predict(x)*m['direction']);c=m['calibration'];scores[arm]=np.clip((values-c['mean'])/c['std'],-config['score_clip'],config['score_clip'])
            scores['fusion']=sum(config['weights'][g]*scores[g] for g in config['weights'])
            for arm in scores:
                expected=public[(public.model==arm)&(public.fold==fold)].sort_values(['date','symbol'])
                z=q[['date','symbol','label','target']].copy();z['prediction']=scores[arm];z['model']=arm;z['fold']=fold;z=z.sort_values(['date','symbol'])
                assert expected[['date','symbol']].reset_index(drop=True).equals(z[['date','symbol']].reset_index(drop=True))
                np.testing.assert_allclose(expected[['label','target','prediction']],z[['label','target','prediction']],equal_nan=True,atol=1e-10,rtol=1e-10);generated.append(z)
            checkpoints+=1
    generated=pd.concat(generated,ignore_index=True);paths=pd.read_csv(r/'screen_paths.csv.gz');cal=calendar[calendar<=f'{max(years)}-12-31'];local=dict(config);local['symbols']=symbols;curves=0
    for arm in list(GROUPS)+['fusion','equal_weight']:
        for cost in config['screen_cost_bps']:
            d=generated[generated.model==('price_volume' if arm=='equal_weight' else arm)];path=screen_path(d,adj,cal,symbols,local,cost,arm=='equal_weight')
            expected=paths[(paths.model==arm)&(paths.cost_bps==cost)]
            assert len(path)==len(expected)
            np.testing.assert_allclose(path[['gross_return','net_return','nav','turnover']],expected[['gross_return','net_return','nav','turnover']],atol=1e-9,rtol=1e-8);curves+=1
    return {'status':'passed','case':r.name,'checkpoint_sets':checkpoints,'predictions_replayed':len(generated),'curves_replayed':curves}


def run(root='reports/v6',local=False):
    root=Path(root);config=json.loads(Path('configs/v6.json').read_text());cases=[p for p in sorted(root.iterdir()) if p.is_dir() and (p/'provenance.json').exists()]
    results=[];replays=[]
    for p in cases:
        results.append(audit_case(p,config))
        if local:replays.append(replay_case(p,Path('outputs/v6')/p.name,config))
    if len(cases)>1:
        # A matched component comparison: price and financial signals are identical.
        base=pd.read_csv(root/'followup_baseline/predictions.csv.gz',dtype={'symbol':str})
        for name in ['entity_filter','entity_tfidf']:
            q=pd.read_csv(root/('followup_'+name)/'predictions.csv.gz',dtype={'symbol':str})
            for group in ['price_volume','fundamental']:
                a=base[base.model==group].sort_values(['date','symbol']);b=q[q.model==group].sort_values(['date','symbol'])
                assert a[['date','symbol']].reset_index(drop=True).equals(b[['date','symbol']].reset_index(drop=True))
                np.testing.assert_allclose(a.prediction,b.prediction,atol=1e-12,rtol=1e-12)
    paired=root/'followup_paired_comparisons.json'
    if paired.exists():
        frames={v:pd.read_csv(root/('followup_'+v)/'daily_ic.csv',parse_dates=['date']) for v in ['baseline','entity_filter','entity_tfidf']}
        computed={v:{g:paired_ic_increment(frames['baseline'][frames['baseline'].model==g],frames[v][frames[v].model==g],
            {'seed':config['seed'],'signal_gate':{'bootstrap_repetitions':config['bootstrap_repetitions']}}) for g in ['alternative','fusion']} for v in ['entity_filter','entity_tfidf']}
        close(computed,json.loads(paired.read_text()))
    old=root/'old_reports_integrity.json'
    if old.exists():
        record=json.loads(old.read_text())
        for name,digest in record['sha256'].items():
            assert sha256(name)==digest
            content=subprocess.check_output(['git','show',record['reference_commit']+':'+name])
            assert hashlib.sha256(content).hexdigest()==digest
        assert len(record['sha256'])==record['unchanged_files']
    result={'status':'passed','cases':results,'local_replays':replays,'strict_pit':False,'paper_admitted':False};print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports/v6');p.add_argument('--local-replay',action='store_true');a=p.parse_args();run(a.reports,a.local_replay)
if __name__=='__main__':main()
