"""Published arithmetic, selection, seal and optional trusted-model replay."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..v4.metrics import daily_ic,path_statistics
from ..v6.audit import close
from .study import ic_summary,split
from ..v6.models import percentile


def audit_case(root,config):
    r=Path(root);prov=json.loads((r/'provenance.json').read_text())
    assert sha256('configs/v7.json')==prov['config_sha256']
    frozen=subprocess.check_output(['git','show',prov['source_commit']+':configs/v7.json']);assert hashlib.sha256(frozen).hexdigest()==prov['config_sha256']
    assert sha256(r/'data_manifest.json')==prov['data_manifest_sha256']
    manifest=json.loads((r/'data_manifest.json').read_text());assert manifest['summary_field_used'] is False and manifest['new_source']['summary_field_forbidden'] is True
    assert prov['strict_pit'] is False and manifest['config_sha256']==prov['config_sha256']
    pred=pd.read_csv(r/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
    assert not pred.duplicated(['method','group','date','symbol']).any()
    wide=pred.pivot(index=['method','date','symbol'],columns='group',values='prediction');assert np.isfinite(wide).all().all()
    np.testing.assert_allclose(wide.fusion,sum(config['weights'][g]*wide[g] for g in config['weights']),atol=1e-11,rtol=1e-11)
    # Identical evaluation rows, regardless of method's training window or confidence.
    common=None
    for m in config['methods']:
        index=pred[(pred.method==m)&(pred.group=='fusion')][['date','symbol']].sort_values(['date','symbol']).reset_index(drop=True)
        if common is None:common=index
        else:assert common.equals(index)
    published=pd.read_csv(r/'daily_ic.csv',parse_dates=['date'],float_precision='round_trip');computed=[]
    for (m,g,fold),x in pred.groupby(['method','group','fold']):
        z=daily_ic(x,x.prediction,config['min_cross_section']);z['method']=m;z['group']=g;z['fold']=fold;computed.append(z)
    a=pd.concat(computed).sort_values(['method','group','fold','date']).reset_index(drop=True);b=published.sort_values(['method','group','fold','date']).reset_index(drop=True)
    assert a[['method','group','fold','date']].equals(b[['method','group','fold','date']])
    np.testing.assert_allclose(a[['n','ic','rank_ic']],b[['n','ic','rank_ic']],atol=1e-9,rtol=1e-8,equal_nan=True)
    summaries={m:ic_summary(g[g.group=='fusion'],config) for m,g in published.groupby('method')};close(summaries,json.loads((r/'ic_summary.json').read_text()))
    for p in sorted((r/'evidence').glob('*/selection.json')):
        fold,method=p.parent.name.split('-',1);rec=json.loads(p.read_text());assert rec['provenance']==prov;info=rec['split']
        assert pd.Timestamp(info['training_exit20_max'])<pd.Timestamp(info['validation_start']) and pd.Timestamp(info['refit_exit20_max'])<pd.Timestamp(info['test_start'])
        for group,state in rec['groups'].items():
            chosen=max(state['candidates'],key=lambda z:z['selection_score']);close(chosen,state['selected'])
            for candidate in state['candidates']:
                diag=candidate['validation'];mean=diag['mean']
                if mean is not None:
                    blocks=np.array(diag['blocks']);disp=float(np.std(blocks,ddof=1));stable=mean-config['instability_penalty']*disp
                    rel=max(mean,0)/(max(mean,0)+disp+config['reliability_floor'])*float(np.mean(blocks>0))
                    assert np.isclose(diag['stable_score'],stable,atol=1e-10) and np.isclose(diag['reliability'],rel,atol=1e-10)
                    expected=abs(mean) if method=='legacy_matched' else stable
                    assert np.isclose(candidate['selection_score'],expected,atol=1e-10)
                    assert candidate['direction']==(1 if method!='legacy_matched' or mean>=0 else -1)
            assert 0<=state['confidence']<=1
            if method not in ['legacy_matched','signed_stable']:
                assert np.isclose(state['confidence'],chosen['validation']['reliability']) or state.get('constant_validation',False)
    paths=pd.read_csv(r/'screen_paths.csv.gz');table=pd.read_csv(r/'screen_summary.csv');curves=0
    for (m,policy,cost),d in paths.groupby(['method','policy','cost_bps']):
        assert d.date.is_monotonic_increasing and not d.date.duplicated().any()
        assert np.all(pd.to_datetime(d.signal_date)<pd.to_datetime(d.date)) and np.all(pd.to_datetime(d.exit_date)>pd.to_datetime(d.date))
        np.testing.assert_allclose(d.nav,np.cumprod(1+d.net_return),atol=1e-9,rtol=1e-8)
        stat=path_statistics(d);row=table[(table.method==m)&(table.policy==policy)&(table.cost_bps==cost)].iloc[0]
        for k,v in stat.items():
            if v is None:assert pd.isna(row[k])
            else:assert np.isclose(v,row[k],atol=1e-9,rtol=1e-8)
        curves+=1
    years=config['development_years'] if prov['phase']=='development' else config['holdout_years'];assert set(pred.date.dt.year)==set(years)
    if prov['phase']=='development':
        eligible={m:x['coverage_adjusted_ic'] if x['ic_coverage']>=config['minimum_ic_coverage'] else -1. for m,x in summaries.items()}
        champion=max(eligible,key=eligible.get);candidate=json.loads((r/'champion_candidate.json').read_text());assert champion==candidate['method'];close(eligible,candidate['scores'])
    else:
        seal=json.loads(Path('reports/v7/champion_seal.json').read_text());assert sha256('reports/v7/champion_seal.json')==prov['champion_seal_sha256']
        frozen_seal=subprocess.check_output(['git','show',prov['source_commit']+':reports/v7/champion_seal.json']);assert hashlib.sha256(frozen_seal).hexdigest()==prov['champion_seal_sha256']
        gate=json.loads((r/'financial_gate.json').read_text());m=seal['method'];x=summaries[m];row=table[(table.method==m)&(table.policy=='buffer')&(table.cost_bps==20)].iloc[0];reasons=[]
        if x['ic_coverage']<config['minimum_ic_coverage']:reasons.append('insufficient_IC_coverage')
        if x['rank_ic'] is None or x['rank_ic']<config['minimum_ic']:reasons.append('IC_below_0.02')
        if x['passing_fold_fraction']<config['minimum_passing_fold_fraction']:reasons.append('IC_not_stable_in_75pct_folds')
        if x['ci95'][0] is None or x['ci95'][0]<=0:reasons.append('IC_interval_crosses_zero')
        if not np.isfinite(row.net_sharpe) or row.net_sharpe<=config['minimum_net_sharpe']:reasons.append('net_sharpe_not_above_1')
        assert gate['champion']==m and gate['passed']==(not reasons) and gate['reasons']==reasons and not gate['paper_admitted']
    return {'status':'passed','phase':prov['phase'],'prediction_rows':len(pred),'daily_ic_rows':len(a),'curves':curves,'source_commit':prov['source_commit']}


def replay_case(root,config):
    import pickle
    from .data import load
    from .screen import simulate
    r=Path(root);out=Path('outputs/v7')/r.name;prov=json.loads((r/'provenance.json').read_text())
    assert sha256('data/v7/manifest.json')==prov['data_manifest_sha256'] and sha256(Path(config['data_prices'])/'manifest.json')==prov['price_manifest_sha256']
    panel,adj,calendar,symbols,m,quality=load();public=pd.read_csv(r/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip');generated=[];sets=0
    years=config['development_years'] if prov['phase']=='development' else config['holdout_years']
    for year in years:
        for half in [1,2]:
            fold=f'{year}H{half}'
            for method in config['methods']:
                name=fold+'-'+method;receipt=json.loads((r/'evidence'/name/'checkpoint.json').read_text());fp=out/fold/method/'models.pkl'
                assert sha256(fp)==receipt['sha256'] and receipt['provenance']==prov
                with fp.open('rb') as f:models=pickle.load(f)
                months=config['short_train_months'] if method in ['legacy_matched','signed_stable','reliable'] else config['long_train_months']
                _,_,_,q,_=split(panel,year,half,months,config);q.loc[q.exit_date>pd.Timestamp(f'{max(years)}-12-31'),['label','target']]=np.nan
                scores={}
                for group,state in models.items():
                    raw=state['model'].predict(q[state['features']].to_numpy())*state['direction'];rank=percentile(q,raw);c=state['calibration']
                    scores[group]=np.clip((rank-c['mean'])/c['std'],-config['score_clip'],config['score_clip'])*state['confidence']
                scores['fusion']=sum(config['weights'][g]*scores[g] for g in config['weights'])
                for group in scores:
                    expected=public[(public.method==method)&(public.group==group)&(public.fold==fold)].sort_values(['date','symbol'])
                    z=q[['date','symbol','label','target']].copy();z['prediction']=scores[group];z['method']=method;z['group']=group;z['fold']=fold;z=z.sort_values(['date','symbol'])
                    assert expected[['date','symbol']].reset_index(drop=True).equals(z[['date','symbol']].reset_index(drop=True))
                    np.testing.assert_allclose(expected[['label','target','prediction']],z[['label','target','prediction']],atol=1e-10,rtol=1e-10,equal_nan=True);generated.append(z)
                sets+=1
    generated=pd.concat(generated,ignore_index=True);published=pd.read_csv(r/'screen_paths.csv.gz');cal=calendar[calendar<=f'{max(years)}-12-31'];curves=0
    for method in config['methods']+['equal_weight']:
        pred=generated[(generated.method==('legacy_matched' if method=='equal_weight' else method))&(generated.group=='fusion')]
        for policy in ['weekly','buffer']:
            if method=='equal_weight' and policy=='buffer':continue
            for cost in config['screen_cost_bps']:
                got=simulate(pred,adj,cal,symbols,config,cost,policy=='buffer',method=='equal_weight');expected=published[(published.method==method)&(published.policy==policy)&(published.cost_bps==cost)]
                assert len(got)==len(expected)
                np.testing.assert_allclose(got[['net_return','gross_return','nav','turnover','gross_exposure']],expected[['net_return','gross_return','nav','turnover','gross_exposure']],atol=1e-9,rtol=1e-8);curves+=1
    return {'status':'passed','phase':prov['phase'],'checkpoint_sets':sets,'prediction_rows_replayed':len(generated),'curves_replayed':curves}


def run(root='reports/v7',local=False):
    r=Path(root);config=json.loads(Path('configs/v7.json').read_text());results=[];replays=[]
    for name in ['development','holdout']:
        if not (r/name/'provenance.json').exists():continue
        results.append(audit_case(r/name,config))
        if local:replays.append(replay_case(r/name,config))
    seal=json.loads((r/'champion_seal.json').read_text());assert seal['method']==json.loads((r/'development/champion_candidate.json').read_text())['method']
    assert sha256(r/'development/ic_summary.json')==seal['development_ic_summary_sha256'] and seal['no_holdout_based_replacement'] is True
    old=r/'old_reports_integrity.json'
    if old.exists():
        obj=json.loads(old.read_text())
        for name,digest in obj['sha256'].items():assert sha256(name)==digest and hashlib.sha256(subprocess.check_output(['git','show',obj['reference_commit']+':'+name])).hexdigest()==digest
        assert len(obj['sha256'])==obj['unchanged_files']
    attribution=r/'cost_attribution.json'
    if attribution.exists():
        paths=pd.read_csv(r/'holdout/screen_paths.csv.gz')
        for row in json.loads(attribution.read_text()):
            d=paths[(paths.method==row['method'])&(paths.policy==row['policy'])&(paths.cost_bps==20)]
            close(path_statistics(d),row['net'])
            gross=d.copy();gross['net_return']=gross.gross_return
            close(path_statistics(gross),row['same_holdings_zero_cost_attribution'])
    paired=r/'paired_ic_comparisons.json'
    if paired.exists():
        from ..v4.metrics import paired_ic_increment
        ic=pd.read_csv(r/'holdout/daily_ic.csv',parse_dates=['date']);ic=ic[ic.group=='fusion']
        expected={m:paired_ic_increment(ic[ic.method=='legacy_matched'],ic[ic.method==m],{'seed':config['seed'],'signal_gate':{'bootstrap_repetitions':config['bootstrap_repetitions']}}) for m in config['methods'] if m!='legacy_matched'}
        close(expected,json.loads(paired.read_text()))
    result={'status':'passed','cases':results,'local_replays':replays,'sealed_champion':seal['method'],'strict_pit':False,'paper_admitted':False};print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--local-replay',action='store_true');a=p.parse_args();run(local=a.local_replay)
if __name__=='__main__':main()
