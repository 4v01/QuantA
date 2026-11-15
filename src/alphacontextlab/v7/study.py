import argparse,json,pickle,subprocess
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.metrics import daily_ic,block_mean_interval,path_statistics
from .data import load,prepare
from .models import fit_method
from .screen import simulate


def split(panel,year,half,months,c):
    begin=pd.Timestamp(year,1 if half==1 else 7,1);end=begin+pd.DateOffset(months=6);val=begin-pd.DateOffset(months=c['validation_months']);start=max(pd.Timestamp(c['start']),val-pd.DateOffset(months=months))
    good=panel.eligible&panel.target.notna()&panel.target20.notna();count=panel[good].groupby('date').symbol.count();good &= panel.date.isin(count[count>=c['min_cross_section']].index)
    a=panel[good&(panel.date>=start)&(panel.date<val)&(panel.exit20<val)].copy()
    b=panel[good&(panel.date>=val)&(panel.date<begin)&(panel.exit20<begin)].copy()
    r=panel[good&(panel.date>=start)&(panel.date<begin)&(panel.exit20<begin)].copy()
    q=panel[panel.eligible&(panel.date>=begin)&(panel.date<end)].copy();counts=q.groupby('date').symbol.count();q=q[q.date.isin(counts[counts>=c['min_cross_section']].index)].copy()
    if min(len(a),len(b),len(q))<500:raise ValueError('Insufficient fold')
    assert a.exit20.max()<val and r.exit20.max()<begin
    info={'train_start':str(start),'validation_start':str(val),'test_start':str(begin),'test_end_exclusive':str(end),
      'training_exit20_max':str(a.exit20.max()),'refit_exit20_max':str(r.exit20.max()),'training_rows':len(a),'validation_rows':len(b),'refit_rows':len(r),'test_rows':len(q)}
    return a,b,r,q,info


def ic_summary(d,c):
    x=d.rank_ic.dropna().to_numpy();n=len(d);std=float(np.std(x,ddof=1)) if len(x)>1 else 0.
    means=d.groupby('fold').rank_ic.mean();fraction=float((means>=c['minimum_ic']).mean())
    return {'days':n,'valid_ic_days':len(x),'ic_coverage':len(x)/n if n else 0.,'rank_ic':float(x.mean()) if len(x) else None,
      'coverage_adjusted_ic':float(x.sum()/n) if n else None,'ic_ir':float(x.mean()/std) if std>0 else None,
      'ci95':block_mean_interval(x,c['seed'],c['bootstrap_repetitions'],c['bootstrap_block']),
      'passing_fold_fraction':fraction,'note':'Flat scores have undefined IC; valid coverage and zero-credit selection statistic are separate.'}


def run(phase,config_path='configs/v7.json',root='data/v7',out='outputs/v7/development'):
    c=json.loads(Path(config_path).read_text());out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Freeze clean source/config before fitting')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    seal=None
    if phase=='holdout':
        seal=json.loads(Path('reports/v7/champion_seal.json').read_text())
        if seal['config_sha256']!=sha256(config_path):raise ValueError('Sealed config changed')
    panel,adj,calendar,symbols,manifest,coverage=load(config_path,root);out.mkdir(parents=True)
    prov={'source_commit':source,'config_sha256':sha256(config_path),'data_manifest_sha256':sha256(Path(root)/'manifest.json'),
      'price_manifest_sha256':sha256(Path(c['data_prices'])/'manifest.json'),'phase':phase,'seed':c['seed'],'strict_pit':False,
      'versions':{x:version(x) for x in ['numpy','pandas','scipy','scikit-learn','lightgbm']},'champion_seal_sha256':sha256('reports/v7/champion_seal.json') if seal else None}
    write_json(out/'provenance.json',prov);write_json(out/'coverage.json',coverage);write_json(out/'data_manifest.json',manifest)
    years=c['development_years'] if phase=='development' else c['holdout_years'];predictions=[];ics=[];validations=[];selections=[]
    for year in years:
        for half in [1,2]:
            fold=f'{year}H{half}'
            for method in c['methods']:
                months=c['short_train_months'] if method in ['legacy_matched','signed_stable','reliable'] else c['long_train_months']
                a,b,r,q,info=split(panel,year,half,months,c);q.loc[q.exit_date>pd.Timestamp(f'{max(years)}-12-31'),['label','target']]=np.nan
                print('START',fold,method,flush=True);p,vp,models,selection=fit_method(a,b,r,q,method,c)
                dst=out/fold/method;dst.mkdir(parents=True)
                write_json(dst/'selection.json',{'split':info,'groups':selection,'provenance':prov})
                with (dst/'models.pkl').open('wb') as f:pickle.dump(models,f)
                write_json(dst/'checkpoint.json',{'sha256':sha256(dst/'models.pkl'),'provenance':prov})
                for group in list(c['weights'])+['fusion']:
                    z=q[['date','symbol','label','target']].copy();z['prediction']=p[group];z['method']=method;z['group']=group;z['fold']=fold;predictions.append(z)
                    d=daily_ic(q,p[group],c['min_cross_section']);d['method']=method;d['group']=group;d['fold']=fold;ics.append(d)
                    vd=daily_ic(b,vp[group],c['min_cross_section']);vd['method']=method;vd['group']=group;vd['fold']=fold;validations.append(vd)
                print('FUSION',fold,method,float(ics[-1].rank_ic.mean()),flush=True)
    pred=pd.concat(predictions,ignore_index=True);ic=pd.concat(ics,ignore_index=True);val=pd.concat(validations,ignore_index=True)
    pred.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g');ic.to_csv(out/'daily_ic.csv',index=False,float_format='%.17g');val.to_csv(out/'validation_ic.csv',index=False,float_format='%.17g')
    measures={m:ic_summary(g[g.group=='fusion'],c) for m,g in ic.groupby('method')};write_json(out/'ic_summary.json',measures)
    paths=[];statistics=[];cal=calendar[calendar<=f'{max(years)}-12-31']
    for method in c['methods']+['equal_weight']:
        d=pred[(pred.method==('legacy_matched' if method=='equal_weight' else method))&(pred.group=='fusion')]
        for policy in ['weekly','buffer']:
            if method=='equal_weight' and policy=='buffer':continue
            for cost in c['screen_cost_bps']:
                path=simulate(d,adj,cal,symbols,c,cost,policy=='buffer',method=='equal_weight');path['method']=method;path['policy']=policy;path['cost_bps']=cost;paths.append(path)
                statistics.append({'method':method,'policy':policy,'cost_bps':cost,**path_statistics(path)})
    pd.concat(paths,ignore_index=True).to_csv(out/'screen_paths.csv.gz',index=False,float_format='%.17g');pd.DataFrame(statistics).to_csv(out/'screen_summary.csv',index=False,float_format='%.17g')
    if phase=='development':
        candidates={m:z['coverage_adjusted_ic'] if z['ic_coverage']>=c['minimum_ic_coverage'] else -1. for m,z in measures.items()}
        winner=max(candidates,key=candidates.get);write_json(out/'champion_candidate.json',{'method':winner,'selection':'development coverage-adjusted mean IC, all methods included','scores':candidates,'config_sha256':sha256(config_path),'data_manifest_sha256':prov['data_manifest_sha256']})
    else:
        m=seal['method'];z=measures[m];t=pd.read_csv(out/'screen_summary.csv');row=t[(t.method==m)&(t.policy=='buffer')&(t.cost_bps==c['screen_primary_cost_bps'])].iloc[0]
        reasons=[]
        if z['ic_coverage']<c['minimum_ic_coverage']:reasons.append('insufficient_IC_coverage')
        if z['rank_ic'] is None or z['rank_ic']<c['minimum_ic']:reasons.append('IC_below_0.02')
        if z['passing_fold_fraction']<c['minimum_passing_fold_fraction']:reasons.append('IC_not_stable_in_75pct_folds')
        if z['ci95'][0] is None or z['ci95'][0]<=0:reasons.append('IC_interval_crosses_zero')
        if not np.isfinite(row.net_sharpe) or row.net_sharpe<=c['minimum_net_sharpe']:reasons.append('net_sharpe_not_above_1')
        write_json(out/'financial_gate.json',{'champion':m,'passed':not reasons,'reasons':reasons,'strict_pit':False,'paper_admitted':False,'orders_sent':0,'structural_blocks':['vintage_data_unverified','fractional_adjusted_return_screen_not_executable_ledger']})
    print('COMPLETE',out,flush=True)


def publish(out,reports):
    import shutil
    out=Path(out);r=Path(reports)
    if r.exists() and any(r.iterdir()):raise ValueError('No report overwrite')
    r.mkdir(parents=True)
    for p in out.glob('*'):
        if p.is_file():shutil.copy2(p,r/p.name)
    for p in out.glob('*/*/selection.json'):
        dst=r/'evidence'/(p.parent.parent.name+'-'+p.parent.name);dst.mkdir(parents=True);shutil.copy2(p,dst/'selection.json');shutil.copy2(p.parent/'checkpoint.json',dst/'checkpoint.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','development','holdout','publish']);p.add_argument('--outputs',default='outputs/v7/development');p.add_argument('--reports',default='reports/v7/development');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='publish':publish(a.outputs,a.reports)
    else:run(a.action,out=a.outputs)
if __name__=='__main__':main()
