import argparse,json,pickle,shutil,subprocess
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.metrics import daily_ic
from ..v7.study import ic_summary
from ..v8.portfolio import make_tape
from ..v8.study import path_metrics
from .data import load
from .models import fit_fold,calibrate,apply_calibration
from .portfolio import simulate


def select_candidate(summary,signals,base,config):
    candidates=[]
    for method in config['methods']:
        s=signals[method]
        signal_ok=s['ic_coverage']>=base['minimum_ic_coverage'] and s['rank_ic'] is not None and s['rank_ic']>=base['minimum_ic'] and s['passing_fold_fraction']>=base['minimum_passing_fold_fraction']
        for policy in config['policies']:
            row=summary[(summary.method==method)&(summary.policy==policy)&(summary.cost_bps==config['primary_cost_bps'])].iloc[0]
            eligible=signal_ok and row.mean_exposure>=config['minimum_development_exposure'] and row.noncash_days>=config['minimum_development_noncash_days'] and np.isfinite(row.net_sharpe)
            candidates.append({'method':method,'policy':policy,'eligible':bool(eligible),'net_sharpe':float(row.net_sharpe) if np.isfinite(row.net_sharpe) else None})
    eligible=[r for r in candidates if r['eligible']]
    winner=max(eligible,key=lambda r:r['net_sharpe']) if eligible else {'method':'matched_control','policy':'weekly_buffer'}
    return {'method':winner['method'],'policy':winner['policy'],'selection_qualified':bool(eligible),'candidates':candidates,
            'criterion':'H5 signal stability/coverage then development 20bp net Sharpe, >=10% exposure and >=100 noncash days; config order breaks ties.'}


def financial_gate(signals,summary,seal,base,config):
    s=signals[seal['method']]; row=summary[(summary.method==seal['method'])&(summary.policy==seal['policy'])&(summary.cost_bps==20)].iloc[0]; reasons=[]
    if s['rank_ic'] is None or s['rank_ic']<base['minimum_ic']:reasons.append('H5_IC_below_0.02')
    if s['ic_coverage']<base['minimum_ic_coverage']:reasons.append('H5_IC_coverage_below_90pct')
    if s['passing_fold_fraction']<base['minimum_passing_fold_fraction']:reasons.append('H5_IC_fold_stability_failed')
    if s['ci95'][0] is None or s['ci95'][0]<=0:reasons.append('H5_IC_interval_crosses_zero')
    if not np.isfinite(row.net_sharpe) or row.net_sharpe<=config['minimum_net_sharpe']:reasons.append('net_sharpe_not_above_1')
    if not seal['selection_qualified']:reasons.append('development_selection_not_qualified')
    return {'method':seal['method'],'policy':seal['policy'],'financial_passed':not reasons,'reasons':reasons,
            'strict_pit':False,'paper_admitted':False,'orders_sent':0,'structural_blocks':['fixed_survivor_universe','financial_vintages_and_share_basis_unverified','non_executable_adjusted_return_screen']}


def excess_r2(pred):
    rows={}
    for method,g in pred.groupby('method'):
        g=g.dropna(subset=['label20','expected_excess20']).copy(); y=g.label20-g.groupby('date').label20.transform('mean')
        sse=float(((y-g.expected_excess20)**2).sum());sst=float(((y-y.mean())**2).sum())
        rows[method]={'rows':len(g),'excess_H20_r2':1-sse/sst if sst>0 else None,'excess_H20_rmse':float(np.sqrt(sse/len(g))),
                      'note':'On H20 cross-section demeaned returns, not absolute returns or IC squared.'}
    return rows


def run(phase,out):
    config=json.loads(Path('configs/v9.json').read_text());base=json.loads(Path(config['base_config']).read_text());out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Freeze clean source/config first')
    seal=json.loads(Path('reports/v9/champion_seal.json').read_text()) if phase=='evaluation' else None
    if seal and seal['config_sha256']!=sha256('configs/v9.json'):raise ValueError('Sealed config mismatch')
    panel,adj,calendar,symbols,manifest,quality=load(); years=config['development_years'] if phase=='development' else config['evaluation_years']
    end=f'{max(years)}-12-31'; tape=make_tape(adj,calendar,symbols,end); out.mkdir(parents=True)
    prov={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'phase':phase,'seed':17,
          'config_sha256':sha256('configs/v9.json'),'base_config_sha256':sha256(config['base_config']),
          'data_manifest_sha256':sha256('data/v7/manifest.json'),'price_manifest_sha256':sha256('data/v4_akshare/manifest.json'),
          'symbols':symbols,'strict_pit':False,'versions':{x:version(x) for x in ['numpy','pandas','scipy','scikit-learn','lightgbm']},
          'seal_sha256':sha256('reports/v9/champion_seal.json') if seal else None,
          'development_forecasts_sha256':sha256('reports/v9/development/predictions.csv.gz') if seal else None}
    write_json(out/'provenance.json',prov);write_json(out/'coverage.json',quality);write_json(out/'data_manifest.json',manifest)
    tape.to_csv(out/'market_tape.csv.gz',index=False,float_format='%.17g')
    if phase=='evaluation':
        prior=pd.read_csv('reports/v9/development/predictions.csv.gz',dtype={'symbol':str},parse_dates=['date','exit_date','exit20'],float_precision='round_trip')
    else:prior=pd.DataFrame({'date':pd.to_datetime([]),'method':pd.Series([],dtype=str),'symbol':pd.Series([],dtype=str),'prediction':pd.Series([],dtype=float),'label20':pd.Series([],dtype=float),'exit20':pd.to_datetime([])})
    all_predictions=[]; selections=[]
    for year in (config['warmup_years']+years if phase=='development' else years):
        for half in [1,2]:
            fold=f'{year}H{half}';begin=pd.Timestamp(year,1 if half==1 else 7,1)
            for method in config['methods']:
                print('FIT',phase,fold,method,flush=True)
                p,models,selection=fit_fold(panel,year,half,method,base,config)
                p.loc[p.exit_date>pd.Timestamp(end),['label','target']]=np.nan;p.loc[p.exit20>pd.Timestamp(end),['label20','target20']]=np.nan
                p['method']=method;p['fold']=fold;p['evaluation_period']=year in years
                c=calibrate(prior,begin,method,config);selection['return_calibration']=c
                p['expected_excess20']=apply_calibration(p,c)
                dst=out/'evidence'/(fold+'-'+method);dst.mkdir(parents=True)
                with (dst/'models.pkl').open('wb') as f:pickle.dump(models,f)
                write_json(dst/'selection.json',selection);write_json(dst/'checkpoint.json',{'sha256':sha256(dst/'models.pkl'),'provenance':prov})
                all_predictions.append(p);prior=pd.concat([prior,p],ignore_index=True)
    predictions=pd.concat(all_predictions,ignore_index=True);predictions.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g')
    pred=predictions[predictions.evaluation_period];ics=[];signals={};secondary={}
    for method,g in pred.groupby('method'):
        for horizon in [5,20]:
            parts=[]
            for fold,q in g.groupby('fold'):
                temp=q.copy()
                if horizon==20:temp['label']=temp.label20
                d=daily_ic(temp,temp.prediction,base['min_cross_section']);d['method']=method;d['fold']=fold;d['horizon']=horizon;parts.append(d)
            d=pd.concat(parts,ignore_index=True);ics.append(d)
            (signals if horizon==5 else secondary)[method]=ic_summary(d,base)
    pd.concat(ics,ignore_index=True).to_csv(out/'daily_ic.csv',index=False,float_format='%.17g');write_json(out/'ic_summary.json',signals);write_json(out/'secondary_H20_ic.json',secondary)
    write_json(out/'return_metrics.json',excess_r2(pred));paths=[];holdings=[];summary=[]
    pairs=[(m,p) for m in config['methods'] for p in config['policies']]+[('equal_weight',p) for p in config['benchmarks']]
    for method,policy in pairs:
        q=pred[pred.method==('matched_control' if method=='equal_weight' else method)]
        for cost in config['cost_bps']:
            print('ALLOCATE',phase,method,policy,cost,flush=True)
            path,held=simulate(q,tape,symbols,base,config,method,policy,cost)
            summary.append({'method':method,'policy':policy,'cost_bps':cost,**path_metrics(path)})
            for frame in [path,held]:frame['method']=method;frame['policy']=policy;frame['cost_bps']=cost
            paths.append(path);holdings.append(held)
    pd.concat(paths,ignore_index=True).to_csv(out/'screen_paths.csv.gz',index=False,float_format='%.17g')
    pd.concat(holdings,ignore_index=True).to_csv(out/'holdings.csv.gz',index=False,float_format='%.17g')
    summary=pd.DataFrame(summary);summary.to_csv(out/'screen_summary.csv',index=False,float_format='%.17g')
    if phase=='development':write_json(out/'champion_candidate.json',select_candidate(summary,signals,base,config))
    else:write_json(out/'financial_gate.json',financial_gate(signals,summary,seal,base,config))
    print('COMPLETE',phase,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['development','evaluation','publish']);p.add_argument('--outputs',default='outputs/v9/development');p.add_argument('--reports',default='reports/v9/development');a=p.parse_args()
    if a.action=='publish':
        if Path(a.reports).exists():raise ValueError('Never overwrite published evidence')
        shutil.copytree(a.outputs,a.reports,ignore=shutil.ignore_patterns('*.pkl'))
    else:run(a.action,a.outputs)
if __name__=='__main__':main()
