"""First fusion diagnosis followed by an independently frozen single change."""
import argparse,json,pickle,subprocess
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from ..study import write_json
from ..data import sha256
from ..v4.metrics import daily_ic,block_mean_interval,path_statistics,paired_ic_increment
from ..v4.screen import screen_path
from .data import load,prepare,NEWS_FEATURES,FINANCIAL
from .models import GROUPS,fit


def splits(panel,year,half,config):
    begin=pd.Timestamp(year,1 if half==1 else 7,1);end=begin+pd.DateOffset(months=6)
    val=begin-pd.DateOffset(months=config['validation_months']);start=val-pd.DateOffset(months=config['train_months'])
    valid=panel.eligible & panel.label.notna() & panel.target.notna()
    count=panel[valid].groupby('date').symbol.count();valid &= panel.date.isin(count[count>=config['min_cross_section']].index)
    a=panel[valid&(panel.date>=start)&(panel.date<val)&(panel.exit_date<val)].copy()
    b=panel[valid&(panel.date>=val)&(panel.date<begin)&(panel.exit_date<begin)].copy()
    r=panel[valid&(panel.date>=start)&(panel.date<begin)&(panel.exit_date<begin)].copy()
    q=panel[panel.eligible&(panel.date>=begin)&(panel.date<end)].copy()
    # Same research rows across all groups, including the dates with no archive article.
    n=q.groupby('date').symbol.count();q=q[q.date.isin(n[n>=config['min_cross_section']].index)].copy()
    if min(len(a),len(b),len(q))<500:raise ValueError('Insufficient common-group fold')
    if not a.exit_date.max()<val or not r.exit_date.max()<begin:raise ValueError('Label leakage')
    return a,b,r,q,{'train_start':str(start),'validation_start':str(val),'test_start':str(begin),'test_end_exclusive':str(end),
                    'training_label_exit_max':str(a.exit_date.max()),'refit_label_exit_max':str(r.exit_date.max()),
                    'train_rows':len(a),'validation_rows':len(b),'refit_rows':len(r),'test_rows':len(q)}


def summary(d,config):
    x=d.rank_ic.dropna().to_numpy();sd=float(np.std(x,ddof=1))
    return {'days':len(x),'rank_ic':float(x.mean()),'ic_ir':float(x.mean()/sd) if sd>0 else None,
            'ci95':block_mean_interval(x,config['seed'],config['bootstrap_repetitions'],config['bootstrap_block']),
            'ir_definition':'daily mean RankIC / sample standard deviation, no sqrt252'}


def run(phase,config_path='configs/v6.json',root='data/v6',outputs='outputs/v6/pilot',entity_only=False,variant='baseline'):
    config=json.loads(Path(config_path).read_text());out=Path(outputs)
    if out.exists() and any(out.iterdir()):raise ValueError('Use empty output directory')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Freeze clean source before fitting')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if variant not in ['baseline','entity_filter','entity_tfidf']:raise ValueError('Unsealed variant')
    if phase=='followup':
        plan=json.loads(Path('configs/v6_improvement.json').read_text())
        if sha256('reports/v6/pilot/ic_summary.json')!=plan['pilot_summary_sha256']:raise ValueError('Pilot evidence changed after direction selection')
        if config['weights']!=plan['weights'] or config['followup_years']!=plan['followup_years'] or variant not in plan['variants']:raise ValueError('Follow-up differs from sealed plan')
    config['variant']=variant;entity_only=entity_only or variant!='baseline'
    panel,raw,adj,calendar,symbols,manifest,prices,coverage=load(config_path,root,entity_only,variant=='entity_tfidf')
    out.mkdir(parents=True)
    prov={'source_commit':source,'config_sha256':sha256(config_path),'data_manifest_sha256':sha256(Path(root)/'manifest.json'),
          'price_manifest_sha256':sha256(Path(config['data_prices'])/'manifest.json'),'phase':phase,'entity_only':entity_only,
          'variant':variant,'improvement_sha256':sha256('configs/v6_improvement.json') if phase=='followup' else None,
          'seed':config['seed'],'strict_pit':False,'availability_mode':config['availability_mode'],
          'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','openpyxl']}}
    write_json(out/'provenance.json',prov);write_json(out/'coverage.json',coverage);write_json(out/'data_manifest.json',manifest)
    all_pred=[];all_ic=[];all_val=[]
    for year in config['pilot_years'] if phase=='pilot' else config['followup_years']:
        for half in [1,2]:
            name=f'{year}H{half}';print('START',name,flush=True)
            a,b,r,q,split=splits(panel,year,half,config)
            phase_last=max(config['pilot_years'] if phase=='pilot' else config['followup_years'])
            q.loc[q.exit_date>pd.Timestamp(f'{phase_last}-12-31'),['label','target']]=np.nan
            pred,val_pred,models,selection=fit(a,b,r,q,config)
            dst=out/name;dst.mkdir();write_json(dst/'selection.json',{'split':split,'selection':selection,'provenance':prov})
            with (dst/'models.pkl').open('wb') as f:pickle.dump(models,f)
            write_json(dst/'checkpoint.json',{'sha256':sha256(dst/'models.pkl'),'provenance':prov})
            for arm in list(GROUPS)+['fusion']:
                z=q[['date','symbol','label','target']].copy();z['prediction']=pred[arm];z['model']=arm;z['fold']=name;all_pred.append(z)
                d=daily_ic(q,pred[arm],config['min_cross_section']);d['model']=arm;d['fold']=name;all_ic.append(d)
                v=daily_ic(b,val_pred[arm],config['min_cross_section']);v['model']=arm;v['fold']=name;all_val.append(v)
                print(name,arm,float(d.rank_ic.mean()),flush=True)
    predictions=pd.concat(all_pred,ignore_index=True);ics=pd.concat(all_ic,ignore_index=True);val=pd.concat(all_val,ignore_index=True)
    predictions.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g');ics.to_csv(out/'daily_ic.csv',index=False,float_format='%.17g');val.to_csv(out/'validation_ic.csv',index=False,float_format='%.17g')
    measures={arm:summary(d,config) for arm,d in ics.groupby('model')}
    increments={arm:paired_ic_increment(ics[ics.model=='price_volume'],ics[ics.model==arm],
        {'seed':config['seed'],'signal_gate':{'bootstrap_repetitions':config['bootstrap_repetitions']}}) for arm in ['fundamental','alternative','fusion']}
    write_json(out/'ic_summary.json',measures);write_json(out/'incremental_ic.json',increments)
    # Correlation reveals redundant group information; no inference from training fitness.
    wide=predictions.pivot(index=['date','symbol'],columns='model',values='prediction');wide.corr().to_csv(out/'prediction_correlation.csv',float_format='%.17g')
    paths=[];stats=[];screen_config=dict(config);screen_config['symbols']=symbols
    for arm in list(GROUPS)+['fusion','equal_weight']:
        for cost in config['screen_cost_bps']:
            p=predictions[predictions.model==('price_volume' if arm=='equal_weight' else arm)]
            # Keep the phase calendar, not future years. The price helper always terminates
            # at its provided final calendar date, so pass a phase-specific calendar below.
            phase_end=f'{max(config["pilot_years"] if phase=="pilot" else config["followup_years"])}-12-31'
            cal=calendar[calendar<=min(config['end'],phase_end)]
            path=screen_path(p,adj,cal,symbols,screen_config,cost,arm=='equal_weight')
            path['model']=arm;path['cost_bps']=cost;paths.append(path);stats.append({'model':arm,'cost_bps':cost,**path_statistics(path)})
    pd.concat(paths,ignore_index=True).to_csv(out/'screen_paths.csv.gz',index=False,float_format='%.17g');pd.DataFrame(stats).to_csv(out/'screen_summary.csv',index=False,float_format='%.17g')
    write_json(out/'status.json',{'status':'completed','phase':phase,'strict_pit':False,'paper_admitted':False,'orders_sent':0,
            'scope':'Publication-aware reconstructed retrospective research; no prospective confirmation'})
    print('COMPLETE',out,flush=True)


def publish(outputs,reports):
    import shutil
    out=Path(outputs);r=Path(reports)
    if r.exists() and any(r.iterdir()):raise ValueError('Do not overwrite a prior report')
    r.mkdir(parents=True)
    for p in out.glob('*'):
        if p.is_file():shutil.copy2(p,r/p.name)
    e=r/'evidence';e.mkdir()
    for p in out.glob('*/selection.json'):shutil.copy2(p,e/(p.parent.name+'-selection.json'))
    for p in out.glob('*/checkpoint.json'):shutil.copy2(p,e/(p.parent.name+'-checkpoint.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pilot','followup','publish']);p.add_argument('--config',default='configs/v6.json');p.add_argument('--data',default='data/v6');p.add_argument('--outputs',default='outputs/v6/pilot');p.add_argument('--reports',default='reports/v6/pilot');p.add_argument('--entity-only',action='store_true');p.add_argument('--variant',default='baseline',choices=['baseline','entity_filter','entity_tfidf']);a=p.parse_args()
    if a.action=='prepare':prepare(a.config,root=a.data)
    elif a.action=='publish':publish(a.outputs,a.reports)
    else:run(a.action,a.config,a.data,a.outputs,a.entity_only,a.variant)
if __name__=='__main__':main()
