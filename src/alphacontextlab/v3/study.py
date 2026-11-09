"""Small shared Ridge forecast, prespecified portfolio families and cash-share ledgers."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd
import sklearn
import torch

from ..data import sha256
from ..study import provenance,write_json
from ..v2.data import split
from ..v2.factors import Scale
from ..v2.models import Ridge
from ..v2.execution import certainty_equivalent
from .data import load,prepare_events
from .ledger import simulate
from .statistics import centered_max_check


def fetch_raw(config,root):
    root=Path(root);raw=root/'raw';raw.mkdir(parents=True,exist_ok=True);frames=[];sources=[]
    for symbol in config['symbols']:
        url=('https://push2his.eastmoney.com/api/qt/stock/kline/get?'
             f'secid={0 if symbol.startswith("1") else 1}.{symbol}&klt=101&fqt=0'
             f'&beg={config["start"].replace("-","")}&end={config["end"].replace("-","")}&lmt=10000'
             '&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61')
        path=raw/f'{symbol}.json'
        if not path.exists():
            with urlopen(url,timeout=45) as response:payload=response.read()
            obj=json.loads(payload)
            if obj.get('rc')!=0 or not obj.get('data',{}).get('klines'):raise ValueError('No raw prices')
            path.write_bytes(payload)
        obj=json.loads(path.read_bytes());f=pd.DataFrame([s.split(',') for s in obj['data']['klines']],
            columns=['date','open','close','high','low','volume','amount','amplitude','pct','change','turnover'])
        for c in f.columns[1:]:f[c]=pd.to_numeric(f[c],errors='raise')
        if f.date.duplicated().any() or not f.date.is_monotonic_increasing or (f[['open','close','low']]<=0).any().any():raise ValueError('Bad raw quotes')
        f['symbol']=symbol;f['adjustment']=0;frames.append(f);sources.append({'symbol':symbol,'url':url,'sha256':sha256(path),'rows':len(f)})
    pd.concat(frames,ignore_index=True).to_csv(root/'quotes.csv',index=False,float_format='%.12g')
    manifest={'provider':'Eastmoney raw fqt0 only','retrieved_utc':datetime.now(timezone.utc).isoformat(),
              'quotes_sha256':sha256(root/'quotes.csv'),'sources':sources}
    write_json(root/'manifest.json',manifest);return manifest


def policy_grid(config):
    return [(name,None if name=='legacy_discrete' else gamma) for name in config['policies']
            for gamma in ([None] if name=='legacy_discrete' else config['gamma_candidates'])]


def one_year(year,x,meta,market,version,config,prov,root):
    folder=root/str(year);folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():
        if json.loads((folder/'result.json').read_text())['provenance']!=prov:raise ValueError('Mixed V3 resume')
        return
    begin=time.time();train,val,val_labeled,test,refit=split(meta,year);y=meta[['label_return','label_downside']].to_numpy(dtype=float)
    scale=Scale().fit(x[train]);tx=scale.apply(x[train]);vx=scale.apply(x[val]);valid=np.isin(val,val_labeled)
    model_records=[];models=[];predictions=[]
    for j,alpha in enumerate(config['ridge_candidates']):
        model=Ridge(alpha).fit(tx,y[train]);pred=model.predict(vx)
        rmse=float(np.sqrt(np.mean((pred[valid]-y[val_labeled])**2)))
        model_records.append({'index':j,'alpha':alpha,'validation_joint_rmse':rmse});models.append(model);predictions.append(pred)
    model_winner=min(model_records,key=lambda r:r['validation_joint_rmse']);chosen=model_winner['index']
    primary=config['scenarios']['primary'];benchmark,_=simulate(meta.iloc[val],np.zeros((len(val),2)),market,'csi300',None,primary,config)
    policy_records=[];validation_paths=[];matrix=[];case=0
    for j,pred in enumerate(predictions):
        for family,gamma in policy_grid(config):
            record={'case':case,'forecast_index':j,'family':family,'gamma':gamma}
            try:
                path,_=simulate(meta.iloc[val],pred,market,family,gamma,primary,config)
                ce=certainty_equivalent(path,config['validation_ce_gamma'])
                record.update(status='success',validation_net_ce=ce,solver_fallbacks=int(path.solver_failures_cumulative.max()))
                p=path[['entry_date','exit_date','net_return']].copy();p['case']=case;validation_paths.append(p)
                matrix.append(path.net_return.to_numpy()-benchmark.net_return.to_numpy())
            except Exception as error:record.update(status='failed',error=repr(error))
            policy_records.append(record);case+=1
    selected={}
    for family in config['policies']:
        available=[p for p in policy_records if p['forecast_index']==chosen and p['family']==family and p['status']=='success']
        if not available:write_json(folder/'failure.json',{'policy_candidates':policy_records});raise RuntimeError('No successful policy '+family)
        selected[family]=max(available,key=lambda p:p['validation_net_ce'])
    selection={'forecast_candidates':model_records,'selected_forecast':model_winner,'policy_candidates':policy_records,'selected_policies':selected,
               'train_last_label_exit':str(meta.iloc[train].label_exit.max()),'validation_first_entry':str(meta.iloc[val].entry_date.min()),
               'validation_last_label_exit':str(meta.iloc[val_labeled].label_exit.max()),'test_first_entry':str(meta.iloc[test].entry_date.min()),
               'refit_last_label_exit':str(meta.iloc[refit].label_exit.max()),'train_rows':len(train),'validation_rows':len(val),'refit_rows':len(refit),
               'validation_search_check':centered_max_check(np.column_stack(matrix),config['bootstrap_repetitions'],config['bootstrap_block_length'])}
    write_json(folder/'selection.json',selection)
    pd.concat(validation_paths,ignore_index=True).to_csv(folder/'validation_paths.csv.gz',index=False,float_format='%.12g',compression={'method':'gzip','mtime':0})
    benchmark[['entry_date','exit_date','net_return']].to_csv(folder/'validation_benchmark.csv',index=False,float_format='%.12g')
    final_scale=Scale().fit(x[refit]);model=Ridge(model_winner['alpha']).fit(final_scale.apply(x[refit]),y[refit])
    with (folder/'final.pkl').open('wb') as f:pickle.dump({'model':model,'scale':final_scale,'provenance':prov},f,protocol=5)
    with (folder/'validation.pkl').open('wb') as f:pickle.dump({'model':models[chosen],'scale':scale,'provenance':prov},f,protocol=5)
    pred=model.predict(final_scale.apply(x[test]));pd.DataFrame(pred,columns=['mu','downside']).to_csv(folder/'predictions.csv',index=False,float_format='%.12g')
    meta.iloc[test].to_csv(folder/'metadata.csv',index=False)
    scenario_records=[]
    for family in list(config['policies'])+config['baselines']:
        gamma=selected[family]['gamma'] if family in selected else None
        for scenario_name,scenario in config['scenarios'].items():
            dest=folder/family;dest.mkdir(exist_ok=True)
            try:
                path,trades=simulate(meta.iloc[test],pred,market,family,gamma,scenario,config)
                path.to_csv(dest/f'{scenario_name}-path.csv',index=False,float_format='%.12g');trades.to_csv(dest/f'{scenario_name}-orders.csv',index=False)
                scenario_records.append({'family':family,'scenario':scenario_name,'status':'success',
                    'path_sha256':sha256(dest/f'{scenario_name}-path.csv'),'orders_sha256':sha256(dest/f'{scenario_name}-orders.csv')})
            except Exception as error:
                failure={'error':repr(error),'provenance':prov};write_json(dest/f'{scenario_name}-failure.json',failure)
                scenario_records.append({'family':family,'scenario':scenario_name,'status':'failed','error':repr(error)})
    result={'year':year,'forecast_candidates':len(model_records),'policy_validation_candidates':len(policy_records),'final_forecasts':1,
            'selected_forecast':model_winner,'selected_policies':selected,'scenarios':scenario_records,'elapsed_seconds':time.time()-begin,'provenance':prov,
            'hashes':{name:sha256(folder/name) for name in ['selection.json','predictions.csv','metadata.csv','final.pkl','validation.pkl','validation_paths.csv.gz','validation_benchmark.csv']}}
    write_json(folder/'result.json',result)
    print(json.dumps({'year':year,'alpha':model_winner['alpha'],'validation_ce':{k:v['validation_net_ce'] for k,v in selected.items()},'elapsed_seconds':result['elapsed_seconds']}),flush=True)


def run(mode,config_path,quotes,events,outputs):
    c=json.loads(Path(config_path).read_text());torch.set_num_threads(2);x,meta,market,version=load(quotes,events,c)
    if mode=='development' and market.dates.max().year>max(c['development_years']):raise ValueError('Development includes the new check year')
    prov=provenance(config_path,{'quotes_sha256':version['quotes_sha256']});prov.update(study='v3',sklearn=sklearn.__version__,corporate_actions_sha256=version['corporate_actions_sha256'])
    base=Path(outputs);root=base/mode;root.mkdir(parents=True,exist_ok=True)
    if (root/'provenance.json').exists() and json.loads((root/'provenance.json').read_text())!=prov:raise ValueError('Refusing different provenance overwrite')
    if mode=='confirmation':
        seal=json.loads((base/'champion.json').read_text())
        if seal['source_commit']!=prov['source_commit'] or seal['config_sha256']!=prov['config_sha256'] or seal['corporate_actions_sha256']!=prov['corporate_actions_sha256']:
            raise ValueError('Confirmation source/config/events differ from the public seal')
    write_json(root/'provenance.json',prov);write_json(root/'data_version.json',version)
    years=c['development_years'] if mode=='development' else [c['confirmation_year']]
    for year in years:one_year(year,x,meta,market,version,c,prov,root)
    if mode=='development':
        scores={family:float(np.mean([json.loads((root/str(year)/'selection.json').read_text())['selected_policies'][family]['validation_net_ce'] for year in years])) for family in c['policies']}
        champion=max(c['policies'],key=lambda p:scores[p])
        seal={'family':champion,'selection_scores':scores,'criterion':'mean_validation_net_CE_development_folds','source_commit':prov['source_commit'],
              'config_sha256':prov['config_sha256'],'corporate_actions_sha256':prov['corporate_actions_sha256'],
              'development_quotes_sha256':version['quotes_sha256'],'new_price_snapshot_loaded':False,'strict_prospective_blind_trial':False,
              'metadata_available_before_seal':'2026 corporate actions and isolated web-search snippets; no project-wide 2026 return backtest'}
        write_json(base/'champion.json',seal);print('SEALED '+json.dumps(seal),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-events','development','fetch-confirmation','confirmation'])
    p.add_argument('--config',default='configs/v3.json');p.add_argument('--data');p.add_argument('--events',default='data/v3_corporate');p.add_argument('--outputs',default='outputs/v3');a=p.parse_args()
    if a.action=='prepare-events':print(json.dumps({'events':len(prepare_events(a.events)['events'])}))
    elif a.action=='fetch-confirmation':
        if not (Path(a.outputs)/'champion.json').exists():raise ValueError('Seal before loading new prices')
        print(json.dumps(fetch_raw(json.loads(Path(a.config).read_text()),a.data or 'data/v3_prices')))
    else:run(a.action,a.config,a.data or ('data/v2' if a.action=='development' else 'data/v3_prices'),a.events,a.outputs)
