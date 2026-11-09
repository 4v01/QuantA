"""Bounded research loop, validation utility selection and sealed 2025 champion."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
import torch

from ..data import fetch,sha256
from ..study import provenance,write_json
from .data import samples,split
from .execution import simulate,certainty_equivalent
from .factors import FactorSearch,Scale
from .models import Ridge,Trees,LSTM,fit_lstm


def candidates(arm,config):
    if arm.endswith('ridge'):return [{'alpha':a} for a in config['ridge_candidates']]
    if arm.endswith('hgb'):return config['hgb_candidates']
    return config['neural_candidates']


def seeds(arm,config):return config['seeds'] if arm.endswith('lstm') else [config['seeds'][0]]


def create(arm,features,config,candidate,seed):
    if arm.endswith('ridge'):return Ridge(candidate['alpha'])
    if arm.endswith('hgb'):return Trees(config,candidate,seed)
    return LSTM(features,config['hidden_size'],seed)


def save_checkpoint(path,model,scale,search,arm,prov):
    # Local-only original models/derived statistics, no quote histories embedded.
    with Path(path).open('wb') as f:pickle.dump({'model':model,'scale':scale,'search':search,'arm':arm,'provenance':prov},f,protocol=5)


def run_year(year,x,meta,corrs,config,prov,root):
    train_ids,val_ids,val_label_ids,test_ids,refit_ids=split(meta,year)
    y=meta[['label_return','label_downside']].to_numpy(dtype=np.float64)
    search=FactorSearch(config['factor_search']).fit(x[train_ids],y[train_ids,0],meta.iloc[train_ids].date.dt.year.to_numpy())
    refit_search=FactorSearch(config['factor_search']).fit(x[refit_ids],y[refit_ids,0],meta.iloc[refit_ids].date.dt.year.to_numpy())
    write_json(root/f'{year}-factor_search.json',{'training':search.state(),'refit':refit_search.state(),
        'train_last_label_exit':str(meta.iloc[train_ids].label_exit.max()),
        'validation_first_entry':str(meta.iloc[val_ids].entry_date.min()),
        'validation_last_label_exit':str(meta.iloc[val_label_ids].label_exit.max()),
        'test_first_entry':str(meta.iloc[test_ids].entry_date.min()),
        'refit_last_label_exit':str(meta.iloc[refit_ids].label_exit.max())})
    meta.iloc[val_ids].to_csv(root/f'{year}-validation_metadata.csv',index=False)
    meta.iloc[test_ids].to_csv(root/f'{year}-test_metadata.csv',index=False)
    # Index validation-label rows in the complete, unfiltered validation execution calendar.
    val_mask=np.isin(val_ids,val_label_ids)
    for arm in config['arms']:
        tx=search.transform(x[train_ids],arm);scale=Scale().fit(tx);tx=scale.apply(tx)
        vx=scale.apply(search.transform(x[val_ids],arm))
        rx=refit_search.transform(x[refit_ids],arm);refit_scale=Scale().fit(rx);rx=refit_scale.apply(rx)
        qx=refit_scale.apply(refit_search.transform(x[test_ids],arm))
        for seed in seeds(arm,config):
            dest=root/f'{year}-{arm}-{seed}';dest.mkdir(exist_ok=True)
            if (dest/'result.json').exists():
                saved=json.loads((dest/'result.json').read_text())
                if saved['provenance']!=prov:raise ValueError('Mixed resume provenance')
                continue
            records=[];winner_model=None;winner_prediction=None;winner=None;started=time.time()
            for index,candidate in enumerate(candidates(arm,config)):
                begin=time.time()
                try:
                    model=create(arm,tx.shape[-1],config,candidate,seed)
                    if arm.endswith('lstm'):
                        history,best=fit_lstm(model,tx,y[train_ids],config,candidate,seed,(vx[val_mask],y[val_label_ids]))
                    else:model.fit(tx,y[train_ids]);history=[];best=None
                    prediction=model.predict(vx)
                    path=simulate(meta.iloc[val_ids],prediction,corrs,config,config['primary_profile'],config['primary_cost_bps'])
                    utility=certainty_equivalent(path,config['validation_ce_gamma'])
                    record={'index':index,'config':candidate,'status':'success','validation_net_ce':utility,
                            'validation_return_rmse':float(np.sqrt(np.mean((prediction[val_mask,0]-y[val_label_ids,0])**2))),
                            'history':history,'best':best,'elapsed_seconds':time.time()-begin}
                    if winner is None or utility>winner['validation_net_ce']:
                        winner=record;winner_model=model;winner_prediction=prediction
                except Exception as error:
                    record={'index':index,'config':candidate,'status':'failed','error':repr(error),'elapsed_seconds':time.time()-begin}
                records.append(record)
            if winner is None:
                write_json(dest/'failure.json',{'candidates':records,'provenance':prov});raise RuntimeError('All candidates failed '+dest.name)
            write_json(dest/'selection.json',{'selected':winner,'candidates':records,'criterion':'maximum_validation_net_certainty_equivalent',
                         'train_rows':len(train_ids),'validation_score_rows':len(val_label_ids),'validation_execution_rows':len(val_ids),
                         'refit_rows':len(refit_ids),'test_execution_rows':len(test_ids),'provenance':prov})
            save_checkpoint(dest/'validation.pkl',winner_model,scale,search,arm,prov)
            pd.DataFrame(winner_prediction,columns=['mu','downside']).to_csv(dest/'validation_predictions.csv',index=False,float_format='%.12g')
            final=create(arm,rx.shape[-1],config,winner['config'],seed)
            if arm.endswith('lstm'):
                history,_=fit_lstm(final,rx,y[refit_ids],config,winner['config'],seed,epochs=winner['best']['epoch'])
            else:final.fit(rx,y[refit_ids]);history=[]
            save_checkpoint(dest/'final.pkl',final,refit_scale,refit_search,arm,prov)
            prediction=final.predict(qx)
            pd.DataFrame(prediction,columns=['mu','downside']).to_csv(dest/'test_predictions.csv',index=False,float_format='%.12g')
            record={'arm':arm,'seed':seed,'test_year':year,'selected_validation_net_ce':winner['validation_net_ce'],
                    'features':int(rx.shape[-1]),'refit_epochs':winner['best']['epoch'] if arm.endswith('lstm') else None,
                    'refit_history':history,'elapsed_seconds':time.time()-started,
                    'hashes':{name:sha256(dest/name) for name in ['selection.json','validation.pkl','final.pkl','validation_predictions.csv','test_predictions.csv']},
                    'provenance':prov}
            write_json(dest/'result.json',record)
            print(json.dumps({'year':year,'arm':arm,'seed':seed,'validation_net_ce':winner['validation_net_ce'],'elapsed_seconds':record['elapsed_seconds']}),flush=True)


def run(mode,config_path,data_root,outputs):
    config=json.loads(Path(config_path).read_text());torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    x,meta,corrs,manifest=samples(data_root,config)
    if mode=='development' and meta.date.dt.year.max()>max(config['development_years']):
        raise ValueError('Development must use the old data snapshot without 2025 quotes')
    prov=provenance(config_path,manifest);prov.update(study='v2',scikit_learn=sklearn.__version__)
    base=Path(outputs);root=base/mode;root.mkdir(parents=True,exist_ok=True)
    if (root/'provenance.json').exists() and json.loads((root/'provenance.json').read_text())!=prov:raise ValueError('Output provenance changed')
    if mode=='confirmation':
        champion=json.loads((base/'champion.json').read_text())
        if champion['source_commit']!=prov['source_commit'] or champion['config_sha256']!=prov['config_sha256']:
            raise ValueError('Champion not sealed under this code/config')
    write_json(root/'provenance.json',prov);write_json(root/'data_manifest.json',manifest);np.save(root/'correlations.npy',corrs)
    years=config['development_years'] if mode=='development' else [config['confirmation_year']]
    for year in years:run_year(year,x,meta,corrs,config,prov,root)
    if mode=='development':
        grouped={}
        for arm in config['arms']:
            records=[json.loads((root/f'{year}-{arm}-{seed}'/'result.json').read_text()) for year in years for seed in seeds(arm,config)]
            grouped[arm]=float(np.mean([r['selected_validation_net_ce'] for r in records]))
        winner=max(config['arms'],key=lambda arm:grouped[arm])
        seal={'arm':winner,'profile':config['primary_profile'],'selection_scores':grouped,
              'criterion':config['champion_selection'],'source_commit':prov['source_commit'],'config_sha256':prov['config_sha256'],
              'development_data_sha256':prov['data_sha256'],'confirmation_data_opened':False,
              'neural_seed_aggregation':'arithmetic mean of all predeclared seed forecasts; deterministic arms use seed17'}
        write_json(base/'champion.json',seal);print('SEALED_CHAMPION '+json.dumps(seal),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['development','fetch-confirmation','confirmation'])
    p.add_argument('--config',default='configs/v2.json');p.add_argument('--data');p.add_argument('--outputs',default='outputs/v2');a=p.parse_args()
    if a.action=='fetch-confirmation':
        if not (Path(a.outputs)/'champion.json').exists():raise ValueError('Seal champion before downloading 2025 data')
        manifest=fetch(json.loads(Path(a.config).read_text()),a.data or 'data/v2')
        print(json.dumps({'confirmation_data_sha256':manifest['quotes_sha256']}))
    else:run(a.action,a.config,a.data or ('data' if a.action=='development' else 'data/v2'),a.outputs)


if __name__=='__main__':main()
