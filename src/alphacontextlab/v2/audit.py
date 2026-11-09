"""Public result arithmetic plus optional local two-stage checkpoint replay."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np
import pandas as pd
import torch

from ..backtest import statistics
from ..data import sha256
from ..study import write_json
from .data import samples,split
from .execution import simulate,certainty_equivalent
from .report import confirmation_test
from .study import seeds


def audit(reports='reports/v2'):
    root=Path(reports);prov=json.loads((root/'provenance.json').read_text());seal=json.loads((root/'champion.json').read_text())
    frozen=subprocess.check_output(['git','show',prov['source_commit']+':configs/v2.json'])
    if hashlib.sha256(frozen).hexdigest()!=prov['config_sha256']:raise ValueError('Frozen config mismatch')
    config=json.loads(frozen);expected={(y,a,s) for y in config['development_years']+[config['confirmation_year']] for a in config['arms'] for s in seeds(a,config)}
    observed=set();validation={};candidate_count=0
    for p in (root/'evidence').glob('*-result.json'):
        record=json.loads(p.read_text());key=(record['test_year'],record['arm'],record['seed'])
        if key in observed:raise ValueError('Duplicate fit')
        observed.add(key)
        rp=record['provenance'];mode='development' if key[0] in config['development_years'] else 'confirmation'
        if rp['source_commit']!=prov['source_commit'] or rp['data_sha256']!=prov['data_sha256'][mode]:raise ValueError('Mixed provenance')
        selection=json.loads(p.with_name(p.name.replace('-result.json','-selection.json')).read_text())
        if hashlib.sha256(p.with_name(p.name.replace('-result.json','-selection.json')).read_bytes()).hexdigest()!=record['hashes']['selection.json']:
            raise ValueError('Selection evidence hash changed')
        good=[c for c in selection['candidates'] if c['status']=='success'];candidate_count+=len(selection['candidates'])
        winner=max(good,key=lambda c:c['validation_net_ce'])
        if selection['selected']!=winner:raise ValueError('Wrong hyperparameter winner')
        if key[0] in config['development_years']:validation.setdefault(key[1],[]).append(winner['validation_net_ce'])
    if observed!=expected or len(observed)!=prov['final_fits']:raise ValueError('Incomplete V2 study')
    if candidate_count!=prov['candidate_fits']:raise ValueError('Candidate budget mismatch')
    scores={arm:float(np.mean(validation[arm])) for arm in config['arms']}
    for arm,value in scores.items():
        if not np.isclose(value,seal['selection_scores'][arm]):raise ValueError('Champion score mismatch')
    if max(config['arms'],key=lambda a:scores[a])!=seal['arm']:raise ValueError('Champion selected from different objective')
    sealed_public=root/'champion_pre_confirmation.json'
    if not sealed_public.exists() or json.loads(sealed_public.read_text())!=seal:raise ValueError('Missing or changed pre-confirmation seal')
    for year in config['development_years']+[config['confirmation_year']]:
        factor=json.loads((root/'evidence'/f'{year}-factors.json').read_text())
        for phase in ['training','refit']:
            if len(factor[phase]['ledger'])>24 or len(factor[phase]['selected'])>6:raise ValueError('Factor budget exceeded')
        if pd.Timestamp(factor['train_last_label_exit'])>=pd.Timestamp(factor['validation_first_entry']):raise ValueError('Horizon label leak')
        if pd.Timestamp(factor['validation_last_label_exit'])>=pd.Timestamp(factor['test_first_entry']):raise ValueError('Validation horizon leak')
        if pd.Timestamp(factor['refit_last_label_exit'])>=pd.Timestamp(factor['test_first_entry']):raise ValueError('Refit horizon leak')
    daily=pd.read_csv(root/'daily_paths.csv.gz');table=pd.read_csv(root/'results.csv');checked=0
    grouped={key:g for key,g in daily.groupby(['arm','seed','profile','cost_bps','mode','year'],sort=False)}
    for row in table.to_dict('records'):
        key=(row['arm'],row['seed'],row['profile'],row['cost_bps'],row['mode'])
        data=pd.concat([grouped[key+(year,)] for year in config['development_years']],ignore_index=True) if row['period']=='pooled' else grouped[key+(int(row['period']),)]
        actual=statistics(data.net_return,data.market_return)
        for name in ['cumulative_return','annualized_return','annualized_volatility','sharpe_rf_zero','max_drawdown','capm_alpha_annual_arithmetic','capm_beta']:
            if actual[name] is None:
                if not pd.isna(row[name]):raise ValueError('Undefined metric changed')
            elif not np.isclose(actual[name],row[name],rtol=1e-8,atol=1e-9):raise ValueError('Daily path recomputation mismatch '+name)
        checked+=1
    actual=json.loads((root/'confirmation_test.json').read_text());confirm=daily[(daily['mode']=='confirmation')&(daily.profile==config['primary_profile'])&(daily.cost_bps==config['primary_cost_bps'])]
    champion=confirm[confirm.arm=='champion_ensemble']
    for control in ['csi300','inverse_volatility']:
        other=confirm[confirm.arm==control]
        recomputed=confirmation_test(champion.net_return,other.net_return,config['bootstrap_repetitions'],config['bootstrap_block_length'])
        if not np.allclose(recomputed['paired_block_bonferroni97_5'],actual['tests'][control]['paired_block_bonferroni97_5']):raise ValueError('Confirmation interval mismatch')
    result={'status':'passed','final_fits':len(observed),'candidate_fits':candidate_count,'metric_rows_recomputed':checked,
            'champion':seal['arm'],'source_commit':prov['source_commit']}
    print(json.dumps(result,indent=2));return result


def replay(config_path='configs/v2.json',outputs='outputs/v2',development_data='data',confirmation_data='data/v2'):
    torch.set_num_threads(2);config=json.loads(Path(config_path).read_text());base=Path(outputs);verified=[]
    for mode,data in [('development',development_data),('confirmation',confirmation_data)]:
        x,meta,corr,manifest=samples(data,config);root=base/mode
        years=config['development_years'] if mode=='development' else [config['confirmation_year']]
        for year in years:
            _,val,_,test,_=split(meta,year)
            for arm in config['arms']:
                for seed in seeds(arm,config):
                    directory=root/f'{year}-{arm}-{seed}';result=json.loads((directory/'result.json').read_text())
                    if manifest['quotes_sha256']!=result['provenance']['data_sha256']:raise ValueError('Replay data differs')
                    for name,ids,file in [('validation',val,'validation_predictions.csv'),('final',test,'test_predictions.csv')]:
                        checkpoint=directory/(name+'.pkl')
                        if sha256(checkpoint)!=result['hashes'][name+'.pkl']:raise ValueError('Checkpoint hash mismatch')
                        with checkpoint.open('rb') as f:state=pickle.load(f)
                        raw=state['search'].transform(x[ids],arm);prediction=state['model'].predict(state['scale'].apply(raw))
                        saved=pd.read_csv(directory/file).to_numpy()
                        if not np.allclose(prediction,saved,rtol=1e-8,atol=1e-10):raise ValueError('Checkpoint prediction mismatch')
                        if name=='validation':
                            path=simulate(meta.iloc[ids],prediction,corr,config,config['primary_profile'],config['primary_cost_bps'])
                            ce=certainty_equivalent(path,config['validation_ce_gamma'])
                            if not np.isclose(ce,result['selected_validation_net_ce'],rtol=1e-8,atol=1e-10):raise ValueError('Validation utility mismatch')
                        verified.append({'case':directory.name,'phase':name,'max_absolute_prediction_error':float(np.max(np.abs(prediction-saved)))})
    result={'status':'passed','checkpoints_replayed':len(verified),'cases':verified,'scope':'Hash-verified own local pickle files only; saved-parameter replay, not full retraining.'}
    write_json(base/'replay_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports/v2');p.add_argument('--local-replay',action='store_true');p.add_argument('--outputs',default='outputs/v2');a=p.parse_args()
    replay(outputs=a.outputs) if a.local_replay else audit(a.reports)
