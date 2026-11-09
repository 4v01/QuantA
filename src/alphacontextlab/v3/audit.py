"""Public arithmetic/selection audit and optional full local account replay."""
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
from ..v2.data import split
from ..v2.execution import certainty_equivalent
from .data import load
from .ledger import simulate
from .report import case_years
from .statistics import centered_max_check,paired_excess
from .study import policy_grid


def audit(reports='reports/v3'):
    root=Path(reports);prov=json.loads((root/'provenance.json').read_text());seal=json.loads((root/'champion.json').read_text())
    frozen=subprocess.check_output(['git','show',prov['source_commit']+':configs/v3.json']);c=json.loads(frozen)
    if hashlib.sha256(frozen).hexdigest()!=prov['config_sha256']:raise ValueError('V3 config differs from frozen commit')
    public=json.loads((root/'champion_pre_confirmation.json').read_text())
    if public!=seal:raise ValueError('Champion changed after seal')
    acquisition=root/'acquisition_audit.json'
    if acquisition.exists():
        a=json.loads(acquisition.read_text());sealed=subprocess.check_output(['git','show',a['seal_commit']+':reports/v3/champion_pre_confirmation.json'])
        if json.loads(sealed)!=seal or pd.Timestamp(a['seal_utc'])>=pd.Timestamp(a['first_raw_price_file_utc']):raise ValueError('Invalid public seal/download order')
    old=root/'old_reports_integrity.json'
    if old.exists():
        old=json.loads(old.read_text())
        for row in old['files']:
            original=subprocess.check_output(['git','show',old['reference_commit']+':'+row['path']])
            if hashlib.sha256(original).hexdigest()!=row['sha256'] or sha256(row['path'])!=row['sha256']:raise ValueError('Historical reports changed')
    scores={family:[] for family in c['policies']};forecast_count=0;policy_count=0;cases=set()
    for year,mode in case_years(c):
        r=json.loads((root/'evidence'/f'{year}-result.json').read_text());s=json.loads((root/'evidence'/f'{year}-selection.json').read_text())
        if r['provenance']['source_commit']!=prov['source_commit'] or r['provenance']['config_sha256']!=prov['config_sha256'] or r['provenance']['data_sha256']!=prov['quotes_sha256'][mode] or r['provenance']['corporate_actions_sha256']!=prov['corporate_actions_sha256']:raise ValueError('Mixed source/config/quotes/events')
        if sha256(root/'evidence'/f'{year}-selection.json')!=r['hashes']['selection.json']:raise ValueError('Selection evidence changed')
        if s['selected_forecast']!=min(s['forecast_candidates'],key=lambda a:a['validation_joint_rmse']):raise ValueError('Wrong forecast winner')
        forecast_count+=len(s['forecast_candidates']);policy_count+=len(s['policy_candidates'])
        if len(s['policy_candidates'])!=len(c['ridge_candidates'])*len(policy_grid(c)):raise ValueError('Policy search budget differs')
        if pd.Timestamp(s['train_last_label_exit'])>=pd.Timestamp(s['validation_first_entry']) or pd.Timestamp(s['validation_last_label_exit'])>=pd.Timestamp(s['test_first_entry']) or pd.Timestamp(s['refit_last_label_exit'])>=pd.Timestamp(s['test_first_entry']):raise ValueError('Horizon label boundary violation')
        v=pd.read_csv(root/'validation_paths'/f'{year}-validation_paths.csv.gz');benchmark=pd.read_csv(root/'validation_paths'/f'{year}-validation_benchmark.csv')
        matrix=[]
        for record in s['policy_candidates']:
            if record['status']!='success':continue
            path=v[v.case==record['case']]
            ce=certainty_equivalent(path,c['validation_ce_gamma'])
            if not np.isclose(ce,record['validation_net_ce'],rtol=1e-8,atol=1e-10):raise ValueError('Validation utility recomputation failed')
            matrix.append(path.net_return.to_numpy()-benchmark.net_return.to_numpy())
        check=centered_max_check(np.column_stack(matrix),c['bootstrap_repetitions'],c['bootstrap_block_length'])
        if not np.isclose(check['conditional_pvalue'],s['validation_search_check']['conditional_pvalue'],atol=1e-8):raise ValueError('Validation max bootstrap differs')
        for family in c['policies']:
            options=[p for p in s['policy_candidates'] if p['forecast_index']==s['selected_forecast']['index'] and p['family']==family and p['status']=='success']
            selected=max(options,key=lambda p:p['validation_net_ce'])
            if selected!=s['selected_policies'][family]:raise ValueError('Policy selection used another objective')
            if mode=='development':scores[family].append(selected['validation_net_ce'])
        for record in r['scenarios']:cases.add((year,record['family'],record['scenario']))
    expected={(year,family,scenario) for year,_ in case_years(c) for family in list(c['policies'])+c['baselines'] for scenario in c['scenarios']}
    if cases!=expected or len(cases)!=prov['planned_scenarios']:raise ValueError('Missing planned scenarios')
    means={k:float(np.mean(v)) for k,v in scores.items()}
    if max(c['policies'],key=lambda k:means[k])!=seal['family'] or any(not np.isclose(v,seal['selection_scores'][k]) for k,v in means.items()):raise ValueError('Champion not selected from development validation')
    if forecast_count!=len(case_years(c))*len(c['ridge_candidates']) or policy_count!=len(case_years(c))*len(c['ridge_candidates'])*len(policy_grid(c)):
        raise ValueError('Forecast/policy budget differs from frozen plan')
    daily=pd.read_csv(root/'daily_paths.csv.gz');table=pd.read_csv(root/'results.csv')
    if daily.duplicated(['year','family','scenario','entry_date']).any():raise ValueError('Duplicate daily observations')
    grouped={key:g for key,g in daily.groupby(['year','family','scenario'],sort=False)};checked=0
    for row in table.to_dict('records'):
        if row['status']!='success':continue
        part=pd.concat([grouped[y,row['family'],row['scenario']] for y in c['development_years']],ignore_index=True) if row['period']=='pooled' else grouped[int(row['period']),row['family'],row['scenario']]
        actual=statistics(part.net_return,part.market_return)
        for name in ['cumulative_return','annualized_return','annualized_volatility','sharpe_rf_zero','max_drawdown','capm_alpha_annual_arithmetic','capm_beta']:
            if actual[name] is None:
                if not pd.isna(row[name]):raise ValueError('Undefined metric replaced')
            elif not np.isclose(actual[name],row[name],rtol=1e-8,atol=1e-9):raise ValueError('Account metric recomputation differs '+name)
        checked+=1
    conclusion=json.loads((root/'confirmation_test.json').read_text());champ=seal['family'];year=c['confirmation_year'];required=[champ]+c['confirmation_controls']
    if all((year,k,'primary') in grouped for k in required):
        a=grouped[year,champ,'primary'];passed=statistics(a.net_return,a.market_return)['sharpe_rf_zero']
        passed=passed is not None and passed>0
        for control in c['confirmation_controls']:
            b=grouped[year,control,'primary'];test=paired_excess(a.net_return,b.net_return,c['bootstrap_repetitions'],c['bootstrap_block_length'],len(c['confirmation_controls']))
            if not np.allclose(test['bonferroni_interval'],conclusion['tests'][control]['bonferroni_interval'],atol=1e-9):raise ValueError('Confirmation interval changed')
            passed=passed and test['bonferroni_interval'][0]>0
        if conclusion['status']!=('passed' if passed else 'not_passed'):raise ValueError('Confirmation gate changed')
    elif conclusion['status']!='not_evaluable':raise ValueError('Failed scenario hidden from gate')
    result={'status':'passed','forecast_candidates':forecast_count,'policy_validation_candidates':policy_count,
            'planned_scenarios':len(cases),'metric_rows_recomputed':checked,'source_commit':prov['source_commit'],'champion':champ}
    print(json.dumps(result,indent=2));return result


def replay(outputs='outputs/v3',config_path='configs/v3.json'):
    c=json.loads(Path(config_path).read_text());base=Path(outputs);torch.set_num_threads(2);checks=[];models=0;validation_checked=0
    for mode,quotes in [('development','data/v2'),('confirmation','data/v3_prices')]:
        x,meta,market,version=load(quotes,'data/v3_corporate',c)
        years=c['development_years'] if mode=='development' else [c['confirmation_year']]
        for year in years:
            folder=base/mode/str(year);record=json.loads((folder/'result.json').read_text());_,val,_,test,_=split(meta,year)
            if version['quotes_sha256']!=record['provenance']['data_sha256'] or version['corporate_actions_sha256']!=record['provenance']['corporate_actions_sha256']:raise ValueError('Replay quote/event version differs')
            model_file=folder/'final.pkl'
            if sha256(model_file)!=record['hashes']['final.pkl']:raise ValueError('Own checkpoint hash differs')
            with model_file.open('rb') as f:state=pickle.load(f)
            prediction=state['model'].predict(state['scale'].apply(x[test]));stored=pd.read_csv(folder/'predictions.csv').to_numpy()
            if not np.allclose(prediction,stored,rtol=1e-8,atol=1e-10):raise ValueError('Forecast replay differs')
            models+=1
            if sha256(folder/'validation.pkl')!=record['hashes']['validation.pkl']:raise ValueError('Validation checkpoint hash differs')
            with (folder/'validation.pkl').open('rb') as f:validation=pickle.load(f)
            vpred=validation['model'].predict(validation['scale'].apply(x[val]))
            if not np.isfinite(vpred).all():raise ValueError('Invalid validation checkpoint')
            selection=json.loads((folder/'selection.json').read_text());vpaths=pd.read_csv(folder/'validation_paths.csv.gz')
            for candidate in selection['policy_candidates']:
                if candidate['forecast_index']!=selection['selected_forecast']['index'] or candidate['status']!='success':continue
                path,_=simulate(meta.iloc[val],vpred,market,candidate['family'],candidate['gamma'],c['scenarios']['primary'],c)
                old=vpaths[vpaths.case==candidate['case']]
                if not np.allclose(path.net_return,old.net_return,rtol=1e-8,atol=1e-10):raise ValueError('Validation account replay differs')
                validation_checked+=1
            models+=1
            for scenario in record['scenarios']:
                if scenario['status']!='success':continue
                family,name=scenario['family'],scenario['scenario'];gamma=record['selected_policies'][family]['gamma'] if family in record['selected_policies'] else None
                path,orders=simulate(meta.iloc[test],prediction,market,family,gamma,c['scenarios'][name],c)
                saved=pd.read_csv(folder/family/f'{name}-path.csv')
                if not np.allclose(path.net_return,saved.net_return,rtol=1e-8,atol=1e-10) or not np.allclose(path.nav_end,saved.nav_end,rtol=1e-10,atol=1e-5):raise ValueError('Cash/share replay differs')
                checks.append({'year':year,'family':family,'scenario':name,'maximum_return_error':float(np.max(np.abs(path.net_return-saved.net_return)))})
    result={'status':'passed','checkpoints_loaded':models,'validation_paths_replayed':validation_checked,'account_paths_replayed':len(checks),'cases':checks,
            'scope':'Own hash-checked saved models and regenerated account paths; not independent proof of market assumptions.'}
    write_json(base/'replay_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports/v3');p.add_argument('--local-replay',action='store_true');p.add_argument('--outputs',default='outputs/v3');a=p.parse_args()
    replay(a.outputs) if a.local_replay else audit(a.reports)
