"""Independent arithmetic and provenance checks for published evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .backtest import statistics


def audit(root='reports/v1'):
    root=Path(root);prov=json.loads((root/'provenance.json').read_text())
    frozen=subprocess.check_output(['git','show',prov['source_commit']+':configs/study.json'])
    if hashlib.sha256(frozen).hexdigest()!=prov['config_sha256']:raise ValueError('Source config mismatch')
    config=json.loads(frozen);expected={(year,arm,seed) for year in config['test_years']
        for arm in ['ridge','lstm','lstm_capacity','lstm_memory']
        for seed in ([config['seeds'][0]] if arm=='ridge' else config['seeds'])}
    observed=set();candidate_count=0
    for path in sorted((root/'evidence').glob('*-result.json')):
        result=json.loads(path.read_text());key=(result['test_year'],result['arm'],result['seed'])
        if key in observed:raise ValueError('Duplicate final fit')
        observed.add(key)
        if result['provenance']!=prov:raise ValueError('Mixed fit provenance')
        selection=json.loads(path.with_name(path.name.replace('-result.json','-selection.json')).read_text())
        candidates=selection['candidates'];candidate_count+=len(candidates)
        winner=min(candidates,key=lambda c:c['validation_rmse'])
        if selection['selected']!=winner:raise ValueError('Non-validation selection')
        split=selection['split']
        if pd.Timestamp(split['train_last_exit'])>=pd.Timestamp(split['validation_first_entry']):raise ValueError('Train/validation label overlap')
        if pd.Timestamp(split['validation_last_exit'])>=pd.Timestamp(split['test_first_entry']):raise ValueError('Validation/test label overlap')
        if result['arm']!='ridge':
            for c in candidates:
                if len(c['history'])>config['max_epochs']:raise ValueError('Exceeded candidate epochs')
                # Replay early stopping, including the declared improvement tolerance.
                best=None;bad=0
                for row in c['history']:
                    if best is None or row['validation_rmse']<best['validation_rmse']-config['min_delta']:
                        best=row;bad=0
                    else:bad+=1
                if c['best']!=best:raise ValueError('Wrong selected epoch')
    if observed!=expected:raise ValueError(f'Incomplete fit manifest: {len(observed)}/{len(expected)}')
    manifest=json.loads((root/'data_manifest.json').read_text())
    if manifest['quotes_sha256']!=prov['data_sha256']:raise ValueError('Data version mismatch')
    daily=pd.read_csv(root/'daily_net_returns.csv');table=pd.read_csv(root/'backtest_results.csv')
    count=0
    for (method,seed),group in daily.groupby(['method','seed'],dropna=False,sort=False):
        select=table[(table.method==method)&(table.cost_bps==config['primary_cost_bps'])]
        select=select[select.seed.isna()] if pd.isna(seed) else select[select.seed==seed]
        for period in ['pooled']+[str(y) for y in config['test_years']]:
            part=group if period=='pooled' else group[group.test_year==int(period)]
            recorded=select[select.period==period].iloc[0]
            actual=statistics(part.net_return,part.market_return)
            for metric in ['cumulative_return','annualized_return','annualized_volatility','sharpe_rf_zero','max_drawdown','capm_alpha_annual_arithmetic','capm_beta','capm_regression_r2']:
                if actual[metric] is None:
                    if not pd.isna(recorded[metric]):raise ValueError('Undefined metric replaced')
                elif not np.isclose(actual[metric],recorded[metric],rtol=1e-8,atol=1e-9):
                    raise ValueError(f'Return recomputation mismatch: {method}/{seed}/{period}/{metric}')
            count+=1
    result={'status':'passed','final_fits':len(observed),'candidate_fits':candidate_count,
            'primary_metric_rows_recomputed':count,'source_commit':prov['source_commit'],
            'limitations':'Checks reported provenance and arithmetic; not source-data certification or replication of stochastic training.'}
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reports',default='reports/v1');args=parser.parse_args();audit(args.reports)
