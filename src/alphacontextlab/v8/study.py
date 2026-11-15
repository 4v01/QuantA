"""Frozen two-stage retrospective experiment, with publicly replayable allocation."""
import argparse
import json
import pickle
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.metrics import daily_ic, path_statistics
from ..v7.data import load
from ..v7.study import ic_summary
from .models import fit_fold
from .portfolio import make_tape, risk_state, simulate


def return_metrics(pred):
    d = pred.dropna(subset=['label', 'expected_return'])
    error = d.label.to_numpy()-d.expected_return.to_numpy()
    sse = float(error@error); y = d.label.to_numpy(); sst = float(((y-y.mean())**2).sum())
    zero = float(y@y)
    return {'rows': len(d), 'rmse': float(np.sqrt(sse/len(d))),
            'r2': 1-sse/sst if sst>0 else None, 'r2_vs_zero': 1-sse/zero if zero>0 else None,
            'note': 'R2 on absolute H5 returns; never IC squared. Zero-return baseline is reported separately.'}


def path_metrics(path):
    return {**path_statistics(path), 'mean_exposure': float(path.gross_exposure.mean()),
            'cash_day_fraction': float((path.gross_exposure<1e-8).mean()),
            'noncash_days': int((path.gross_exposure>1e-8).sum()),
            'stale_held_marks': int(path.stale_held_marks.sum())}


def select_policy(summary, config):
    eligible = []
    for policy in config['policies']:
        row = summary[(summary.policy==policy)&(summary.cost_bps==config['primary_cost_bps'])].iloc[0]
        if row.mean_exposure>=config['minimum_development_exposure'] and row.noncash_days>=100 and np.isfinite(row.net_sharpe):
            eligible.append((policy, float(row.net_sharpe)))
    winner = max(eligible, key=lambda x: x[1])[0] if eligible else 'buffer'
    return {'policy': winner, 'eligible_scores': dict(eligible), 'selection_qualified': bool(eligible),
            'criterion': 'Development net Sharpe at 20bp; mean exposure>=0.1 and >=100 noncash days; config order breaks ties.'}


def gate(signal, summary, policy, base, config, qualified):
    row = summary[(summary.policy==policy)&(summary.cost_bps==config['primary_cost_bps'])].iloc[0]
    reasons = []
    if signal['rank_ic'] is None or signal['rank_ic'] < base['minimum_ic']: reasons.append('IC_below_0.02')
    if signal['ic_coverage'] < base['minimum_ic_coverage']: reasons.append('IC_coverage_below_90pct')
    if signal['passing_fold_fraction'] < base['minimum_passing_fold_fraction']: reasons.append('IC_fold_stability_failed')
    if signal['ci95'][0] is None or signal['ci95'][0]<=0: reasons.append('IC_interval_crosses_zero')
    if not np.isfinite(row.net_sharpe) or row.net_sharpe<=config['minimum_net_sharpe']: reasons.append('net_sharpe_not_above_1')
    if not qualified: reasons.append('development_selection_not_qualified')
    return {'policy': policy, 'financial_passed': not reasons, 'reasons': reasons,
            'strict_pit': False, 'paper_admitted': False, 'orders_sent': 0,
            'structural_blocks': ['fixed_survivor_universe', 'financial_vintages_unverified', 'non_executable_adjusted_return_screen']}


def run(phase, out):
    cpath = Path('configs/v8.json'); config = json.loads(cpath.read_text()); base = json.loads(Path(config['base_config']).read_text())
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Output directory must be empty')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip(): raise ValueError('Freeze clean source/config first')
    seal = json.loads(Path('reports/v8/champion_seal.json').read_text()) if phase=='evaluation' else None
    if seal and seal['config_sha256'] != sha256(cpath): raise ValueError('Seal/config mismatch')
    panel, adj, calendar, symbols, manifest, quality = load()
    years = config['development_years'] if phase=='development' else config['evaluation_years']
    end = f'{max(years)}-12-31'; tape = make_tape(adj, calendar, symbols, end)
    source = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    prov = {'source_commit': source, 'phase': phase, 'config_sha256': sha256(cpath),
            'base_config_sha256': sha256(config['base_config']), 'data_manifest_sha256': sha256('data/v7/manifest.json'),
            'price_manifest_sha256': sha256(Path(base['data_prices'])/'manifest.json'),
            'seed': config['seed'], 'versions': {x: version(x) for x in ['numpy','pandas','scipy','scikit-learn','lightgbm']},
            'symbols': symbols, 'strict_pit': False,
            'seal_sha256': sha256('reports/v8/champion_seal.json') if seal else None}
    out.mkdir(parents=True); write_json(out/'provenance.json', prov); write_json(out/'data_manifest.json', manifest)
    write_json(out/'coverage.json', quality); tape.to_csv(out/'market_tape.csv.gz',index=False,float_format='%.17g')
    predictions = []; calibration_rows = []; ics = []
    for year in years:
        for half in [1,2]:
            fold = f'{year}H{half}'; print('FIT',phase,fold,flush=True)
            p, cal, models, selection = fit_fold(panel,year,half,base,config)
            p.loc[p.exit_date>pd.Timestamp(end),['label','target']] = np.nan
            p['fold'] = fold; cal['fold'] = fold
            dst = out/'evidence'/fold; dst.mkdir(parents=True)
            with (dst/'models.pkl').open('wb') as f: pickle.dump(models,f)
            write_json(dst/'selection.json',selection)
            write_json(dst/'checkpoint.json',{'sha256':sha256(dst/'models.pkl'),'provenance':prov})
            predictions.append(p); calibration_rows.append(cal)
            d = daily_ic(p,p.prediction,base['min_cross_section']); d['fold'] = fold; ics.append(d)
    pred = pd.concat(predictions,ignore_index=True)
    # Absolute market return estimate is computed from known closes only.
    dates = pd.DatetimeIndex(sorted(tape.date.unique()))
    close = tape.pivot(index='date',columns='symbol',values='close_return').reindex(index=dates,columns=symbols).to_numpy()
    market_mu = {}; prediction_dates = set(pred.date)
    for i, day in enumerate(dates):
        if day in prediction_dates:
            state = risk_state(close[:i+1],config); market_mu[day] = state['market_expected'] if state else 0.
    pred['expected_return'] = pred.expected_excess + pred.date.map(market_mu)
    pred.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g')
    pd.concat(calibration_rows,ignore_index=True).to_csv(out/'calibration_predictions.csv.gz',index=False,float_format='%.17g')
    ic = pd.concat(ics,ignore_index=True); ic.to_csv(out/'daily_ic.csv',index=False,float_format='%.17g')
    signal = ic_summary(ic,base); write_json(out/'ic_summary.json',signal); write_json(out/'return_metrics.json',return_metrics(pred))
    paths = []; holdings = []; summaries = []
    for policy in config['policies']+[config['benchmark']]:
        for cost in config['cost_bps']:
            print('ALLOCATE',phase,policy,cost,flush=True)
            path, weights = simulate(pred,tape,symbols,base,config,policy,cost)
            path['policy'] = policy; path['cost_bps'] = cost; paths.append(path)
            weights['policy'] = policy; weights['cost_bps'] = cost; holdings.append(weights)
            summaries.append({'policy':policy,'cost_bps':cost,**path_metrics(path)})
    summary = pd.DataFrame(summaries)
    pd.concat(paths,ignore_index=True).to_csv(out/'screen_paths.csv.gz',index=False,float_format='%.17g')
    pd.concat(holdings,ignore_index=True).to_csv(out/'holdings.csv.gz',index=False,float_format='%.17g')
    summary.to_csv(out/'screen_summary.csv',index=False,float_format='%.17g')
    if phase=='development': write_json(out/'champion_candidate.json',select_policy(summary,config))
    else: write_json(out/'financial_gate.json',gate(signal,summary,seal['policy'],base,config,seal['selection_qualified']))
    print('COMPLETE',phase,flush=True)


def publish(out, reports):
    out = Path(out); reports = Path(reports)
    if reports.exists() and any(reports.iterdir()): raise ValueError('Never overwrite published evidence')
    shutil.copytree(out,reports,ignore=shutil.ignore_patterns('*.pkl'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['development','evaluation','publish'])
    p.add_argument('--outputs',default='outputs/v8/development'); p.add_argument('--reports',default='reports/v8/development')
    a=p.parse_args()
    if a.action=='publish': publish(a.outputs,a.reports)
    else: run(a.action,a.outputs)


if __name__=='__main__': main()
