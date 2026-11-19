"""Prespecified V11 reranking trials: reversal, defensive tilt, and joint tilt."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphacontextlab.v4.metrics import daily_ic, block_mean_interval
from alphacontextlab.v7.study import ic_summary
from alphacontextlab.v8.study import path_metrics
from alphacontextlab.v10_clock.study import inputs
from alphacontextlab.v10_clock.ledger import apply_cost, simulate

ROOT = Path('reports/v13_factor_trials')
# Fixed rank weights: original score, five-session reversal, 60-session low volatility.
TRIALS = {'v11': (1, 0, 0), 'reversal': (.75, .25, 0),
          'defensive': (.75, 0, .25), 'joint': (.5, .25, .25)}


def factors(p, tape):
    returns = tape.pivot(index='date', columns='symbol', values='close_return').sort_index()
    rank = p.groupby('date').prediction.rank(pct=True)
    ranks = [rank]
    for signal in (-np.log1p(returns).rolling(5).sum(), -returns.rolling(60).std()):
        values = signal.stack().reindex(pd.MultiIndex.from_frame(p[['date','symbol']]))
        factor = pd.Series(values.to_numpy(), index=p.index)
        ranks.append(factor.groupby(p.date).rank(pct=True).fillna(.5))
    return ranks


def run(phase):
    config = json.loads(Path('configs/v10_clock.json').read_text())
    base = json.loads(Path(config['base_config']).read_text())
    base['buffer_exit_fraction'] = .6
    seal = ROOT/'selection.json'
    selected = json.loads(seal.read_text())['method'] if phase == 'evaluation' else None
    pred, tape, symbols = inputs(phase, config)
    p = pred.loc[pred.method == 'economic_h5'].copy()
    ranks = factors(p, tape)
    metrics, paths, signals = [], [], {}
    for name, weights in TRIALS.items():
        frame = p.copy()
        frame['prediction'] = sum(w*r for w,r in zip(weights,ranks))
        daily = pd.concat([daily_ic(g, g.prediction).assign(fold=fold) for fold,g in frame.groupby('fold')])
        signals[name] = ic_summary(daily, base)
        gross, *_ = simulate(frame, tape, symbols, base, config, 'stagger5')
        for cost in [0,10,20,30]:
            path = apply_cost(gross,cost)
            metrics.append(dict(method=name,cost=cost,**path_metrics(path)))
            paths.append(path.assign(method=name))
    out = ROOT/phase
    out.mkdir(exist_ok=True,parents=True)
    m = pd.DataFrame(metrics)
    primary = m.query('cost == 20').set_index('method')
    old = pd.read_csv('reports/v11_retention/metrics.csv').query("phase == @phase and policy == 'wider_exit' and cost == 20").iloc[0]
    np.testing.assert_allclose(primary.loc['v11','net_sharpe'],old.net_sharpe,atol=1e-10)
    if selected is None:
        eligible = [n for n,s in signals.items() if s['rank_ic'] >= .02 and s['passing_fold_fraction'] >= .75 and s['ic_coverage'] >= .9]
        selected = max(eligible,key=lambda n:primary.loc[n,'net_sharpe']) if eligible else 'v11'
        seal.write_text(json.dumps(dict(method=selected,eligible=eligible),indent=2))
    all_paths = pd.concat(paths)
    wide = all_paths.query('cost_bps == 20').pivot(index='date',columns='method',values='net_return')
    for n in signals:
        signals[n]['paired_daily_net_delta_bp_ci95'] = block_mean_interval((wide[n]-wide.v11)*10000)
    (out/'signal.json').write_text(json.dumps(signals,indent=2))
    m.to_csv(out/'summary.csv',index=False)
    all_paths.to_csv(out/'daily.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    print(phase,selected,primary.net_sharpe.to_dict(),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['development','evaluation'])
    run(parser.parse_args().phase)
