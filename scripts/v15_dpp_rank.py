"""Greedy DPP reorder of the frozen defensive ranking before buffered selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v13_factor_trials import factors
from alphacontextlab.v4.metrics import daily_ic, block_mean_interval
from alphacontextlab.v7.study import ic_summary
from alphacontextlab.v8.study import path_metrics
from alphacontextlab.v10_clock.study import inputs
from alphacontextlab.v10_clock.ledger import simulate, apply_cost


def reorder(score, history):
    # Similarity is a shifted Gram matrix so opposite returns imply low similarity.
    x = np.nan_to_num(history, nan=0.)
    x = x - x.mean(axis=0)
    norms = np.linalg.norm(x, axis=0)
    x /= np.where(norms > 0, norms, 1)
    similarity = .25 * (1 + x.T @ x) + .5 * np.eye(len(score))
    np.fill_diagonal(similarity, 1.)
    residual = np.ones(len(score))
    projection = np.zeros_like(similarity)
    result = np.empty(len(score))
    for step in range(len(score)):
        objective = 4 * score + np.log(np.maximum(residual, 1e-15))
        objective[residual < 0] = -np.inf
        pick = np.argmax(objective)
        result[pick] = len(score) - step
        projection[step] = (similarity[pick] - projection[:step,pick] @ projection[:step]) / np.sqrt(residual[pick])
        residual -= projection[step] ** 2
        residual[pick] = -1
    return result


def run(phase):
    config = json.loads(Path('configs/v10_clock.json').read_text())
    base = json.loads(Path(config['base_config']).read_text())
    base['buffer_exit_fraction'] = .6
    pred,tape,symbols = inputs(phase,config)
    frame = pred.query("method == 'economic_h5'").copy()
    raw,_,low = factors(frame,tape)
    frame['prediction'] = .75*raw + .25*low
    returns = tape.pivot(index='date',columns='symbol',values='close_return').sort_index()
    reference = pd.read_csv(Path('reports/v13_factor_trials')/phase/'daily.csv.gz',parse_dates=['date'])
    reference = reference.query("method == 'defensive'").set_index(['cost_bps','date']).sort_index()
    for date,g in frame.dropna(subset=['prediction']).groupby('date'):
        g = g.sort_values('symbol')
        history = returns.loc[:date,g.symbol].tail(60)
        if len(history)==60 and g.prediction.nunique()>1:
            frame.loc[g.index,'prediction'] = reorder(g.prediction.to_numpy(),history.to_numpy())
    daily = pd.concat([daily_ic(g,g.prediction).assign(fold=f) for f,g in frame.groupby('fold')])
    signal = ic_summary(daily,base)
    gross,_,_,_ = simulate(frame,tape,symbols,base,config,'stagger5')
    paths = pd.concat([apply_cost(gross,c) for c in [0,10,20,30]])
    net = paths.query('cost_bps == 20').set_index('date').net_return
    baseline = reference.loc[20].net_return
    assert net.index.equals(baseline.index)
    signal['paired_mean_daily_bp_ci95'] = block_mean_interval((net-baseline)*10000)
    out = Path('reports/v15_dpp')/phase
    out.mkdir(parents=True,exist_ok=True)
    paths.to_csv(out/'daily.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame([dict(cost=c,**path_metrics(g)) for c,g in paths.groupby('cost_bps')]).to_csv(out/'metrics.csv',index=False)
    (out/'signal.json').write_text(json.dumps(signal,indent=2))


if __name__ == '__main__':
    for phase in ('development','evaluation'):
        run(phase)
