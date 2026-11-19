"""Monthly expanding-to-252-session ridge using matured cross-sectional rank labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v13_factor_trials import factors
from alphacontextlab.v10_clock.study import inputs
from alphacontextlab.v10_clock.ledger import simulate, apply_cost
from alphacontextlab.v7.study import ic_summary
from alphacontextlab.v4.metrics import daily_ic, block_mean_interval
from alphacontextlab.v8.study import path_metrics


def run(phase):
    config = json.loads(Path('configs/v10_clock.json').read_text())
    base = json.loads(Path(config['base_config']).read_text())
    base['buffer_exit_fraction'] = .6
    preds,tape,symbols = inputs(phase,config)
    frame = preds.query("method == 'economic_h5'").copy()
    x = pd.concat(factors(frame,tape),axis=1).to_numpy() - .5
    y = frame.groupby('date').label.rank(pct=True).to_numpy() - .5
    exit_date = pd.to_datetime(frame.exit_date)
    history = []
    predictions = frame.prediction.copy()
    for period, group in frame.groupby(frame.date.dt.to_period('M')):
        cutoff = group.date.min()
        past = (exit_date < cutoff) & (frame.date < cutoff)
        dates = np.sort(frame.loc[past,'date'].unique())[-252:]
        if len(dates) < 60:
            continue
        mask = past & frame.date.isin(dates) & np.isfinite(x).all(axis=1) & np.isfinite(y)
        design = x[mask]
        weights = np.linalg.solve(design.T@design + .01*len(design)*np.eye(3), design.T@y[mask])
        predictions.loc[group.index] = x[frame.index.get_indexer(group.index)]@weights
        history.append([str(period),str(cutoff.date()),str(exit_date[mask].max().date()),len(design),*weights])
    frame['prediction'] = predictions
    gross,*_ = simulate(frame,tape,symbols,base,config,'stagger5')
    daily = pd.concat([daily_ic(g,g.prediction).assign(fold=f) for f,g in frame.groupby('fold')])
    out = Path('reports/v14_rolling_rank')/phase
    out.mkdir(parents=True,exist_ok=True)
    signal = ic_summary(daily,base)
    paths = pd.concat([apply_cost(gross,c) for c in [0,10,20,30]])
    old = pd.read_csv(Path('reports/v13_factor_trials')/phase/'daily.csv.gz')
    old = old.query("method == 'v11' and cost_bps == 20").set_index('date')
    old.index = pd.to_datetime(old.index)
    returns = paths.query('cost_bps == 20').set_index('date').net_return
    assert returns.index.equals(old.index)
    signal['paired_daily_net_delta_bp_ci95'] = block_mean_interval((returns-old.net_return)*10000)
    (out/'signal.json').write_text(json.dumps(signal,indent=2))
    pd.DataFrame(history,columns=['month','cutoff','latest_exit','training_rows','v11','reversal','low_vol']).to_csv(out/'fits.csv',index=False)
    paths.to_csv(out/'daily.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame([dict(cost=c,**path_metrics(g)) for c,g in paths.groupby('cost_bps')]).to_csv(out/'summary.csv',index=False)


if __name__ == '__main__':
    for phase in ('development','evaluation'):
        run(phase)
