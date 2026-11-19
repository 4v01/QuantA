"""Replay V15 names with half equal and half inverse downside-risk weights."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphacontextlab.v10_clock.study import inputs
from alphacontextlab.v10_clock.ledger import simulate, apply_cost
from alphacontextlab.v4.metrics import block_mean_interval, path_statistics
from v13_factor_trials import factors


def replay(names, template, risk, future, tilt):
    positions = np.zeros((5, len(future.columns)))
    budgets = np.full(5, .2)
    rows = []
    for day in template.itertuples():
        old = positions.copy()
        s = day.active_sleeve
        if not day.missing_prediction_rebalance:
            selected = names.loc[day.date].to_numpy() > 0
            positions[s] = 0.
            if selected.any():
                weights = np.full(selected.sum(), 1 / selected.sum())
                vol = risk.loc[day.signal_date].to_numpy()[selected]
                if tilt and np.isfinite(vol).all() and np.min(vol) > 0:
                    inverse = 1 / vol
                    weights = .5 * weights + .5 * inverse / inverse.sum()
                positions[s, selected] = budgets[s] * weights
        entry = np.abs((positions-old).sum(axis=0)).sum()
        r = future.loc[day.date].fillna(0).to_numpy()
        pnl = positions @ r
        gross = pnl.sum()
        exposure = positions.sum()
        positions *= (1+r)/(1+gross)
        budgets = (budgets+pnl)/(1+gross)
        assert np.isclose(budgets.sum(),1) and np.all(positions.sum(axis=1)<=budgets+1e-10)
        exit_turn = positions.sum() if day.date == template.date.iloc[-1] else 0.
        rows.append((day.date, gross, entry, exit_turn, entry+exit_turn, exposure))
    return pd.DataFrame(rows,columns=['date','gross_return','entry_turnover','terminal_turnover','turnover','gross_exposure'])


def portfolio_inputs(phase, transform=None):
    config = json.loads(Path('configs/v10_clock.json').read_text())
    base = json.loads(Path(config['base_config']).read_text())
    base['buffer_exit_fraction'] = .6
    pred,tape,symbols = inputs(phase,config)
    frame = pred.query("method == 'economic_h5'").copy()
    raw,_,_ = factors(frame,tape)
    def matrix(key):
        return tape.pivot(index='date',columns='symbol',values=key).reindex(columns=symbols).sort_index()
    risk = matrix('close_return').clip(upper=0).pow(2).rolling(60).mean().pow(.5)
    vals = -risk.stack().reindex(pd.MultiIndex.from_frame(frame[['date','symbol']])).to_numpy()
    rank = pd.Series(vals,index=frame.index).groupby(frame.date).rank(pct=True)
    frame['prediction'] = .75*raw + .25*rank.fillna(.5)
    if transform is not None:
        frame = transform(frame, matrix('close_return'))
    template,_,_,sleeves = simulate(frame,tape,symbols,base,config,'stagger5')
    active = sleeves.merge(template[['date','active_sleeve']],on='date').query('sleeve == active_sleeve')
    names = active.pivot(index='date',columns='symbol',values='weight').reindex(index=template.date,columns=symbols).fillna(0)
    return names, template, risk, matrix('next_open_return'), matrix('open_observed')


def run(phase):
    names, template, risk, future, _ = portfolio_inputs(phase)
    old = pd.read_csv(Path('reports/v15_downside')/phase/'daily.csv.gz',parse_dates=['date'])
    reference = replay(names,template,risk,future,False)
    candidate = replay(names,template,risk,future,True)
    paths, metrics = [], []
    for cost in (0,10,20,30):
        control = old.query('cost_bps == @cost').set_index('date')
        actual = apply_cost(reference,cost).set_index('date')
        assert actual.index.equals(control.index)
        np.testing.assert_allclose(actual.net_return,control.net_return,atol=1e-12)
        p = apply_cost(candidate,cost)
        lo,hi = block_mean_interval(10000*(p.net_return.to_numpy()-control.net_return.to_numpy()))
        paths.append(p)
        metrics.append(dict(cost=cost,delta_bp_low=lo,delta_bp_high=hi,**path_statistics(p)))
    out = Path('reports/v17_position_size')/phase
    out.mkdir(exist_ok=True,parents=True)
    pd.concat(paths).to_csv(out/'daily.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)


if __name__ == '__main__':
    for phase in ('development','evaluation'):
        run(phase)
