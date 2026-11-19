"""One preregistered trial of a profit-activated trailing exit."""
from pathlib import Path

import numpy as np
import pandas as pd

from v18_stops import simulate_stops, portfolio_inputs, apply_cost, path_statistics, block_mean_interval


def run(phase):
    inputs = portfolio_inputs(phase)
    source = Path('reports/v18_stops')/phase/'daily.csv.gz'
    history = pd.read_csv(source,parse_dates=['date'])
    gross = simulate_stops(*inputs,loss=-np.inf,profit=.10,trailing_gap=.05)
    fixed = simulate_stops(*inputs,loss=-np.inf,profit=.10)
    rows, paths = [], []
    for cost in (0,10,20,30):
        p = apply_cost(gross,cost).set_index('date')
        paths.append(p)
        stats = dict(cost=cost,profit_exits=int(p.profit_fills.sum()),**path_statistics(p))
        for rule in ('baseline','profit'):
            control = history.query('rule == @rule and cost_bps == @cost').set_index('date')
            assert p.index.equals(control.index)
            if rule=='profit':
                np.testing.assert_allclose(apply_cost(fixed,cost).net_return,control.net_return,atol=1e-12)
            delta = (p.net_return-control.net_return)*10000
            stats[rule+'_mean_bp'] = delta.mean()
            stats[rule+'_lo'], stats[rule+'_hi'] = block_mean_interval(delta)
        rows.append(stats)
    out = Path('reports/v19_trailing')/phase
    out.mkdir(parents=True,exist_ok=True)
    pd.concat(paths).to_csv(out/'daily.csv.gz',compression={'method':'gzip','mtime':0})
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)


if __name__=='__main__':
    for phase in ('development','evaluation'):
        run(phase)
