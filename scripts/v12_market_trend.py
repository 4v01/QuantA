"""Fixed weekly 126-session absolute momentum overlay on the V11 shadow portfolios."""
from pathlib import Path

import numpy as np
import pandas as pd

from alphacontextlab.v4.metrics import block_mean_interval, path_statistics
from alphacontextlab.v10_clock.ledger import apply_cost


def invested(market):
    # Row t earns open t to open t+1. At t-1 close, row t-2 is known.
    trend = np.log1p(market).shift(2).rolling(126).sum()
    decisions = trend.iloc[::5].dropna().gt(0)
    return decisions.reindex(market.index).ffill().astype('boolean').fillna(True).astype(bool)


def overlay(shadow, active):
    p = shadow.copy()
    previous = active.shift(fill_value=False)
    exposure = p.gross_exposure
    # Stock exposure after the previous open-to-open move, before today's order.
    drifted = ((exposure + p.gross_return) / (1 + p.gross_return)).shift(fill_value=0.)
    p['entry_turnover'] = np.select([active & previous, active, previous],
                                    [p.entry_turnover, exposure, drifted], default=0.)
    p['terminal_turnover'] *= active
    p['turnover'] = p.entry_turnover + p.terminal_turnover
    p['gross_return'] *= active
    p['gross_exposure'] *= active
    return p


def main():
    source = pd.read_csv('reports/v11_retention/daily.csv.gz', float_precision='round_trip')
    target = Path('reports/v12_market_trend')
    target.mkdir(exist_ok=True)
    records, daily = [], []
    for phase, group in source.groupby('phase'):
        market = group.query("cost_bps == 0 and policy == 'equal_weight'").set_index('date').sort_index()
        active = invested(market.gross_return)
        for policy in ['wider_exit', 'equal_weight']:
            shadow = group.query('cost_bps == 0 and policy == @policy').set_index('date').sort_index()
            assert shadow.index.equals(active.index) and shadow.index.is_unique
            for rule, flags in [('always', active | True), ('trend', active)]:
                p = overlay(shadow, flags)
                for cost in (0, 10, 20, 30):
                    net = apply_cost(p, cost)
                    reference = group.query('cost_bps == @cost and policy == @policy').set_index('date').loc[p.index]
                    if rule == 'always':
                        np.testing.assert_allclose(net.net_return, reference.net_return, atol=1e-14)
                    diff = (net.net_return - reference.net_return) * 10000
                    lo, hi = block_mean_interval(diff, seed=17)
                    records.append(dict(phase=phase, policy=policy, rule=rule, cost=cost,
                                        exposure=p.gross_exposure.mean(), delta_daily_bp=diff.mean(),
                                        lower_bp=lo, upper_bp=hi, **path_statistics(net)))
                    daily.append(net[['gross_return','gross_exposure','entry_turnover','terminal_turnover','net_return','nav']].assign(
                        invested=flags, phase=phase, policy=policy, rule=rule, cost=cost))
    pd.DataFrame(records).to_csv(target/'metrics.csv', index=False)
    pd.concat(daily).to_csv(target/'daily.csv.gz', compression={'method':'gzip','mtime':0})


if __name__ == '__main__':
    main()
