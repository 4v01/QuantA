"""Per-sleeve entry-anchored exit orders observed at one open, filled at a later open."""
from pathlib import Path
import numpy as np
import pandas as pd
from v17_position_size import portfolio_inputs
from alphacontextlab.v10_clock.ledger import apply_cost
from alphacontextlab.v4.metrics import path_statistics, block_mean_interval


def simulate_stops(names, template, risk, future, observed, loss, profit, trailing_gap=None):
    positions = np.zeros((5,len(future.columns)))
    budgets = np.full(5,.2)
    prices = np.ones(len(future.columns))
    basis = np.ones_like(positions)
    peak = basis.copy()
    pending = np.zeros_like(positions,dtype=int)
    rows = []
    for day in template.itertuples():
        old = positions.copy()
        visible = observed.loc[day.date].to_numpy(dtype=bool)
        fills = np.where(visible, pending, 0)
        # Observe existing holdings before today's scheduled transactions.
        gain = prices / basis - 1
        eligible = (old>0) & visible & (pending==0)
        take_profit = gain >= profit
        if trailing_gap is not None:
            peak = np.where((old>0) & visible, np.maximum(peak,prices), peak)
            take_profit = (peak/basis-1 >= profit) & (prices <= peak*(1-trailing_gap))
        next_order = np.where(eligible & (gain<=loss),-1,np.where(eligible & take_profit,1,0))
        s = day.active_sleeve
        if not day.missing_prediction_rebalance:
            selected = names.loc[day.date].to_numpy()>0
            positions[s] = 0
            if selected.any():
                vol = risk.loc[day.signal_date].to_numpy()[selected]
                weights = np.full(selected.sum(),1/selected.sum())
                if np.isfinite(vol).all() and vol.min()>0:
                    inverse = 1/vol
                    weights = .5*weights + .5*inverse/inverse.sum()
                positions[s,selected] = budgets[s]*weights
        positions[fills!=0] = 0
        new = (positions>0) & (old==0)
        basis = np.where(new,prices,basis)
        peak = np.where(new,prices,peak)
        pending = np.where(positions>0,np.where(pending!=0,pending,next_order),0)
        entry = np.abs((positions-old).sum(axis=0)).sum()
        exposure = positions.sum()
        r = future.loc[day.date].fillna(0).to_numpy()
        pnl = positions@r
        gross = pnl.sum()
        positions *= (1+r)/(1+gross)
        budgets = (budgets+pnl)/(1+gross)
        prices *= 1+r
        assert np.isclose(budgets.sum(),1) and np.all(positions.sum(axis=1)<=budgets+1e-10)
        terminal = positions.sum() if day.date==template.date.iloc[-1] else 0.
        rows.append((day.date,gross,entry,terminal,entry+terminal,exposure,(fills<0).sum(),(fills>0).sum()))
    return pd.DataFrame(rows,columns=['date','gross_return','entry_turnover','terminal_turnover','turnover','exposure','loss_fills','profit_fills'])


def run(phase):
    inputs = portfolio_inputs(phase)
    old = pd.read_csv(Path('reports/v17_position_size')/phase/'daily.csv.gz',parse_dates=['date'])
    metrics,paths = [],[]
    for rule,loss,profit in [('baseline',-np.inf,np.inf),('loss',-.05,np.inf),('profit',-np.inf,.10),('both',-.05,.10)]:
        gross = simulate_stops(*inputs,loss,profit)
        for cost in (0,10,20,30):
            path = apply_cost(gross,cost)
            control = old.query('cost_bps == @cost').set_index('date')
            net = path.set_index('date').net_return
            assert net.index.equals(control.index)
            if rule=='baseline':
                np.testing.assert_allclose(net,control.net_return,atol=1e-12)
            ci = block_mean_interval((net-control.net_return)*10000)
            metrics.append(dict(rule=rule,cost=cost,loss_fills=path.loss_fills.sum(),profit_fills=path.profit_fills.sum(),
                                exposure=path.exposure.mean(),delta_bp_lo=ci[0],delta_bp_hi=ci[1],**path_statistics(path)))
            paths.append(path.assign(rule=rule))
    out = Path('reports/v18_stops')/phase
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.concat(paths).to_csv(out/'daily.csv.gz',index=False,compression={'method':'gzip','mtime':0})


if __name__=='__main__':
    for phase in ('development','evaluation'):
        run(phase)
