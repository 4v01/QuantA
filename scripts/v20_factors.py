"""Incremental factor tests with the V18 fixed profit exit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v18_stops import simulate_stops, portfolio_inputs, apply_cost, path_statistics, block_mean_interval
from alphacontextlab.v4.metrics import daily_ic
from alphacontextlab.v7.study import ic_summary

ROOT = Path('reports/v20_factors')


def run(phase):
    out = ROOT/phase
    out.mkdir(exist_ok=True)
    base = json.loads(Path('configs/v7.json').read_text())
    volume = pd.read_csv(ROOT/'turnover.csv.gz',dtype={'symbol':str},parse_dates=['date']).pivot(
        index='date',columns='symbol',values='turnover')
    old = pd.read_csv(Path('reports/v18_stops')/phase/'daily.csv.gz',parse_dates=['date']).query("rule=='profit'")
    rows, paths, signals = [], [], {}
    for method in ['control','momentum','quiet']:
        def transform(frame,returns):
            strength = np.log1p(returns).shift(21).rolling(105).sum()
            turn = volume.reindex_like(returns)
            quiet = -turn.rolling(5).mean().div(turn.shift(5).rolling(60).mean().replace(0,np.nan))
            if method!='control':
                factor = strength if method=='momentum' else quiet
                values = factor.stack().reindex(pd.MultiIndex.from_frame(frame[['date','symbol']]))
                rank = pd.Series(values.to_numpy(),index=frame.index).groupby(frame.date).rank(pct=True)
                signals[method] = {'factor_coverage':float(rank.notna().mean())}
                scores = {'standalone':rank,'combined':.8*frame.prediction+.2*rank.fillna(.5)}
            else:
                signals[method] = {}
                scores = {'combined':frame.prediction}
            for label,score in scores.items():
                daily = pd.concat([daily_ic(g,score.loc[g.index]).assign(fold=f) for f,g in frame.groupby('fold')])
                signals[method][label] = ic_summary(daily,base)
            return frame.assign(prediction=scores['combined'])
        args = portfolio_inputs(phase,transform)
        gross = simulate_stops(*args,loss=-np.inf,profit=.10)
        for cost in (0,10,20,30):
            p = apply_cost(gross,cost)
            control = old.query('cost_bps==@cost').set_index('date')
            net = p.set_index('date').net_return
            assert net.index.equals(control.index)
            if method=='control':
                np.testing.assert_allclose(net,control.net_return,atol=1e-12)
            lo,hi = block_mean_interval((net-control.net_return)*10000)
            rows.append(dict(method=method,cost=cost,delta_low_bp=lo,delta_high_bp=hi,**path_statistics(p)))
            paths.append(p.assign(method=method))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.concat(paths).to_csv(out/'daily.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (out/'signals.json').write_text(json.dumps(signals,indent=2))


if __name__=='__main__':
    for phase in ('development','evaluation'):
        run(phase)
