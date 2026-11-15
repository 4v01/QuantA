"""Causal constrained allocation; future open returns are used only for marking."""
import numpy as np
import pandas as pd
from ..v7.screen import choose_target


def make_tape(adj, calendar, symbols, end):
    days = calendar[(calendar >= '2018-01-01') & (calendar <= end)]
    rows = []
    for symbol in symbols:
        d = adj[symbol].set_index('date').reindex(days)
        close = d.close.ffill(); op = d.open.ffill()
        rows.append(pd.DataFrame({'date': days, 'symbol': symbol,
            'close_return': close.pct_change(fill_method=None).to_numpy(),
            'next_open_return': (op.shift(-1) / op - 1).to_numpy(),
            'open_observed': d.open.notna().to_numpy()}))
    return pd.concat(rows, ignore_index=True).sort_values(['date', 'symbol']).reset_index(drop=True)


def risk_state(history, config):
    lookback = config['covariance_sessions']
    if len(history) < max(lookback, config['trend_sessions']):
        return None
    # Missing current/vendor marks are a limitation, never future tradability filters.
    x = np.nan_to_num(np.asarray(history[-lookback:], float), nan=0.)
    cov = np.cov(x, rowvar=False, ddof=1)
    s = config['covariance_diagonal_shrinkage']
    cov = (1-s)*cov + s*np.diag(np.diag(cov)) + np.eye(x.shape[1])*1e-10
    raw = np.asarray(history, float)
    market = np.nansum(raw, axis=1)/np.maximum(np.isfinite(raw).sum(axis=1), 1)
    levels = np.cumprod(1 + np.nan_to_num(market, nan=0.))
    trend = bool(levels[-1] > levels[-config['trend_sessions']:].mean())
    return {'covariance': cov, 'trend': trend,
            'market_expected': float(np.nanmean(market[-lookback:])*5*config['market_mean_shrinkage'])}


def net_edge_target(mu, old, covariance, cap, cost_bps, config):
    if cap <= 0:
        return np.zeros_like(old), {'success': True, 'cash_reason': 'trend_off'}
    mu = np.asarray(mu, float); old = np.asarray(old, float)
    good = np.isfinite(mu); mu = np.where(good, mu, 0.)
    cost = cost_bps/10000; quadratic = covariance*5*config['risk_aversion']
    lipschitz = max(float(np.linalg.eigvalsh(quadratic)[-1]), 1e-10)
    upper = np.where(good, config['maximum_stock_weight'], 0.)

    def prox(v):
        # Exact L1 proximal map with box and budget constraints. The scalar
        # budget multiplier is monotone, so bisection solves the joint prox.
        def at(multiplier):
            delta = v-multiplier-old
            return np.clip(old+np.sign(delta)*np.maximum(np.abs(delta)-cost/lipschitz,0.),0.,upper)
        w = at(0.)
        if w.sum() <= cap: return w
        lo = 0.; hi = max(float(np.max(v))+float(np.max(old))+cost/lipschitz+1.,1.)
        for _ in range(45):
            mid = (lo+hi)/2
            if at(mid).sum()>cap: lo=mid
            else: hi=mid
        return at(hi)

    w = np.minimum(old,upper); w *= min(1.,cap/max(w.sum(),1e-12))
    y = w.copy(); momentum = 1.; residual = np.inf
    # Accelerated proximal gradient solves the same convex objective as the
    # original epigraph formulation, without a generic line search at kinks.
    for iteration in range(1,3001):
        candidate = prox(y-(quadratic@y-mu+cost)/lipschitz)
        next_momentum = (1+np.sqrt(1+4*momentum**2))/2
        extrapolated = candidate+(momentum-1)/next_momentum*(candidate-w)
        if (y-candidate)@(candidate-w)>0:
            next_momentum=1.; extrapolated=candidate.copy()
        w=candidate; y=extrapolated; momentum=next_momentum
        if iteration % 10 == 0:
            mapped = prox(w-(quadratic@w-mu+cost)/lipschitz)
            residual = float(lipschitz*np.max(np.abs(w-mapped)))
            if residual<1e-10: break
    if residual>=1e-10 or w.sum()>cap+1e-8:
        raise RuntimeError(f'Convex allocator did not converge: residual={residual}')
    w[w < 1e-9] = 0.
    objective=float(-mu@w+.5*w@quadratic@w+cost*(np.abs(w-old).sum()+w.sum()))
    return w, {'success': True, 'objective': objective, 'iterations': iteration, 'optimality_residual':residual}


def target_for(policy, scores, excess, old, state, cost, base, config):
    if state is None:
        return np.zeros_like(old), {'cash_reason': 'insufficient_risk_history'}
    eligible = np.isfinite(scores)
    if eligible.sum() < base['min_cross_section'] or np.std(scores[eligible]) < 1e-12:
        return np.zeros_like(old), {'cash_reason': 'missing_or_flat_signal'}
    timing = policy in ['trend_vol_buffer', 'trend_net_edge', 'equal_trend_vol']
    if timing and not state['trend']:
        return np.zeros_like(old), {'cash_reason': 'trend_off'}
    if policy in ['net_edge', 'trend_net_edge']:
        mu = excess + state['market_expected']
        return net_edge_target(mu, old, state['covariance'], 1., cost, config)
    w, missing = choose_target(scores, old, base, buffer=True, benchmark=policy=='equal_trend_vol')
    if missing:
        return np.zeros_like(old), {'cash_reason': 'missing_signal'}
    if policy != 'buffer':
        vol = float(np.sqrt(w@state['covariance']@w*252))
        w *= min(1., config['target_annual_volatility']/max(vol, 1e-12))
    return w, {'success': True}


def simulate(pred, tape, symbols, base, config, policy, cost):
    dates = pd.DatetimeIndex(sorted(tape.date.unique()))
    close = tape.pivot(index='date', columns='symbol', values='close_return').reindex(index=dates, columns=symbols).to_numpy()
    future = tape.pivot(index='date', columns='symbol', values='next_open_return').reindex(index=dates, columns=symbols).to_numpy()
    observed = tape.pivot(index='date', columns='symbol', values='open_observed').reindex(index=dates, columns=symbols).to_numpy(bool)
    scores = pred.pivot(index='date', columns='symbol', values='prediction').reindex(index=dates, columns=symbols).to_numpy()
    excess = pred.pivot(index='date', columns='symbol', values='expected_excess').reindex(index=dates, columns=symbols).to_numpy()
    first = dates.get_loc(pred.date.min())+1; weights = np.zeros(len(symbols)); nav = 1.; paths = []; holdings = []
    for j, day in enumerate(range(first, len(dates)-1)):
        scheduled = j % config['rebalance_sessions'] == 0
        target = weights.copy(); info = {}; turn = 0.
        if scheduled:
            # At next open, only the prior close and its predecessors are known.
            state = risk_state(close[:day], config)
            target, info = target_for(policy, scores[day-1], excess[day-1], weights, state, cost, base, config)
            turn = float(np.abs(target-weights).sum())
        r = np.nan_to_num(future[day], nan=0.); gross = float(target@r); fee = turn*cost/10000
        net = (1-fee)*(1+gross)-1
        weights = target*(1+r)/(1+gross)
        terminal = 0.
        if day == len(dates)-2:
            liquidation = float(weights.sum()); terminal = liquidation*cost/10000
            net = (1+net)*(1-terminal)-1; turn += liquidation; weights[:] = 0.
        nav *= 1+net
        paths.append({'date': dates[day], 'signal_date': dates[day-1], 'exit_date': dates[day+1],
          'gross_return': gross, 'net_return': net, 'nav': nav, 'turnover': turn,
          'cost_fraction': fee+terminal, 'entry_fee': fee, 'terminal_fee': terminal,
          'gross_exposure': float(target.sum()), 'scheduled': scheduled,
          'stale_held_marks': int(((target>0)&(~observed[day]|~observed[day+1])).sum()),
          'cash_reason': info.get('cash_reason', '')})
        for i in np.flatnonzero(target > 0):
            holdings.append({'date': dates[day], 'symbol': symbols[i], 'weight': target[i]})
    return pd.DataFrame(paths), pd.DataFrame(holdings, columns=['date', 'symbol', 'weight'])
