"""Causal constrained allocation; future open returns are used only for marking."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
    n = len(mu); cost = cost_bps/10000
    cov = covariance*5; risk = config['risk_aversion']
    # Epigraph variables u make absolute turnover convex and differentiable.
    def objective(z):
        w, u = z[:n], z[n:]
        return float(-mu@w + .5*risk*(w@cov@w) + cost*(u.sum()+w.sum()))
    def gradient(z):
        return np.r_[-mu + risk*cov@z[:n] + cost, np.full(n, cost)]
    matrix = np.vstack([np.c_[-np.eye(n), np.eye(n)], np.c_[np.eye(n), np.eye(n)]])
    offset = np.r_[old, -old]
    constraints = [
        {'type': 'ineq', 'fun': lambda z: matrix@z+offset, 'jac': lambda z: matrix},
        {'type': 'ineq', 'fun': lambda z: cap-z[:n].sum(), 'jac': lambda z: np.r_[-np.ones(n), np.zeros(n)]}]
    bounds = [(0., config['maximum_stock_weight'] if good[j] else 0.) for j in range(n)] + [(0., None)]*n
    start = np.minimum(old, config['maximum_stock_weight']); start *= min(1., cap/max(start.sum(), 1e-12))
    z0 = np.r_[start, np.abs(start-old)]
    fit = minimize(objective, z0, jac=gradient, bounds=bounds, constraints=constraints,
                   method='SLSQP', options={'ftol': 1e-10, 'maxiter': 200})
    if not fit.success or fit.x[:n].sum() > cap+1e-7:
        raise RuntimeError('Allocation solver failed: '+str(fit.message))
    w = np.clip(fit.x[:n], 0., config['maximum_stock_weight'])
    w[w < 1e-9] = 0.
    # Cash must dominate if every prospective edge is nonpositive after exit fee.
    return w, {'success': True, 'objective': float(fit.fun), 'iterations': int(fit.nit)}


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
