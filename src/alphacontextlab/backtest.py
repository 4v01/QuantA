"""Self-financing fractional-notional ETF research simulator, once per open."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def weights_from_scores(scores):
    values=np.asarray(scores,dtype=float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite score')
    out=np.zeros_like(values);best=np.argmax(values,axis=1)
    active=values[np.arange(len(values)),best]>0
    out[np.flatnonzero(active),best[active]]=1.
    return out


def limit_fraction(symbol,date):
    return .20 if symbol=='159915' and pd.Timestamp(date)>=pd.Timestamp('2020-08-24') else .10


def execute(pre_weights,target,buy_allowed,sell_allowed,cost):
    """Pre-trade NAV is one; sell first, scale buys to available cash including costs."""
    pre=np.asarray(pre_weights,dtype=float);target=np.asarray(target,dtype=float)
    if (pre<0).any() or pre.sum()>1+1e-8 or (target<0).any() or target.sum()>1+1e-8:
        raise ValueError('Long-only, unlevered weights required')
    if cost<0 or cost>=1:raise ValueError('Invalid cost')
    holdings=pre.copy();cash=max(0.,1-pre.sum());traded=0.
    sells=np.where(sell_allowed,np.maximum(holdings-target,0),0)
    holdings-=sells;cash+=sells.sum()*(1-cost);traded+=sells.sum()
    buys=np.where(buy_allowed,np.maximum(target-holdings,0),0)
    required=buys.sum()*(1+cost)
    if required>cash and required>0:buys*=cash/required
    holdings+=buys;cash-=buys.sum()*(1+cost);traded+=buys.sum()
    if cash < -1e-9:raise AssertionError('Negative cash')
    return holdings,max(0.,cash),float(traded)


def simulate(meta,scores,symbols,cost_bps,policy='model'):
    """Each input row: close-t signal, open-(t+1) entry, open-(t+2) mark.

    Uses continuous adjusted notionals. No lot sizes, minimum commission, impact
    model or guaranteed auction fills; these remain explicit research limitations.
    """
    data=meta.copy();data['score']=scores
    if data.duplicated(['entry_date','symbol']).any():raise ValueError('Duplicate signal')
    def wide(column):return data.pivot(index='entry_date',columns='symbol',values=column).reindex(columns=symbols)
    forecasts=wide('score');returns=wide('target');gaps=wide('entry_gap')
    if not all(np.isfinite(x.to_numpy()).all() for x in (forecasts,returns,gaps)):
        raise ValueError('Incomplete basket or invalid return')
    dates=forecasts.index
    if policy=='model':targets=weights_from_scores(forecasts.to_numpy())
    elif policy=='cash':targets=np.zeros(forecasts.shape)
    elif policy=='equal_buy_hold':targets=np.tile(1/len(symbols),forecasts.shape)
    elif policy=='csi300_buy_hold':
        targets=np.zeros(forecasts.shape);targets[:,symbols.index('510300')]=1.
    else:raise ValueError(policy)
    fee=cost_bps/10000;pre=np.zeros(len(symbols));records=[]
    for j,date in enumerate(dates):
        if j and pd.Timestamp(data[data.entry_date==dates[j-1]].exit_date.iloc[0])!=pd.Timestamp(date):
            raise ValueError('Non-contiguous holding intervals; simulate each fold independently')
        limits=np.array([limit_fraction(s,date) for s in symbols])
        gap=gaps.iloc[j].to_numpy()
        # A conservative 0.1 percentage-point buffer accounts for price rounding.
        tradable=data[data.entry_date==date].set_index('symbol').reindex(symbols).entry_tradable.to_numpy(dtype=bool)
        buy=(gap<limits-.001)&tradable;sell=(gap>-(limits-.001))&tradable
        target=pre.copy() if j and policy.endswith('buy_hold') else targets[j]
        holdings,cash,traded=execute(pre,target,buy,sell,fee)
        asset_end=holdings*(1+returns.iloc[j].to_numpy())
        nav_end=cash+asset_end.sum()
        if nav_end<=0:raise ValueError('Insolvent simulation')
        liquidation=0.
        if j==len(dates)-1:
            # Every annual fold ends in cash, with explicit terminal transaction cost.
            exit_gaps=data[data.entry_date==date].set_index('symbol').reindex(symbols).exit_gap.to_numpy()
            exit_date=data[data.entry_date==date].exit_date.iloc[0]
            exit_limits=np.array([limit_fraction(s,exit_date) for s in symbols])
            exit_tradable=data[data.entry_date==date].set_index('symbol').reindex(symbols).exit_tradable.to_numpy(dtype=bool)
            if np.any((asset_end>0)&((exit_gaps<=-(exit_limits-.001))|~exit_tradable)):
                raise ValueError('Terminal liquidation blocked; extend simulator rather than invent a fill')
            liquidation=asset_end.sum();nav_end-=fee*liquidation
        records.append({'entry_date':str(pd.Timestamp(date).date()),'exit_date':str(pd.Timestamp(data[data.entry_date==date].exit_date.iloc[0]).date()),
                        'net_return':float(nav_end-1),'turnover':traded+float(liquidation),
                        'cost_return':float(fee*(traded+liquidation)),
                        'blocked_buys':int(np.sum((target>pre+1e-10)&~buy)),
                        'blocked_sells':int(np.sum((target<pre-1e-10)&~sell)),
                        'invested_after_trade':float(holdings.sum()/(cash+holdings.sum()))})
        pre=asset_end/(cash+asset_end.sum())
    return pd.DataFrame(records)


def statistics(returns,market=None,annualization=252):
    r=np.asarray(returns,dtype=float)
    if len(r)<2 or not np.isfinite(r).all():raise ValueError('Invalid return series')
    equity=np.r_[1.,np.cumprod(1+r)];vol=r.std(ddof=1)
    out={'observations':len(r),'cumulative_return':float(equity[-1]-1),
         'annualized_return':float(equity[-1]**(annualization/len(r))-1),
         'annualized_volatility':float(vol*np.sqrt(annualization)),
         'sharpe_rf_zero':float(r.mean()/vol*np.sqrt(annualization)) if vol>0 else None,
         'max_drawdown':float(np.min(equity/np.maximum.accumulate(equity)-1))}
    if market is not None:
        m=np.asarray(market,dtype=float)
        if m.shape!=r.shape or not np.isfinite(m).all():raise ValueError('Market alignment mismatch')
        x=np.column_stack([np.ones(len(m)),m])
        if np.linalg.matrix_rank(x)<2:raise ValueError('Degenerate market benchmark')
        beta=np.linalg.lstsq(x,r,rcond=None)[0];residual=r-x@beta
        bread=np.linalg.inv(x.T@x);z=x*residual[:,None];meat=z.T@z
        lag=min(20,len(r)-1)
        for k in range(1,lag+1):
            cross=z[k:].T@z[:-k];meat+=(1-k/(lag+1))*(cross+cross.T)
        covariance=bread@meat@bread*len(r)/(len(r)-2)
        se=np.sqrt(max(covariance[0,0],0.));alpha=float(beta[0]*annualization)
        denom=((r-r.mean())**2).sum()
        out.update(capm_alpha_annual_arithmetic=alpha,capm_beta=float(beta[1]),
                   capm_alpha_hac95=[float((beta[0]-1.96*se)*annualization),float((beta[0]+1.96*se)*annualization)],
                   capm_alpha_hac_pvalue=float(2*norm.sf(abs(beta[0]/se))) if se>0 else None,
                   capm_regression_r2=float(1-(residual**2).sum()/denom) if denom>0 else None,
                   hac_lags=lag)
    return out


def bootstrap_sharpe(returns,repetitions=1000,block=20,seed=2026):
    r=np.asarray(returns);rng=np.random.default_rng(seed);values=[]
    for _ in range(repetitions):
        starts=rng.integers(0,len(r),size=int(np.ceil(len(r)/block)))
        ids=((starts[:,None]+np.arange(block))%len(r)).reshape(-1)[:len(r)]
        sample=r[ids];std=sample.std(ddof=1)
        if std>0:values.append(float(sample.mean()/std*np.sqrt(252)))
    return np.quantile(values,[.025,.975]).tolist() if values else None


def forecast_metrics(y,prediction):
    y=np.asarray(y);p=np.asarray(prediction);denom=((y-y.mean())**2).sum()
    return {'rmse':float(np.sqrt(np.mean((y-p)**2))),
            'r2':float(1-((y-p)**2).sum()/denom) if denom>0 else None,
            'pearson':float(np.corrcoef(y,p)[0,1]) if np.std(p)>0 and np.std(y)>0 else None}
