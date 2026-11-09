"""Close-fixed integer orders, event entitlements, actual cash and delayed fills."""
from decimal import Decimal,ROUND_HALF_UP
import math

import numpy as np
import pandas as pd

from ..backtest import limit_fraction
from .portfolio import weights


def money(value):return float(Decimal(str(float(value))).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))


def price_limit(reference,fraction,tick=.001):
    unit=Decimal(str(tick));p=Decimal(str(float(reference)))*(Decimal(1)+Decimal(str(fraction)))
    return float((p/unit).quantize(Decimal('1'),rounding=ROUND_HALF_UP)*unit)


def plan_orders(shares,prices,target,nav,lot=100):
    if np.allclose(target,shares*prices/nav,rtol=0,atol=1e-11):return np.zeros(len(shares),dtype=np.int64)
    desired=np.floor(np.maximum(target,0)*nav/prices/lot+1e-9).astype(np.int64)*lot
    orders=np.zeros(len(shares),dtype=np.int64)
    for k in range(len(shares)):
        if desired[k]>=shares[k]:orders[k]=((desired[k]-shares[k])//lot)*lot
        elif target[k]<1e-12:orders[k]=-shares[k]  # sell entire odd-lot remainder only at full exit
        else:orders[k]=-((shares[k]-desired[k])//lot)*lot
    return orders


def trade(shares,cash,requested,opening,reference,tradable,split_today,adv,scenario,config):
    q=shares.copy();orders=requested.copy();lot=config['lot_size'];tick=config['tick_size'];n=len(q)
    rate=scenario['commission_bps']/10000;slip=scenario['slippage_bps']/10000
    limits=np.array([limit_fraction(s,config['_execution_date']) for s in config['symbols']])
    upper=np.array([price_limit(reference[k],limits[k],tick) for k in range(n)])
    lower=np.array([price_limit(reference[k],-limits[k],tick) for k in range(n)])
    px=np.where(orders>=0,np.ceil(opening*(1+slip)/tick-1e-9)*tick,np.floor(opening*(1-slip)/tick+1e-9)*tick)
    blocked=(~tradable)|split_today|((orders>0)&((opening>=upper-1e-9)|(px>upper+1e-9)))|((orders<0)&((opening<=lower+1e-9)|(px<lower-1e-9)))
    orders[blocked]=0;cap=np.floor(config['adv_participation']*adv/np.maximum(px,1e-8)/lot).astype(np.int64)*lot
    original=orders.copy()
    for k in range(n):
        if orders[k]>cap[k]:orders[k]=cap[k]
        if -orders[k]>cap[k]:orders[k]=-cap[k]
        if orders[k]<0:orders[k]=-min(int(q[k]),int(-orders[k]))
    commission=lambda qty,price:money(max(scenario['min_commission'],money(qty*price)*rate)) if qty else 0.
    fills=np.zeros(n,dtype=np.int64);fees=0.;slippage=0.;notional=0.;rounding=0.;asset_fees=np.zeros(n)
    for k in range(n):
        if orders[k]>=0:continue
        qty=int(-orders[k]);amount=money(qty*px[k]);fee=commission(qty,px[k]);cash+=amount-fee;q[k]-=qty;fills[k]=-qty
        fees+=fee;asset_fees[k]=fee;slippage+=qty*(opening[k]-px[k]);notional+=qty*opening[k];rounding+=amount-qty*px[k]
    buys=np.maximum(orders,0);cost=sum(money(buys[k]*px[k])+commission(int(buys[k]),px[k]) for k in range(n))
    if cost>cash and cost>0:buys=np.floor(buys*(cash/cost)/lot).astype(np.int64)*lot
    def total():return sum(money(buys[k]*px[k])+commission(int(buys[k]),px[k]) for k in range(n))
    while total()>cash+1e-8 and buys.sum()>0:
        k=int(np.argmax(buys*px));buys[k]=max(0,buys[k]-lot)
    for k in range(n):
        qty=int(buys[k])
        if not qty:continue
        amount=money(qty*px[k]);fee=commission(qty,px[k]);cash-=amount+fee;q[k]+=qty;fills[k]=qty
        fees+=fee;asset_fees[k]=fee;slippage+=qty*(px[k]-opening[k]);notional+=qty*opening[k];rounding-=amount-qty*px[k]
    if cash< -1e-7 or (q<0).any():raise ValueError('Cash/position conservation failed')
    return q,max(0.,cash),{'fills':fills,'commission':fees,'asset_commissions':asset_fees,'execution_prices':px,
           'slippage':float(slippage),'notional':float(notional),
           'rounding_pnl':float(rounding),'blocked_orders':int(np.sum(blocked&(requested!=0))),
           'liquidity_capped_orders':int(np.sum(original!=orders))}


def simulate(meta,prediction,market,policy,gamma,scenario,config):
    symbols=config['symbols'];n=len(symbols);dates=pd.DatetimeIndex(sorted(meta.entry_date.unique()))
    order=pd.MultiIndex.from_product([dates,symbols]);frame=meta.copy();frame['mu']=prediction[:,0];frame['downside']=np.maximum(prediction[:,1],0)
    indexed=frame.set_index(['entry_date','symbol']).reindex(order)
    if len(frame)!=len(dates)*n or not np.isfinite(indexed[['mu','downside']].to_numpy()).all():raise ValueError('Incomplete model/calendar')
    mu=indexed.mu.to_numpy().reshape(-1,n);down=indexed.downside.to_numpy().reshape(-1,n)
    signals=indexed.session.to_numpy().reshape(-1,n)[:,0].astype(int);entries=signals+1
    if not np.all(np.diff(entries)==1):raise ValueError('Daily calendar was silently removed')
    capital=float(scenario['capital']);cash=capital;shares=np.zeros(n,dtype=np.int64);claims=[];entitlements={}
    records=[];all_trades=[];previous_nav=None;previous_record=None;solver_failures=0
    def claims_value():return float(sum(c['amount'] for c in claims))
    def capture_record(day):
        for j,event in enumerate(market.dividends):
            if event['record']==day:entitlements[j]=int(shares[event['asset']])
    def plan(signal,j,terminal=False):
        nonlocal solver_failures
        close=market.closing[signal];nav=float(cash+claims_value()+shares@close);pre=shares*close/nav
        liquidation=terminal or j>=len(dates)-config['liquidation_sessions']
        rebalance=j==0 or signal%config['rebalance']==0
        if liquidation:target=np.zeros(n)
        elif not rebalance or (policy=='csi300' and j>0):return np.zeros(n,dtype=np.int64),False
        else:
            target,status=weights(mu[j],down[j],market,signal,pre,policy,gamma,config)
            if not status['success']:solver_failures+=1
        return plan_orders(shares,close,target,nav,config['lot_size']),True
    capture_record(entries[0]-1);requested,scheduled=plan(signals[0],0)
    # One extra valuation/execution point is required for the last holding interval.
    for step,day in enumerate(np.r_[entries,entries[-1]+1]):
        split_today=np.abs(market.split_ratio[day]-1)>1e-12
        for k in np.flatnonzero(split_today):
            value=shares[k]*market.split_ratio[day,k];rule=market.split_rounding[day,k]
            shares[k]=math.ceil(value-1e-10) if rule=='ceil' else math.floor(value+1e-10)
        for j,event in enumerate(market.dividends):
            if event['ex']==day and entitlements.get(j,0):claims.append({'pay':event['pay'],'amount':money(entitlements[j]*event['cash'])})
        paid=sum(c['amount'] for c in claims if c['pay']<=day);cash+=paid;claims[:]=[c for c in claims if c['pay']>day]
        opening=market.opening[day];nav_before=float(cash+claims_value()+shares@opening)
        terminal=step==len(entries)
        if previous_record is not None and not terminal:
            previous_record['nav_end']=nav_before;previous_record['net_return']=nav_before/previous_nav-1;records.append(previous_record)
        event_cash=np.array([sum(e['cash'] for e in market.dividends if e['asset']==k and e['ex']==day) for k in range(n)])
        reference=(market.closing[day-1]-event_cash)/market.split_ratio[day]
        local=dict(config);local['_execution_date']=str(market.dates[day].date())
        shares,cash,execution=trade(shares,cash,requested,opening,reference,market.tradable[day],split_today,
                                   market.adv[day-1],scenario,local)
        after=float(cash+claims_value()+shares@opening)
        if not np.isclose(after-nav_before,-execution['commission']-execution['slippage']+execution['rounding_pnl'],atol=1e-6):
            raise ValueError('Trade self-financing identity violated')
        for k in range(n):
            if requested[k] or execution['fills'][k]:all_trades.append({'date':str(market.dates[day].date()),'symbol':symbols[k],
                'requested':int(requested[k]),'filled':int(execution['fills'][k]),'shares_after':int(shares[k]),
                'signal_date':str(market.dates[day-1].date()),'cash_after_all_trades':cash,
                'execution_price':float(execution['execution_prices'][k]) if execution['fills'][k] else None,
                'commission':float(execution['asset_commissions'][k]),'opening_mark':float(opening[k])})
        if terminal:
            if shares.sum()!=0:raise ValueError('Could not liquidate before boundary; retain failed case, never invent a fill')
            previous_record['nav_end']=after;previous_record['net_return']=after/previous_nav-1
            previous_record['terminal_commission']=execution['commission'];previous_record['terminal_slippage']=execution['slippage']
            previous_record['turnover']+=execution['notional']/previous_nav;records.append(previous_record);break
        record={'entry_date':str(market.dates[day].date()),'exit_date':str(market.dates[day+1].date()),'nav_start':nav_before,
                'cash_after':cash,'receivable_after':claims_value(),'gross_exposure':float(shares@opening/after),
                'commission':execution['commission'],'slippage':execution['slippage'],'turnover':execution['notional']/nav_before,
                'blocked_orders':execution['blocked_orders'],'liquidity_capped_orders':execution['liquidity_capped_orders'],
                'scheduled':scheduled,'solver_failures_cumulative':solver_failures,
                'market_return':float(market.gross_open_return[day,symbols.index('510300')]) if market.gross_open_return is not None else float(market.tr_open[day+1,symbols.index('510300')]/market.tr_open[day,symbols.index('510300')]-1),
                'terminal_commission':0.,'terminal_slippage':0.}
        previous_record=record;previous_nav=nav_before;capture_record(day)
        if step+1<len(entries):requested,scheduled=plan(signals[step+1],step+1)
        else:requested,scheduled=plan(day,len(entries),True)
    return pd.DataFrame(records),pd.DataFrame(all_trades)
