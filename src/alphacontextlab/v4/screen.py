"""OOF factor-return screen; fractional vendor-adjusted proxy, NOT an executable ledger."""
import numpy as np
import pandas as pd


def screen_path(predictions,adjusted,calendar,symbols,config,cost_bps,benchmark=False):
    # A continuous out-of-fold series; no yearly cash resets, all calendar intervals retained.
    scores=predictions.pivot(index='date',columns='symbol',values='prediction').reindex(index=calendar,columns=symbols)
    prices=pd.DataFrame({s:adjusted[s].set_index('date').open for s in symbols}).reindex(calendar)
    observed=prices.notna().to_numpy();mark=prices.ffill().to_numpy()
    dates=sorted(predictions.date.unique());first=calendar.get_loc(dates[0])+1;last=len(calendar)-2
    weights=np.zeros(len(symbols));nav=1.;rows=[];decision_count=0
    for day in range(first,last+1):
        # Instructions depend only on preceding close score. Fractional target weights are
        # a factor-portfolio convention, not a market-on-open quantity/fill guarantee.
        signal=day-1;target=weights.copy();turn=0.;no_candidates=False
        rebalance=decision_count%config['rebalance_sessions']==0
        if rebalance:
            x=scores.iloc[signal];eligible=x.notna().to_numpy();target=np.zeros(len(symbols))
            if eligible.sum()>=config['min_cross_section']:
                ids=np.flatnonzero(eligible)
                if not benchmark:
                    k=max(1,int(np.ceil(eligible.sum()*config['top_fraction'])))
                    ids=ids[np.argsort(-x.to_numpy()[ids],kind='stable')[:k]]
                target[ids]=1/len(ids)
            else:target=weights.copy();no_candidates=True
            turn=float(np.abs(target-weights).sum())
        # Stock suspensions use stale valuation marks; zero-volume/limit execution is NOT
        # claimed. This approximation itself blocks promotion to a live/paper strategy.
        asset_return=mark[day+1]/mark[day]-1
        gross=float(target@asset_return);fee=turn*cost_bps/10000
        # Sell-side tax/commission is included only via an illustrative flat cost stress.
        # Applying costs before the interval makes the self-financing identity exact.
        net=(1-fee)*(1+gross)-1
        nav*=1+net
        weights=target*(1+asset_return)/(1+gross)
        terminal_cost=0.
        if day==last:
            terminal_turn=float(weights.sum());terminal_cost=terminal_turn*cost_bps/10000
            net=(1+net)*(1-terminal_cost)-1;nav*=1-terminal_cost;turn+=terminal_turn;weights[:]=0
        rows.append({'date':str(calendar[day].date()),'signal_date':str(calendar[signal].date()),
                     'exit_date':str(calendar[day+1].date()),'gross_return':gross,'net_return':net,'nav':nav,
                     'turnover':turn,'cost_fraction':fee+terminal_cost,
                     'stale_held_marks':int(((target>0)&(~observed[day]|~observed[day+1])).sum()),
                     'no_candidate_rebalance':no_candidates,'rebalance':rebalance})
        decision_count+=1
    return pd.DataFrame(rows)
