"""Valuation units and causal long-horizon price features, from unchanged caches."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ..v7.data import load as base_load

FUND=['earnings_yield','book_to_price','roe_annual','cash_earnings','fin_revenue_growth','fin_profit_growth','fin_leverage','financial_age_years']
PRICE=['momentum_12_1','momentum_6_1','low_volatility_60','market_beta_126']


def economic_financial(frame,calendar,raw_close,max_age=550):
    d=frame.copy(); d['period']=pd.to_datetime(d.REPORT_DATE)
    d['available']=pd.concat([pd.to_datetime(d.NOTICE_DATE),pd.to_datetime(d.UPDATE_DATE)],axis=1).max(axis=1).dt.normalize()+pd.Timedelta(days=1)
    d=d[d.available.notna()&d.period.notna()&(d.available>=d.period)].sort_values(['available','period'],kind='stable')
    events=list(d.to_dict('records')); position=0; known={}; rows=[]
    for day in calendar:
        while position<len(events) and events[position]['available']<=day:
            r=events[position]; known[r['period']]=r; position+=1
        result={'date':day,'earnings_yield':np.nan,'book_to_price':np.nan,'roe_annual':np.nan,'cash_earnings':np.nan,'financial_age_years':np.nan}
        if known:
            period=max(known); current=known[period]; price=raw_close.get(day,np.nan)
            if (day-period).days<=max_age and np.isfinite(price) and price>0:
                result['financial_age_years']=(day-period).days/365.25
                def number(row,key):return pd.to_numeric(row.get(key,np.nan),errors='coerce')
                bps=number(current,'BPS'); eps=number(current,'EPSJB')
                result['book_to_price']=bps/price if bps>0 else np.nan
                if period.month==12:ttm=eps
                else:
                    prior_fy=known.get(pd.Timestamp(period.year-1,12,31)); prior_same=known.get(period-pd.DateOffset(years=1))
                    ttm=eps+number(prior_fy,'EPSJB')-number(prior_same,'EPSJB') if prior_fy and prior_same else np.nan
                result['earnings_yield']=ttm/price
                annual=[p for p in known if p.month==12 and (day-p).days<=max_age]
                if annual:result['roe_annual']=number(known[max(annual)],'ROEJQ')
                result['cash_earnings']=number(current,'MGJYXJJE')/eps if eps>0 else np.nan
        rows.append(result)
    return pd.DataFrame(rows).set_index('date').replace([np.inf,-np.inf],np.nan)


def load():
    panel,adj,calendar,symbols,manifest,quality=base_load()
    base=json.loads(Path('configs/v7.json').read_text()); days=pd.DatetimeIndex(sorted(panel.date.unique()))
    close=pd.DataFrame({s:adj[s].set_index('date').close for s in symbols}).reindex(calendar)
    returns=close.pct_change(fill_method=None); market=returns.mean(axis=1); parts=[];revision_delays=[]
    for symbol in symbols:
        raw=pd.read_csv(Path(base['data_prices'])/(symbol+'-raw.csv'),parse_dates=['date']).set_index('date').close
        fin=pd.read_csv(Path(base['data_fundamental'])/(symbol+'-financial.csv'))
        notice=pd.to_datetime(fin.NOTICE_DATE); update=pd.to_datetime(fin.UPDATE_DATE)
        revision_delays.extend((update-notice).dt.days[(notice>='2018-01-01')&(notice<='2024-12-31')].dropna().tolist())
        f=economic_financial(fin,days,raw,base['maximum_fundamental_age_days'])
        c=close[symbol]; ret=returns[symbol]
        f['momentum_12_1']=(c.shift(21)/c.shift(252)-1).reindex(days)
        f['momentum_6_1']=(c.shift(21)/c.shift(126)-1).reindex(days)
        f['low_volatility_60']=-ret.rolling(60,min_periods=60).std().reindex(days)
        f['market_beta_126']=(ret.rolling(126,min_periods=126).cov(market)/market.rolling(126,min_periods=126).var()).reindex(days)
        f['symbol']=symbol; parts.append(f.reset_index())
    extra=pd.concat(parts,ignore_index=True); new=list(dict.fromkeys(FUND[:4]+['financial_age_years']+PRICE))
    missing={x:float(extra[x].isna().mean()) for x in new}
    age=extra.financial_age_years.dropna()*365.25
    for column in new:extra[column]=extra.groupby('date')[column].rank(pct=True)-.5
    panel=panel.merge(extra,on=['date','symbol'],validate='one_to_one')
    quality.update({'economic_feature_missing_fraction':missing,'raw_per_share_valuation_is_a_proxy':True,
                    'split_share_basis_certified':False,'old_input_snapshots_changed':False,
                    'revision_delay_days_median':float(np.median(revision_delays)),
                    'revision_delay_above_180_days_fraction':float(np.mean(np.asarray(revision_delays)>180)),
                    'known_report_age_days_median':float(age.median()),
                    'known_report_age_days_p90':float(age.quantile(.9))})
    return panel,adj,calendar,symbols,manifest,quality
