"""AKShare adapter, immutable normalized snapshots and causal dimensionless factors."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json

RENAME={'日期':'date','股票代码':'symbol','开盘':'open','收盘':'close','最高':'high','最低':'low',
        '成交量':'volume','成交额':'amount','换手率':'turnover'}
FEATURES=['ret1','ret5','ret20','ret60','reversal5','vol5','vol20','vol60','downside20',
          'range1','range20','gap1','intraday1','close_location','volume_ratio5','volume_ratio20',
          'amount_ratio20','turnover1','turnover20','volume_vol20','price_volume_corr20',
          'amihud20','ma_gap5','ma_gap20','ma_gap60','trend_strength20','volume_trend20']

def utcnow():return datetime.now(timezone.utc).isoformat()

def normalize(frame,symbol):
    d=frame.rename(columns=RENAME).copy();d['symbol']=symbol
    names=['date','symbol','open','close','high','low','volume','amount','turnover']
    if not set(names)<=set(d):raise ValueError('Provider schema changed')
    d=d[names];d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
    for c in names[2:]:d[c]=pd.to_numeric(d[c],errors='raise')
    if d.date.duplicated().any() or not d.date.is_monotonic_increasing:raise ValueError('Duplicate/unsorted dates')
    p=d[['open','close','high','low']]
    if not np.isfinite(p).all().all() or (p<=0).any().any():raise ValueError('Invalid prices')
    if ((d.high+1e-7<d[['open','close','low']].max(axis=1))|(d.low-1e-7>d[['open','close','high']].min(axis=1))).any():raise ValueError('OHLC inconsistency')
    if (d[['volume','amount','turnover']]<0).any().any():raise ValueError('Negative volume/turnover')
    return d

def fetch(config,root):
    import akshare as ak
    root=Path(root);root.mkdir(parents=True,exist_ok=True);files=[];failures=[]
    # Use ONE provider/adjustment convention; never silently substitute a different vendor.
    for symbol in config['symbols']:
        for adjust in ['raw','hfq']:
            p=root/f'{symbol}-{adjust}.csv';side=p.with_suffix('.json')
            if p.exists():
                meta=json.loads(side.read_text())
                if meta['sha256']!=sha256(p) or meta['start']!=config['start'] or meta['end']!=config['end']:raise ValueError('Immutable cache mismatch')
                files.append(meta);continue
            try:
                d=ak.stock_zh_a_hist(symbol=symbol,period='daily',start_date=config['start'].replace('-',''),
                                   end_date=config['end'].replace('-',''),adjust='' if adjust=='raw' else 'hfq')
                d=normalize(d,symbol)
                if len(d)<1500:raise ValueError('Insufficient history; no replacement allowed')
                d.to_csv(p,index=False,float_format='%.12g')
                meta={'file':p.name,'symbol':symbol,'adjust':adjust,'rows':len(d),'sha256':sha256(p),
                      'first':str(d.date.min()),'last':str(d.date.max()),'start':config['start'],'end':config['end'],
                      'fetched_at':utcnow(),'akshare':ak.__version__,'api':'stock_zh_a_hist','provider':'Eastmoney'}
                write_json(side,meta);files.append(meta);print('fetched',symbol,adjust,len(d),flush=True)
            except Exception as e:
                failures.append({'symbol':symbol,'adjust':adjust,'error':repr(e)});print('FAILED',symbol,adjust,repr(e),flush=True)
    write_json(root/'failures.json',failures)
    if failures:raise RuntimeError('Incomplete universe; inspect failures, do not drop/replace symbols')
    write_json(root/'manifest.json',{'created_at':utcnow(),'akshare':version('akshare'),'files':files,
              'limitations':['Current vendor hfq, no vintage/PIT certification','Fixed survivor sample, no historic membership',
                             'No certification of hfq as realizable dividend cash return']})

def symbol_factors(raw,adjusted,calendar,horizon):
    r=raw.set_index('date').reindex(calendar);a=adjusted.set_index('date').reindex(calendar)
    # Missing sessions remain in the calendar. No implicit pct_change forward-fill.
    c=a.close;o=a.open;ret=c.pct_change(fill_method=None);vol=r.volume.where(r.volume>0)
    f=pd.DataFrame(index=calendar)
    for h in [1,5,20,60]:f[f'ret{h}']=c/c.shift(h)-1
    f['reversal5']=-f.ret5
    for h in [5,20,60]:f[f'vol{h}']=ret.rolling(h,min_periods=h).std()
    f['downside20']=ret.clip(upper=0).pow(2).rolling(20).mean().pow(.5)
    f['range1']=(a.high-a.low)/c;f['range20']=f.range1.rolling(20).mean()
    f['gap1']=o/c.shift(1)-1;f['intraday1']=c/o-1
    f['close_location']=(c-a.low)/(a.high-a.low).replace(0,np.nan)-.5
    for h in [5,20]:f[f'volume_ratio{h}']=vol/vol.rolling(h).mean()-1
    f['amount_ratio20']=r.amount/r.amount.rolling(20).mean()-1
    f['turnover1']=r.turnover/100;f['turnover20']=f.turnover1.rolling(20).mean()
    f['volume_vol20']=vol.pct_change(fill_method=None).rolling(20).std()
    f['price_volume_corr20']=ret.rolling(20).corr(vol.pct_change(fill_method=None))
    f['amihud20']=(ret.abs()/r.amount.where(r.amount>0)*1e8).rolling(20).mean()
    for h in [5,20,60]:f[f'ma_gap{h}']=c/c.rolling(h).mean()-1
    f['trend_strength20']=f.ret20/(f.vol20*np.sqrt(20)+1e-8)
    f['volume_trend20']=vol.rolling(5).mean()/vol.rolling(20).mean()-1
    # Decision at close t; enter t+1 open and exit t+H+1 open.
    f['label']=o.shift(-horizon-1)/o.shift(-1)-1
    f['exit_date']=pd.Series(calendar,index=calendar).shift(-horizon-1)
    f['date']=calendar;f['symbol']=str(raw.symbol.iloc[0])
    f['known_tradable']=(r.volume>0)&r.open.notna()
    return f.replace([np.inf,-np.inf],np.nan)

def cross_rank(frame,columns):
    d=frame.copy()
    # Same-close cross-sectional information, no future date statistics.
    for c in columns:d[c]=d.groupby('date')[c].rank(pct=True,method='average')-.5
    return d

def load(config,root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());raws={};adjs={}
    for item in manifest['files']:
        p=root/item['file']
        if sha256(p)!=item['sha256']:raise ValueError('Data version mismatch')
        d=pd.read_csv(p,dtype={'symbol':str},parse_dates=['date']);(raws if item['adjust']=='raw' else adjs)[item['symbol']]=d
    if sorted(raws)!=sorted(config['symbols']) or sorted(adjs)!=sorted(config['symbols']):raise ValueError('Incomplete universe')
    calendar=pd.DatetimeIndex(sorted(set().union(*(set(d.date) for d in raws.values()))))
    parts=[symbol_factors(raws[s],adjs[s],calendar,config['horizon']) for s in config['symbols']]
    panel=pd.concat(parts,ignore_index=True).sort_values(['date','symbol']).reset_index(drop=True)
    panel=cross_rank(panel,FEATURES);panel['target']=panel.groupby('date').label.rank(pct=True)-.5
    # Require all factors, preserve missing-row accounting separately.
    valid=panel[FEATURES].notna().all(axis=1)&panel.known_tradable
    quality={'calendar_sessions':len(calendar),'raw_rows':sum(len(d) for d in raws.values()),
             'symbols':len(raws),'panel_rows':len(panel),'factor_eligible_rows':int(valid.sum()),
             'missing_factor_or_nontrading_rows':int((~valid).sum()),'all_raw_ohlc_validated':True,
             'adjusted_return_proxy':True,'pit_universe_certified':False,'cash_event_ledger_certified':False}
    panel['eligible']=valid
    return panel,raws,adjs,calendar,manifest,quality
