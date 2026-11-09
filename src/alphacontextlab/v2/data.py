"""Causal daily feature windows, ten-session labels, and horizon-aware purges."""
import numpy as np
import pandas as pd

from ..data import FEATURES, panel


FACTOR_NAMES=['momentum20','momentum60','momentum120','reversal5','downside20',
              'drawdown60','price_volume_coherence','volume_surprise','range_compression',
              'trend_efficiency','bar_pressure','overnight_reversal']
THEMES=['trend','trend','trend','reversal','risk','risk','price_volume','price_volume',
        'stability','stability','geometry','reversal']


def samples(root,config):
    frame,manifest=panel(root,config);xs=[];rows=[];h=config['horizon'];length=config['sequence_length']
    symbols=config['symbols'];dates=pd.DatetimeIndex(sorted(frame.date.unique()))
    returns=np.full((len(dates),len(symbols)),np.nan)
    for k,symbol in enumerate(symbols):
        f=frame[frame.symbol==symbol].sort_values('date').reset_index(drop=True)
        r=f.close.pct_change();vol=f.volume
        f['momentum20']=f.close.pct_change(20);f['momentum60']=f.close.pct_change(60);f['momentum120']=f.close.pct_change(120)
        f['reversal5']=-f.close.pct_change(5)
        f['downside20']=-np.sqrt((r.clip(upper=0)**2).rolling(20).mean())
        f['drawdown60']=f.close/f.close.rolling(60).max()-1
        f['price_volume_coherence']=r.rolling(20).corr(np.log1p(vol).diff()).fillna(0.)
        f['volume_surprise']=np.log((vol+1)/(vol.rolling(20).mean()+1))
        f['range_compression']=-((f.high-f.low)/f.close).rolling(20).mean()
        f['trend_efficiency']=f.momentum20/(r.abs().rolling(20).sum()+1e-8)
        f['bar_pressure']=(f.close-f.open)/(f.high-f.low+1e-6)
        f['overnight_reversal']=-f.overnight
        halted=config.get('known_full_session_halts',{}).get(symbol,[])+config.get('known_no_open_execution',{}).get(symbol,[])
        tradable=~f.date.isin(pd.to_datetime(halted))
        mark=f.open.where(tradable,f.close.shift(1));one_day=mark.shift(-1)/mark-1
        future=pd.concat([one_day.shift(-(j+1)) for j in range(h)],axis=1)
        f['label_return']=mark.shift(-h-1)/mark.shift(-1)-1
        f['label_downside']=np.sqrt((future.clip(upper=0)**2).sum(axis=1,min_count=h))
        f['label_exit']=f.date.shift(-h-1)
        f['label_valid']=tradable.shift(-1,fill_value=False)&tradable.shift(-h-1,fill_value=False)&f.label_return.notna()&f.label_downside.notna()
        f['session']=np.arange(len(f));f['momentum60_signal']=f.momentum60
        values=f[FEATURES+FACTOR_NAMES].to_numpy(dtype=np.float32)
        for end in range(length-1,len(f)):
            block=values[end-length+1:end+1];row=f.iloc[end]
            if not np.isfinite(block).all() or not np.isfinite(row.target):continue
            xs.append(block)
            keys=['date','symbol','session','entry_date','exit_date','target','entry_gap','exit_gap','entry_tradable','exit_tradable',
                  'label_return','label_downside','label_exit','label_valid','momentum60_signal','vol20']
            rows.append(row[keys].to_dict())
        returns[:,k]=r.fillna(0).to_numpy()
    # Lagged market-state correlation at signal close, never at the execution close.
    correlations=[]
    for j in range(len(dates)):
        history=returns[max(0,j-59):j+1]
        if len(history)<2:correlations.append(np.eye(len(symbols)));continue
        cov=np.cov(history,rowvar=False);scale=np.sqrt(np.maximum(np.diag(cov),1e-12))
        corr=cov/scale[:,None]/scale[None,:]
        corr=np.nan_to_num(corr,nan=0.);np.fill_diagonal(corr,1.)
        correlations.append(.9*corr+.1*np.eye(len(symbols)))
    meta=pd.DataFrame(rows);order=meta.sort_values(['entry_date','symbol'],kind='stable').index.to_numpy()
    return np.stack(xs)[order],meta.iloc[order].reset_index(drop=True),np.stack(correlations),manifest


def split(meta,year):
    val_start=pd.Timestamp(year-1,1,1);test_start=pd.Timestamp(year,1,1);end=pd.Timestamp(year+1,1,1)
    valid=meta.label_valid.to_numpy(dtype=bool)
    train=np.flatnonzero((meta.label_exit<val_start).to_numpy()&valid)
    validation=np.flatnonzero(((meta.entry_date>=val_start)&(meta.exit_date<test_start)).to_numpy())
    val_labeled=np.flatnonzero(((meta.entry_date>=val_start)&(meta.label_exit<test_start)).to_numpy()&valid)
    test=np.flatnonzero(((meta.entry_date>=test_start)&(meta.exit_date<end)).to_numpy())
    refit=np.flatnonzero((meta.label_exit<test_start).to_numpy()&valid)
    if min(map(len,[train,validation,val_labeled,test,refit]))==0:raise ValueError('Empty V2 partition')
    if not meta.iloc[train].label_exit.max()<meta.iloc[validation].entry_date.min():raise ValueError('Overlapping labels')
    if not meta.iloc[val_labeled].label_exit.max()<meta.iloc[test].entry_date.min():raise ValueError('Validation leakage')
    return train,validation,val_labeled,test,refit
