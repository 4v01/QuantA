"""Raw quotes -> causal total-return features; no provider forward-adjustment input."""
from dataclasses import dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..data import FEATURES,sha256
from ..study import write_json


class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.table=[]
        elif tag=='tr' and self.table is not None:self.row=[]
        elif tag in ('td','th') and self.row is not None:self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:self.row.append(''.join(self.cell).strip());self.cell=None
        elif tag=='tr' and self.row is not None:self.table.append(self.row);self.row=None
        elif tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None


def prepare_events(root='data/v3_corporate'):
    root=Path(root);events=[];sources=[]
    for symbol in ['510300','510050','510500','159915']:
        p=root/'raw'/f'{symbol}-fhsp.html';parser=Tables();parser.feed(p.read_text())
        sources.append({'symbol':symbol,'url':f'https://fundf10.eastmoney.com/fhsp_{symbol}.html','sha256':sha256(p)})
        found=False
        for table in parser.tables:
            if not table:continue
            if '权益登记日' in table[0]:
                found=True
                for row in table[1:]:
                    if len(row)!=5:continue
                    if row[1]<'2014-01-01':continue
                    match=re.fullmatch(r'每10份派现金([0-9.]+)元',row[3])
                    if not match:raise ValueError('Unknown dividend format '+str(row))
                    if not row[1]<=row[2]<=row[4]:raise ValueError('Invalid dividend dates')
                    events.append({'kind':'dividend','symbol':symbol,'record_date':row[1],'effective_date':row[2],
                                   'pay_date':row[4],'cash_per_share':float(match.group(1))/10,'source':sources[-1]['url'],
                                   'availability':'exact publication time unavailable; cash recognized at ex-date, not a pre-event forecast feature'})
            if '拆分折算日' in table[0]:
                for row in table[1:]:
                    if len(row)==4 and row[1]>='2014-01-01' and (symbol,row[1]) not in [('510500','2015-04-14'),('510500','2022-08-26')]:
                        raise ValueError('Unverified split; obtain precise primary notice before proceeding')
        if not found:raise ValueError('Dividend table absent '+symbol)
    events.extend([
        {'kind':'split','symbol':'510500','record_date':'2015-04-13','effective_date':'2015-04-15','ratio':.28032483,'rounding':'floor',
         'source':'https://www.sse.com.cn/disclosure/fund/announcement/c/2015-04-15/510500_20150415_2.pdf'},
        {'kind':'split','symbol':'510500','record_date':'2022-08-26','effective_date':'2022-08-29','ratio':1.14539,'rounding':'ceil',
         'source':'https://pdf.dfcfw.com/pdf/H2_AN202208291577764682_1.pdf'}])
    result={'events':sorted(events,key=lambda e:(e['effective_date'],e['symbol'],e['kind'])),'sources':sources,
            'primary_split_notices':json.loads((root/'source_pdfs.json').read_text()) if (root/'source_pdfs.json').exists() else [],
            'limitations':['Current third-party dividend table, not a complete publication-time/vintage archive.',
                           '2026 corporate-event metadata was audited before price holdout; this is retrospective research.',
                           'Split precision/effective trading dates override rounded website labels using primary notices.']}
    write_json(root/'events.json',result);return result


def causal_prices(raw,events):
    """Per-share synthetic total-return OHLC, with events applied only as they occur."""
    n=len(raw);s=np.ones(n);d=np.zeros(n);positions={str(x.date()):j for j,x in enumerate(raw.index)}
    for event in events:
        key=event['effective_date']
        if key not in positions:
            if str(raw.index.min().date())<=key<=str(raw.index.max().date()):raise ValueError('Event on missing market session')
            continue
        j=positions[key]
        if event['kind']=='split':s[j]*=event['ratio']
        else:d[j]+=event['cash_per_share']
    names=['open','close','high','low'];prices=raw[names].to_numpy(dtype=float);values=np.empty_like(prices)
    for j in range(n):
        if j==0:
            values[j]=prices[j]/prices[j,1]
        else:
            values[j]=values[j-1,1]*(s[j]*prices[j]+d[j])/prices[j-1,1]
    return pd.DataFrame(values,index=raw.index,columns=names),s,d


def economic_holding_return(opening,split,events,dates,horizon):
    """Gross per-unit buy-open/exit-open return; cash distributions are not reinvested.

    Entry on the ex-date is not entitled to a dividend registered the previous
    close. Fractional split units here describe per-unit economic labels, while
    the account simulator separately applies the issuer's integer rounding.
    """
    opening=np.asarray(opening,dtype=float);factor=np.cumprod(split);n=len(opening)
    result=np.full(n,np.nan);entry=np.arange(n-horizon);exit=entry+horizon
    wealth=factor[exit]/factor[entry]*opening[exit]
    for event in events:
        if event['kind']!='dividend':continue
        record=int(dates.searchsorted(pd.Timestamp(event['record_date']),side='right')-1)
        ex=int(dates.searchsorted(pd.Timestamp(event['effective_date'])))
        if record<0:continue
        eligible=(entry<=record)&(record<exit)&(ex<=exit)
        wealth[eligible]+=event['cash_per_share']*factor[record]/factor[entry[eligible]]
    result[entry]=wealth/opening[entry]-1
    return result


@dataclass
class Market:
    dates: pd.DatetimeIndex
    symbols: list
    opening: np.ndarray
    closing: np.ndarray
    tradable: np.ndarray
    adv: np.ndarray
    split_ratio: np.ndarray
    split_rounding: dict
    dividends: list
    tr_open: np.ndarray
    tr_close: np.ndarray
    sample_cov: np.ndarray
    lw_cov: np.ndarray
    lw_shrinkage: np.ndarray
    events: list
    gross_open_return: np.ndarray | None = None


def load(quotes_root,events_root,config):
    root=Path(quotes_root);manifest=json.loads((root/'manifest.json').read_text())
    if sha256(root/'quotes.csv')!=manifest['quotes_sha256']:raise ValueError('Quote hash mismatch')
    event_state=json.loads((Path(events_root)/'events.json').read_text());events=event_state['events']
    quotes=pd.read_csv(root/'quotes.csv',dtype={'symbol':str},parse_dates=['date']);quotes=quotes[quotes.adjustment==0]
    dates=pd.DatetimeIndex(sorted(quotes.date.unique()));symbols=config['symbols'];n=len(symbols);frames=[]
    ro=[];rc=[];available=[];amount=[];splits=[];rounding={};dividends=[];to=[];tc=[];economic_daily=[]
    for k,symbol in enumerate(symbols):
        old=quotes[quotes.symbol==symbol].set_index('date').sort_index()
        missing=dates.difference(old.index);allowed=pd.DatetimeIndex(pd.to_datetime(config['known_full_session_halts'].get(symbol,[])))
        allowed=allowed[(allowed>=dates.min())&(allowed<=dates.max())]
        if not missing.equals(allowed):raise ValueError(f'Unverified missing raw quotes {symbol}: {missing}')
        raw=old.reindex(dates).copy();present=raw.close.notna()
        fill=raw.close.ffill()
        for key in ['open','close','high','low']:raw[key]=raw[key].fillna(fill)
        for key in ['volume','amount']:raw[key]=raw[key].fillna(0)
        active=present&~dates.isin(pd.to_datetime(config['known_no_open_execution'].get(symbol,[])))
        mine=[e for e in events if e['symbol']==symbol];tr,s,d=causal_prices(raw,mine)
        mark=tr.open.where(active,tr.close.shift(1))
        f=tr.copy();f['symbol']=symbol;f['session']=np.arange(len(dates));f['volume']=raw.volume
        f['ret1']=f.close.pct_change();f['ret5']=f.close.pct_change(5);f['ret20']=f.close.pct_change(20)
        f['vol20']=f.ret1.rolling(20).std(ddof=0);f['range']=(f.high-f.low)/f.close
        f['overnight']=f.open/f.close.shift(1)-1;f['volume_relative']=np.log((raw.volume+1)/(raw.volume.rolling(20).mean()+1))
        f['entry_date']=dates.to_series(index=dates).shift(-1);f['exit_date']=dates.to_series(index=dates).shift(-2)
        raw_open=raw.open.where(active,raw.close.shift(1)).to_numpy()
        gross_daily=economic_holding_return(raw_open,s,mine,dates,1);economic_daily.append(gross_daily)
        h=config['horizon'];one=pd.Series(gross_daily,index=dates)
        future=pd.concat([one.shift(-j-1) for j in range(h)],axis=1)
        f['label_return']=pd.Series(economic_holding_return(raw_open,s,mine,dates,h),index=dates).shift(-1)
        f['label_downside']=np.sqrt((future.clip(upper=0)**2).sum(1,min_count=h));f['label_exit']=dates.to_series(index=dates).shift(-h-1)
        f['label_valid']=active.shift(-1,fill_value=False)&active.shift(-h-1,fill_value=False)&f.label_return.notna()&f.label_downside.notna()
        f['daily_return']=one.shift(-1);frames.append(f)
        opening=raw.open.where(active,raw.close.shift(1)).to_numpy();ro.append(opening);rc.append(raw.close.to_numpy())
        available.append(active.to_numpy());amount.append(raw.amount.rolling(20,min_periods=1).mean().to_numpy())
        splits.append(s);to.append(mark.to_numpy());tc.append(tr.close.to_numpy())
        for e in mine:
            if str(dates.min().date())<=e['effective_date']<=str(dates.max().date()):
                if e['kind']=='split':rounding[(int(dates.get_loc(e['effective_date'])),k)]=e['rounding']
                else:
                    record=int(dates.searchsorted(pd.Timestamp(e['record_date']),side='right')-1)
                    ex=int(dates.get_loc(e['effective_date']));pay=int(dates.searchsorted(pd.Timestamp(e['pay_date'])))
                    dividends.append({'asset':k,'record':record,'ex':ex,'pay':pay,'cash':e['cash_per_share']})
    market_returns=frames[symbols.index('510300')].ret1
    for f in frames:f['market_ret1']=market_returns
    x=[];rows=[];length=config['sequence_length']
    for f in frames:
        values=f[FEATURES].to_numpy(dtype=np.float32)
        for j in range(max(length-1,config['minimum_signal_session']),len(f)-2):
            block=values[j-length+1:j+1]
            if not np.isfinite(block).all():continue
            row=f.iloc[j];x.append(block)
            row=row[['symbol','session','entry_date','exit_date','label_return','label_downside','label_exit','label_valid','daily_return']].to_dict()
            row['date']=dates[j];rows.append(row)
    meta=pd.DataFrame(rows);order=meta.sort_values(['entry_date','symbol'],kind='stable').index.to_numpy()
    close=np.stack(tc,axis=1);ret=pd.DataFrame(close).pct_change().fillna(0).to_numpy();sample=[];lw=[];intensities=[]
    for j in range(len(dates)):
        history=ret[max(1,j-config['risk_window']+1):j+1]
        if len(history)<2:sample.append(np.eye(n)*1e-6);lw.append(np.eye(n)*1e-6);intensities.append(1.);continue
        model=LedoitWolf().fit(history);sample.append(np.cov(history,rowvar=False,ddof=0)+np.eye(n)*1e-10)
        lw.append(model.covariance_+np.eye(n)*1e-10);intensities.append(float(model.shrinkage_))
    market=Market(dates,symbols,np.stack(ro,1),np.stack(rc,1),np.stack(available,1),np.stack(amount,1),
                  np.stack(splits,1),rounding,dividends,np.stack(to,1),close,np.stack(sample),np.stack(lw),np.array(intensities),events,
                  np.stack(economic_daily,axis=1))
    version={'quotes_sha256':manifest['quotes_sha256'],'corporate_actions_sha256':sha256(Path(events_root)/'events.json'),
             'input_adjustment':'raw_fqt0_only_causal_event_total_return','quote_manifest':manifest,'event_sources':event_state['sources'],
             'primary_split_notices':event_state.get('primary_split_notices',[]),
             'limitations':event_state['limitations']+['Original quote vintages and exact dividend announcement timestamps are not certified.']}
    return np.stack(x)[order],meta.iloc[order].reset_index(drop=True),market,version
