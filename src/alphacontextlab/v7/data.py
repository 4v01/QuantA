import json,re,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.data import load as price_load,FEATURES
from ..v6.data import FINANCIAL,NEWS_FEATURES,news_features,financial_features

ALIGNED=['aligned_roe','aligned_eps','fin_revenue_growth','fin_profit_growth','fin_leverage','fin_bps']+[f'change63_{x}' for x in FINANCIAL]+['report_age']
CONTEXT=['market_ret5','market_vol20','ret5_x_market','ret20_x_market','vol20_x_market']


def read_title_archive(path,symbols,minimum_date='2022-01-01',maximum_date='2024-12-31'):
    rows=[];counts={'records':0,'missing_title':0,'duplicate_titles':0};found=set();seen=set()
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            m=re.search(r'/(\d{6})\.',name)
            if not m or m.group(1) not in symbols or not name.endswith('.jsonl'):continue
            symbol=m.group(1);found.add(symbol)
            for line in z.read(name).decode('utf-8-sig').splitlines():
                if not line.strip():continue
                record=json.loads(line);counts['records']+=1
                title=record.get('title')
                if not isinstance(title,str) or not title.strip():counts['missing_title']+=1;continue
                day=pd.Timestamp(record['datetime']).normalize()
                if not pd.Timestamp(minimum_date)<=day<=pd.Timestamp(maximum_date):continue
                key=(symbol,str(day.date()),title.strip())
                if key in seen:counts['duplicate_titles']+=1;continue
                seen.add(key);rows.append({'symbol':symbol,'published_date':day,'research_available_date':day+pd.Timedelta(days=1),'title':title.strip(),'publisher':'FinMultiTime_title_field'})
    if found!=set(symbols):raise ValueError('Archive symbol coverage differs; no replacements allowed')
    return pd.DataFrame(rows),counts


def prepare(config_path='configs/v7.json',root='data/v7'):
    c=json.loads(Path(config_path).read_text());root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Use new immutable root')
    oldroot=Path(c['v6_news_root']);old=json.loads((oldroot/'manifest.json').read_text());symbols=[x['symbol'] for x in old['mapped_symbols']]
    if sha256(oldroot/'news.csv.gz')!=old['news_sha256']:raise ValueError('CMIN cache modified')
    archive=Path('data/v7_sources/hs300news_summary.zip');receipt=json.loads(archive.with_suffix('.zip.json').read_text())
    if sha256(archive)!=receipt['sha256']:raise ValueError('Archive checksum differs')
    later,counts=read_title_archive(archive,set(symbols));earlier=pd.read_csv(oldroot/'news.csv.gz',dtype={'symbol':str},parse_dates=['published_date','research_available_date'])
    earlier=earlier[earlier.published_date<'2022-01-01'];combined=pd.concat([earlier,later],ignore_index=True)
    root.mkdir(parents=True);combined.to_csv(root/'news.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    manifest={'config_sha256':sha256(config_path),'v6_manifest_sha256':sha256(oldroot/'manifest.json'),'symbols':symbols,'mapped_symbols':old['mapped_symbols'],
      'financial_files':old['source_files'],'cmin_news_sha256':old['news_sha256'],'new_source':c['finmultitime'],'archive_receipt':receipt,
      'title_read_counts':counts,'later_title_rows':len(later),'combined_news_rows':len(combined),'news_sha256':sha256(root/'news.csv.gz'),
      'strict_pit':False,'summary_field_used':False,'availability_mode':c['availability_mode']}
    write_json(root/'manifest.json',manifest);print('PREPARED',len(symbols),len(earlier),len(later),flush=True)


def aligned_financial(frame,calendar,base):
    d=frame.copy();d['period']=pd.to_datetime(d.REPORT_DATE)
    d['available']=pd.concat([pd.to_datetime(d.NOTICE_DATE),pd.to_datetime(d.UPDATE_DATE)],axis=1).max(axis=1).dt.normalize()+pd.Timedelta(days=1)
    d=d[d.period.notna()&d.available.notna()&(d.available>=d.period)].sort_values(['available','period'],kind='stable')
    d=d[d.period==d.period.cummax()].drop_duplicates('available',keep='last')
    state=pd.merge_asof(pd.DataFrame({'date':calendar}),d[['available','period']].sort_values('available'),left_on='date',right_on='available',direction='backward')
    quarter=state.period.dt.quarter.to_numpy();result=pd.DataFrame(index=calendar)
    result['aligned_roe']=base.fin_roe*4/quarter;result['aligned_eps']=base.fin_eps*4/quarter
    for name in FINANCIAL:result[f'change63_{name}']=base[name]-base[name].shift(63)
    result['report_age']=(state.date-state.period).dt.days.to_numpy()
    return result


def load(config_path='configs/v7.json',root='data/v7'):
    c=json.loads(Path(config_path).read_text());root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if sha256(root/'news.csv.gz')!=m['news_sha256']:raise ValueError('News cache changed')
    v4=json.loads(Path('configs/v4.json').read_text());panel,raw,adj,calendar,pm,quality=price_load(v4,c['data_prices'])
    symbols=m['symbols'];days=calendar[(calendar>=c['start'])&(calendar<=c['end'])]
    news=pd.read_csv(root/'news.csv.gz',dtype={'symbol':str},parse_dates=['published_date','research_available_date']);extra=[]
    for item in m['financial_files']:
        s=item['symbol'];path=Path(item['financial_file'])
        if sha256(path)!=item['financial_sha256']:raise ValueError('Financial cache changed')
        f=financial_features(pd.read_csv(path),days,c['maximum_fundamental_age_days']);aligned=aligned_financial(pd.read_csv(path),days,f)
        f=f.join(aligned).join(news_features(news[news.symbol==s],days));f['date']=days;f['symbol']=s;extra.append(f)
    panel=panel[panel.symbol.isin(symbols)&panel.date.isin(days)].copy();panel=panel.merge(pd.concat(extra,ignore_index=True),on=['date','symbol'],validate='one_to_one')
    panel['eligible']=panel.eligible&(panel[list(FINANCIAL)].notna().sum(axis=1)>=c['min_fundamental_fields'])
    for f in list(dict.fromkeys(FEATURES+list(FINANCIAL)+ALIGNED)):panel[f]=panel.groupby('date')[f].rank(pct=True)-.5
    # Target H20 uses real future opens only as training labels, never as features.
    horizon=c['fundamental_horizon'];exit_dates=pd.Series(calendar,index=calendar).shift(-horizon-1)
    values=[]
    for s in symbols:
        p=adj[s].set_index('date').open.reindex(calendar);label=p.shift(-horizon-1)/p.shift(-1)-1
        values.append(pd.DataFrame({'date':calendar,'symbol':s,'label20':label.to_numpy(),'exit20':exit_dates.to_numpy()}))
    panel=panel.merge(pd.concat(values,ignore_index=True),on=['date','symbol'],validate='one_to_one')
    panel.loc[panel.exit_date>pd.Timestamp(c['end']),['label','target']]=np.nan
    panel.loc[panel.exit20>pd.Timestamp(c['end']),'label20']=np.nan
    panel['target']=panel.groupby('date').label.rank(pct=True)-.5;panel['target20']=panel.groupby('date').label20.rank(pct=True)-.5
    close=pd.DataFrame({s:adj[s].set_index('date').close for s in symbols}).reindex(calendar)
    returns=close.pct_change(fill_method=None).mean(axis=1);market5=returns.rolling(5).sum();vol=returns.rolling(20).std()
    panel['market_ret5']=panel.date.map(market5);panel['market_vol20']=panel.date.map(vol)
    panel['ret5_x_market']=panel.ret5*panel.market_ret5;panel['ret20_x_market']=panel.ret20*panel.market_ret5;panel['vol20_x_market']=panel.vol20*panel.market_vol20
    coverage={'symbols':len(symbols),'panel_rows':len(panel),'eligible_rows':int(panel.eligible.sum()),'strict_pit':False,
      'new_source_fields_used':['datetime','title'],'generated_summary_used':False,'news_source_switch':'2022-01-01','paper_admitted':False}
    return panel.sort_values(['date','symbol']).reset_index(drop=True),adj,calendar,symbols,m,coverage
