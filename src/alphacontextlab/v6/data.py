"""Whitelisted raw news fields, dated financial updates and immutable inputs."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.data import FEATURES,load as price_load

POS=('增长','增持','回购','盈利','突破','中标','利好','改善','上调','创新高')
NEG=('下滑','减持','亏损','处罚','违约','诉讼','风险','下调','下降','暴跌')
NEWS_FEATURES=[f'news_{f}_{h}' for h in [1,5,20] for f in ['count','sentiment','positive','negative','length']]
FINANCIAL={'fin_roe':'ROEJQ','fin_revenue_growth':'TOTALOPERATEREVETZ','fin_profit_growth':'PARENTNETPROFITTZ',
           'fin_leverage':'ZCFZL','fin_eps':'EPSJB','fin_bps':'BPS'}


def parse_news(path,symbol):
    # These workbooks have NO HEADER. Row zero is a real article, not field names.
    d=pd.read_excel(path,header=None)
    if d.shape[1]!=3:raise ValueError('Unexpected CMIN raw schema')
    d.columns=['published_date','title','publisher'];d['published_date']=pd.to_datetime(d.published_date,errors='raise').dt.normalize()
    if d.published_date.isna().any():raise ValueError('Missing news date')
    d['title']=d.title.fillna('').astype(str).str.strip();d['publisher']=d.publisher.fillna('').astype(str)
    d=d[d.title!=''].drop_duplicates(['published_date','title']).copy();d['symbol']=symbol
    # Date-only news never enters that date's close. No time-of-day is invented.
    d['research_available_date']=d.published_date+pd.Timedelta(days=1)
    return d


def prepare(config_path='configs/v6.json',source='/workspace/work/sources/CMIN-Dataset',root='data/v6'):
    config=json.loads(Path(config_path).read_text());source=Path(source);root=Path(root)
    if root.exists() and any(root.iterdir()):raise ValueError('Use an empty immutable data directory')
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=config['source_cmin_commit']:raise ValueError('Pin the declared CMIN commit')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():raise ValueError('CMIN source tree modified')
    candidates=json.loads(Path(config['raw_price_config']).read_text())['symbols'];news_root=source/'CMIN-CN/news/raw'
    names={p.stem:p for p in news_root.glob('*.xlsx')};mapping=[];excluded=[];news=[];sources=[]
    for symbol in candidates:
        fp=Path(config['data_fundamental'])/(symbol+'-financial.csv');f=pd.read_csv(fp,dtype={'SECURITY_CODE':str})
        if not f.SECURITY_CODE.str.zfill(6).eq(symbol).all():raise ValueError('Financial entity mismatch')
        name=str(f.SECURITY_NAME_ABBR.iloc[0]);clean=name.replace('*ST','').replace('ST','').replace(' ','')
        matches=[n for n in names if n in (name,clean)]
        if len(matches)!=1:
            excluded.append({'symbol':symbol,'name':name,'reason':'No unambiguous historical news filename; no replacement'});continue
        name=matches[0];p=names[name];rawprice=source/'CMIN-CN/price/raw'/(symbol+('.ss.csv' if symbol.startswith('6') else '.sz.csv'))
        if not rawprice.exists():raise ValueError('CMIN price identity cross-check unavailable')
        mapping.append({'symbol':symbol,'name':name,'mapping_evidence':'current vendor security identity + exact archive filename + CMIN price-code presence'})
        news.append(parse_news(p,symbol));sources.append({'symbol':symbol,'file':str(p.relative_to(source)),
             'news_sha256':sha256(p),'financial_file':str(fp),'financial_sha256':sha256(fp)})
    if len(mapping)<config['min_cross_section']:raise ValueError('Insufficient archive intersection')
    root.mkdir(parents=True);frame=pd.concat(news,ignore_index=True);frame.to_csv(root/'news.csv.gz',index=False,float_format='%.17g')
    write_json(root/'manifest.json',{'source_repository':'https://github.com/BigRoddy/CMIN-Dataset','source_commit':commit,
          'source_license':'MIT (repository compilation; original publisher rights remain)','news_sha256':sha256(root/'news.csv.gz'),
          'news_rows':len(frame),'mapped_symbols':mapping,'excluded_symbols':excluded,'source_files':sources,
          'financial_source':'Existing immutable AKShare vendor financial snapshots with NOTICE_DATE and UPDATE_DATE',
          'strict_first_seen_history_available':False,'availability_mode':config['availability_mode'],
          'news_fields_used':['published_date','title','publisher'],'no_dataset_return_labels_or_prices_used_as_features':True})
    print('PREPARED',len(mapping),'stocks',len(frame),'deduplicated headlines',flush=True)


def news_features(news,calendar,entity_only=False,entity_name=''):
    d=news.copy()
    if entity_only:d=d[d.title.str.contains(entity_name,regex=False)].copy()
    title=d.title.astype(str);d['positive']=[sum(t.count(w) for w in POS) for t in title];d['negative']=[sum(t.count(w) for w in NEG) for t in title]
    d['sentiment']=d.positive-d.negative;d['length']=title.str.len();d['count']=1
    daily=d.groupby('research_available_date')[['count','sentiment','positive','negative','length']].sum()
    days=pd.date_range(calendar.min()-pd.Timedelta(days=30),calendar.max(),freq='D')
    daily=daily.reindex(days,fill_value=0.);result=pd.DataFrame(index=days)
    # Zero counts mean no matching item in this finite archive, not an API outage or neutral sentiment label.
    for h in [1,5,20]:
        sums=daily.rolling(h,min_periods=1).sum();count=sums['count'];result[f'news_count_{h}']=np.log1p(count)
        for f in ['sentiment','positive','negative','length']:
            result[f'news_{f}_{h}']=sums[f]/count.replace(0,np.nan)
    return result.reindex(calendar)


def financial_features(frame,calendar,max_age=550):
    d=frame.copy();d['period']=pd.to_datetime(d.REPORT_DATE)
    dates=pd.concat([pd.to_datetime(d.NOTICE_DATE),pd.to_datetime(d.UPDATE_DATE)],axis=1)
    # Delayed revisions stay delayed. No NOTICE_DATE-only assignment of later-restated values.
    d['available']=dates.max(axis=1).dt.normalize()+pd.Timedelta(days=1)
    d=d[d.available.notna() & d.period.notna() & (d.available>=d.period)]
    # Only a same/newer reporting period can change the active state. This also
    # prevents a newly received old-period revision from displacing a newer report.
    d=d.sort_values(['available','period'],kind='stable')
    latest=d.period.cummax();d=d[d.period==latest].drop_duplicates('available',keep='last')
    left=pd.DataFrame({'date':calendar})
    joined=pd.merge_asof(left,d.sort_values('available'),left_on='date',right_on='available',direction='backward')
    valid=(joined.period.notna()) & ((joined.date-joined.period).dt.days<=max_age)
    result=pd.DataFrame(index=calendar)
    for feature,column in FINANCIAL.items():
        result[feature]=pd.to_numeric(joined[column],errors='coerce').where(valid).to_numpy()
    return result



def load(config_path='configs/v6.json',root='data/v6',entity_only=False):
    config=json.loads(Path(config_path).read_text());root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if sha256(root/'news.csv.gz')!=manifest['news_sha256']:raise ValueError('News snapshot changed')
    v4=json.loads(Path(config['raw_price_config']).read_text());panel,raw,adj,calendar,price_manifest,quality=price_load(v4,config['data_prices'])
    days=calendar[(calendar>=config['start'])&(calendar<=config['end'])];symbols=[x['symbol'] for x in manifest['mapped_symbols']]
    news=pd.read_csv(root/'news.csv.gz',dtype={'symbol':str},parse_dates=['published_date','research_available_date'])
    parts=[]
    for mapping,source in zip(manifest['mapped_symbols'],manifest['source_files']):
        s=mapping['symbol'];p=Path(source['financial_file'])
        if sha256(p)!=source['financial_sha256']:raise ValueError('Financial snapshot changed')
        f=financial_features(pd.read_csv(p),days,config['maximum_fundamental_age_days'])
        n=news_features(news[news.symbol==s],days,entity_only,mapping['name'])
        f=f.join(n);f['date']=days;f['symbol']=s;parts.append(f)
    extra=pd.concat(parts,ignore_index=True);panel=panel[panel.symbol.isin(symbols)&panel.date.isin(days)].copy()
    panel=panel.merge(extra,on=['date','symbol'],validate='one_to_one')
    panel['price_eligible']=panel.eligible
    panel['eligible']=panel.eligible & (panel[list(FINANCIAL)].notna().sum(axis=1)>=config['min_fundamental_fields'])
    # Redo ranks within this NEW declared intersection, on the same date only.
    for name in FEATURES+list(FINANCIAL):panel[name]=panel.groupby('date')[name].rank(pct=True)-.5
    panel.loc[panel.exit_date>pd.Timestamp(config['end']),['label','target']]=np.nan
    panel['target']=panel.groupby('date').label.rank(pct=True)-.5
    coverage={'candidate_symbols':len(v4['symbols']),'mapped_symbols':len(symbols),'calendar_sessions':len(days),
       'panel_rows':len(panel),'eligible_rows':int(panel.eligible.sum()),'excluded_rows':int((~panel.eligible).sum()),
       'minimum_fundamental_fields':config['min_fundamental_fields'],'entity_only_news':entity_only,
       'strict_pit':False,'retrospective_reconstruction':True,'paper_admitted':False}
    return panel.sort_values(['date','symbol']).reset_index(drop=True),raw,adj,calendar,symbols,manifest,price_manifest,coverage
