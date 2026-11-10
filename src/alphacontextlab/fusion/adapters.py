"""Public read-only collectors and explicit imported features; no trading/API spending."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import requests
from ..data import sha256
from ..study import write_json
from ..v4.data import utcnow
from .core import Observation,FeatureStore


def date_end(value):
    t=pd.Timestamp(value)
    if t.tzinfo is None:t=t.tz_localize('Asia/Shanghai')
    # Announcement date without an intraday timestamp is only safely known at day end.
    if t.hour==0 and t.minute==0 and t.second==0:t=t.normalize()+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1)
    return t.tz_convert('UTC').isoformat()


def fundamentals(symbol,root,fields):
    import akshare as ak
    root=Path(root);root.mkdir(parents=True,exist_ok=True);suffix='.SH' if symbol.startswith('6') else '.SZ'
    d=ak.stock_financial_analysis_indicator_em(symbol=symbol+suffix,indicator='按报告期');seen=utcnow()
    required={'SECURITY_CODE','REPORT_DATE','NOTICE_DATE',*fields.values()}
    if not required<=set(d):raise ValueError('Fundamental schema changed')
    if not d.SECURITY_CODE.astype(str).str.zfill(6).eq(symbol).all():raise ValueError('Provider entity mismatch')
    p=root/(symbol+'-financial.csv')
    if p.exists():raise ValueError('Snapshot already exists')
    d.to_csv(p,index=False,float_format='%.17g');ref='sha256:'+sha256(p);rows=[]
    for _,r in d.iterrows():
        if pd.isna(r['NOTICE_DATE']) or pd.isna(r['REPORT_DATE']):continue
        event=date_end(r['REPORT_DATE']);published=date_end(r['NOTICE_DATE'])
        # Vendor revision timestamps also constrain availability; never undo a revision.
        if pd.notna(r.get('UPDATE_DATE')):published=max(published,date_end(r['UPDATE_DATE']))
        for feature,column in fields.items():
            value=pd.to_numeric(r[column],errors='coerce')
            if not np.isfinite(value):continue
            unit='CNY/share' if feature.endswith('_cny') else 'vendor_percent'
            rows.append(Observation(source='fundamental',entity=symbol,feature=feature,event_at=event,published_at=published,
                         first_seen_at=seen,value=float(value),unit=unit,observation_key=str(r['REPORT_DATE']),
                         revision=ref,source_ref=ref).validate())
    return rows,{'symbol':symbol,'api':'akshare.stock_financial_analysis_indicator_em','rows':len(d),
                 'feature_observations':len(rows),'first_seen_at':seen,'sha256':sha256(p),'availability_vintage_certified':False,
                 'report_period_is_not_publication_time':True}


def news_cache(path):
    p=Path(path);d=json.loads(p.read_text());ref='sha256:'+sha256(p);observations=[]
    for row in d['records']:
        observations.append(Observation(source='news',entity=row['symbol'],feature='lexicon_sentiment',event_at=row['published_at'],
          published_at=row['published_at'],first_seen_at=row['first_seen_at'],value=float(row['sentiment_proxy']),unit='word_count_difference',
          observation_key=row['content_sha256'],revision=row['content_sha256'],source_ref=ref).validate())
    return observations,{'file':p.name,'sha256':sha256(p),'rows':len(observations),'availability_policy':'max(publication,first_seen)'}


def polymarket(root,limit=10):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    url='https://gamma-api.polymarket.com/markets/keyset';response=requests.get(url,params={'closed':'false','limit':limit},timeout=25)
    response.raise_for_status();payload=response.json();seen=utcnow()
    if not isinstance(payload,dict) or 'markets' not in payload:raise ValueError('Polymarket schema changed')
    p=root/'polymarket.json'
    if p.exists():raise ValueError('Snapshot already exists')
    write_json(p,payload);ref='sha256:'+sha256(p);rows=[]
    for market in payload['markets']:
        outcomes=market.get('outcomes',[]);prices=market.get('outcomePrices',[])
        if isinstance(outcomes,str):outcomes=json.loads(outcomes)
        if isinstance(prices,str):prices=json.loads(prices)
        for outcome,price in zip(outcomes,prices):
            value=float(price)
            if not 0<=value<=1:raise ValueError('Invalid implied probability')
            # startDate/createdAt describe market creation, not availability of this quote.
            rows.append(Observation(source='polymarket',entity=str(market['id']),feature='outcome_probability:'+str(outcome),
                event_at=seen,published_at=seen,first_seen_at=seen,value=value,unit='probability',
                observation_key=str(market['id'])+':'+seen,revision=ref,source_ref=ref).validate())
    return rows,{'url':url,'sha256':sha256(p),'markets':len(payload['markets']),'observations':len(rows),
                 'first_seen_at':seen,'stock_entity_links':0,'status':'unmapped_market_observations_not_stock_alpha'}


def satellite_catalogue(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);url='https://earth-search.aws.element84.com/v1/search'
    params={'collections':'sentinel-2-l2a','bbox':'121.4,31.1,121.5,31.2','limit':2,
            'datetime':'2026-10-01T00:00:00Z/2026-10-30T23:59:59Z'}
    response=requests.get(url,params=params,timeout=25);response.raise_for_status();d=response.json();seen=utcnow()
    if d.get('type')!='FeatureCollection':raise ValueError('STAC schema changed')
    p=root/'satellite-catalogue.json'
    if p.exists():raise ValueError('Snapshot already exists')
    write_json(p,d)
    # Cloud cover/scene count are retrieval quality metadata, not factory output signals.
    return [],{'url':url,'sha256':sha256(p),'catalogue_items':len(d['features']),'first_seen_at':seen,
               'imagery_pixels_processed':0,'economic_feature_observations':0,'verified_company_sites':0,
               'status':'catalogue_only_not_economic_activity_factor','query_bbox_is_demo_not_company_mapping':True}


def import_x_export(path,first_seen_at):
    """Authorized export normalized to entities; no assumed public unauthenticated X API."""
    p=Path(path);data=json.loads(p.read_text());rows=[];ref='sha256:'+sha256(p)
    for r in data:
        # Require exact mapped symbol and caller-provided finance sentiment, not regex ticker guesses.
        if not r.get('symbol') or not r.get('entity_mapping_ref'):raise ValueError('Verified X entity mapping required')
        rows.append(Observation(source='x',entity=r['symbol'],feature='sentiment',event_at=r['created_at'],published_at=r['created_at'],
               first_seen_at=first_seen_at,value=float(r['sentiment']),unit='provided_sentiment_score',
               observation_key=str(r['id']),revision=ref,source_ref=ref).validate())
    return rows


def import_satellite_features(path,first_seen_at):
    """Import actually extracted pixel/activity measurements plus site provenance."""
    p=Path(path);d=pd.read_csv(p,dtype={'entity':str});required={'entity','feature','value','unit','observed_at','processed_at','scene_id','site_mapping_ref'}
    if not required<=set(d):raise ValueError('Incomplete satellite evidence')
    allowed={'ndvi','night_lights_radiance','port_vehicle_count','sar_backscatter'};rows=[];ref='sha256:'+sha256(p)
    for _,r in d.iterrows():
        if r.feature not in allowed or pd.isna(r.site_mapping_ref) or not str(r.site_mapping_ref):raise ValueError('No verified activity feature/site mapping')
        if r.feature=='ndvi' and not -1<=float(r.value)<=1:raise ValueError('Invalid NDVI')
        if r.feature=='port_vehicle_count' and float(r.value)<0:raise ValueError('Negative count')
        rows.append(Observation(source='satellite',entity=r.entity,feature=r.feature,event_at=r.observed_at,
              published_at=r.processed_at,first_seen_at=first_seen_at,value=float(r.value),unit=r.unit,
              observation_key=str(r.scene_id),revision=ref,source_ref=ref).validate())
    return rows
