"""Field-scoped publication checks, isolated from the clock experiment inputs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo
import json,re,shutil
import pandas as pd
from ..data import sha256
from ..study import write_json


def build():
    root=Path('data/v10_clock_sources');out=Path('reports/v10_clock');rows=[];metadata=[]
    specs=[(2019,'1207305488',root/'000001_2019_annual.receipt.json',root/'000001_2019_annual.txt',
        [('basic_eps','EPSJB',1.54,19,'基本每股收益','CNY/share'),('common_book_value_per_share','BPS',14.07,19,'归属于普通股股东的每股净资产','CNY/share'),('weighted_common_roe','ROEJQ',11.30,19,'加权平均净资产收益率','percent')]),
        (2020,'1209224370',Path('data/prep_v10/000001_2020_annual.receipt.json'),Path('data/prep_v10/000001_2020_annual.txt'),
        [('basic_eps','EPSJB',1.40,21,'基本每股收益','CNY/share'),('common_book_value_per_share','BPS',15.16,20,'归属于本行普通股股东的每股净资产','CNY/share'),('weighted_common_roe','ROEJQ',9.58,21,'加权平均净资产收益率','percent')])]
    cache_path=Path('data/fusion/v5_current/fundamental/000001-financial.csv');cache=pd.read_csv(cache_path)
    for year,ident,receipt_path,text_path,fields in specs:
        receipt=json.loads(receipt_path.read_text());assert sha256(receipt['file'])==receipt['sha256']
        meta_path=root/(ident+'_publication.json');body=json.loads(meta_path.read_text());a=body['announcement']
        public_meta=out/'source_publication_metadata'/(ident+'.json');public_meta.parent.mkdir(exist_ok=True);shutil.copy2(meta_path,public_meta)
        assert a['secCode']=='000001' and a['announcementId']==ident and str(year) in a['announcementTitle']
        assert a['adjunctUrl']==receipt['url'].split('static.cninfo.com.cn/')[1]
        date=datetime.fromtimestamp(a['announcementTime']/1000,timezone.utc).astimezone(ZoneInfo('Asia/Shanghai')).date()
        assert str(date)==receipt['archive_path_date']
        pages=text_path.read_text().split('\f');assert str(year) in pages[0] and '平安银行股份有限公司' in pages[0]
        current=cache[cache.REPORT_DATE.str.startswith(f'{year}-12-31')].iloc[0]
        metadata.append({'announcement_id':ident,'security_id':'000001.SZ','report_year':year,
            'endpoint':'https://www.cninfo.com.cn/new/announcement/bulletin_detail','query_method':'POST',
            'request':{'announceId':ident,'flag':'true'},'response_sha256':sha256(meta_path),
            'public_response_file':str(public_meta),
            'announcement':a,'publication_date_china':str(date),'publication_precision':'date',
            'document_url_and_issuer_matched':True,'document_receipt':receipt,
            'api_size_field':a.get('adjunctSize'),'api_size_unit_verified':False,'historical_document_bytes_proven':False})
        for metric,column,value,page,label,unit in fields:
            lines=[line for line in pages[page-1].splitlines() if label in line]
            values=[float(re.search(r'[-+]?\d+(?:\.\d+)?',line.split(label,1)[1].replace(',','')).group()) for line in lines]
            assert any(abs(v-value)<1e-12 for v in values) and abs(float(current[column])-value)<1e-12
            rows.append({'security_id':'000001.SZ','report_period':f'{year}-12-31','metric':metric,'value':value,'unit':unit,
                'document_sha256':receipt['sha256'],'pdf_page_one_based':page,'publication_date_verified':True,
                'publication_date_china':str(date),'candidate_available_date':str(date+timedelta(days=1)),
                'actual_document_first_seen':receipt['retrieved_at'],'historical_first_seen_proven':False,
                'current_cache_field':column,'current_cache_value':float(current[column]),'current_cache_update_date':current.UPDATE_DATE,
                'approved_for_training':False,'used_in_clock_inputs':False,'status':'field_and_publication_date_checked_staging_only'})
    write_json(out/'filing_progress.json',{'stage':'data_engineering_not_model_input','documents_checked':2,'fields_checked':len(rows),
        'official_publication_dates_matched':2,'target_pilot_documents':16,'original_revision_chain_complete':False,
        'current_cache_file':str(cache_path),'current_cache_sha256':sha256(cache_path),'facts':rows,'sources':metadata,
        'api_size_field_units_unverified':True,'used_for_training':False,'used_in_clock_inputs':False,
        'caveat':'Official issuer/id/date/path match does not prove historical byte identity or a complete revision chain; retrieval times remain current. No blanket backfill.'})
    print(json.dumps({'documents_checked':2,'fields_checked':6,'publication_dates_matched':2,'training_inputs_changed':False}))
if __name__=='__main__':build()
