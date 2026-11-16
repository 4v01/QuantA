"""Bounded original-announcement archive pilot; no factor ingestion or training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import argparse
from pathlib import Path
import hashlib,json,re,subprocess,urllib.request,urllib.parse
from ..data import sha256
from ..study import write_json

HEADERS={'User-Agent':'Mozilla/5.0','Referer':'https://www.cninfo.com.cn/','Content-Type':'application/x-www-form-urlencoded'}
ROOT=Path('data/v10_clock_sources/catalogue_v2')


def original_title(title,year):
    title=re.sub(r'\s+','',title)
    if any(word in title for word in ['摘要','英文','修订','更新','更正','补充','取消']):return False
    return bool(re.search(str(year)+r'年?年度报告(?:[（(]全文[）)])?$',title))


def request(url,payload=None):
    data=urllib.parse.urlencode(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(url,data=data,headers=HEADERS),timeout=25) as response:
        raw=response.read(40*1024*1024+1)
    if len(raw)>40*1024*1024:raise ValueError('Source exceeds pilot byte limit')
    return raw


def collect_issuer(code,org_id,years):
    query=ROOT/'queries'/(code+'.json');query.parent.mkdir(parents=True,exist_ok=True)
    payload={'pageNum':'1','pageSize':'30','column':'szse','tabName':'fulltext','plate':'',
        'stock':code+','+org_id,'searchkey':'','secid':'','category':'category_ndbg_szsh','trade':'',
        'seDate':'2020-01-01~2021-12-31','sortName':'','sortType':'','isHLtitle':'false'}
    result=[]
    try:
        if not query.exists():query.write_bytes(request('https://www.cninfo.com.cn/new/hisAnnouncement/query',payload))
        body=json.loads(query.read_text());announcements=body.get('announcements') or []
        if int(body.get('totalAnnouncement') or 0)>len(announcements):raise ValueError('Pilot query needs pagination; not silently truncating')
    except Exception as exc:
        return [{'security_id':code,'report_year':year,'status':'blocked','error':str(exc)} for year in years]
    for year in years:
        record={'security_id':code,'report_year':year,'query_url':'https://www.cninfo.com.cn/new/hisAnnouncement/query',
            'query_method':'POST','query_parameters':payload,'query_response_sha256':sha256(query),
            'used_as_historical_universe':False,'field_values_verified':False,'approved_for_training':False}
        try:
            candidates=[a for a in announcements if a.get('secCode')==code and original_title(a.get('announcementTitle',''),year)]
            if not candidates:raise ValueError('No original full annual report in the declared query window')
            a=min(candidates,key=lambda x:x['announcementTime']);relative=a['adjunctUrl']
            if not relative.startswith('finalpage/') or '..' in relative:raise ValueError('Unexpected document path')
            folder=ROOT/'documents'/(code+'_'+str(year));folder.mkdir(parents=True,exist_ok=True)
            pdf=folder/'annual.pdf';receipt_path=folder/'receipt.json'
            if not pdf.exists():
                # Reuse exactly matching, already observed pilot documents.
                known={('000001',2019):Path('data/v10_clock_sources/000001_2019_annual.pdf'),
                       ('000001',2020):Path('data/prep_v10/000001_2020_annual.pdf')}.get((code,year))
                original=json.loads(known.with_suffix('.receipt.json').read_text()) if known is not None and known.exists() else None
                if original is not None and original['url']=='https://static.cninfo.com.cn/'+relative:
                    raw=known.read_bytes();seen=original['retrieved_at']
                else:raw=request('https://static.cninfo.com.cn/'+relative);seen=datetime.now(timezone.utc).isoformat()
                if not raw.startswith(b'%PDF'):raise ValueError('Document is not a PDF')
                pdf.write_bytes(raw);write_json(receipt_path,{'url':'https://static.cninfo.com.cn/'+relative,'file':str(pdf),
                    'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'first_seen_at':seen,'historical_first_seen_proven':False})
            receipt=json.loads(receipt_path.read_text());assert sha256(pdf)==receipt['sha256']
            text_path=folder/'annual.txt';subprocess.run(['pdftotext','-layout',str(pdf),str(text_path)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            text=text_path.read_text();front=''.join(' '.join(text.split('\f')[:3]).split())
            record.update({'status':'archived_pending_field_review','announcement':a,'document_receipt':receipt,
                'publication_date_china':str(datetime.fromtimestamp(a['announcementTime']/1000,timezone.utc).astimezone(ZoneInfo('Asia/Shanghai')).date()),
                'report_year_token_in_front':str(year) in front,'api_size_unit_verified':False,
                'unselected_same_year_announcements':[x for x in announcements if x.get('secCode')==code and str(year) in x.get('announcementTitle','') and x['announcementId']!=a['announcementId']]})
        except Exception as exc:record.update({'status':'blocked','error':str(exc)})
        result.append(record)
    return result


def materialize():
    catalogue=json.loads(Path('reports/v10_clock/filing_archive_catalogue.json').read_text());written=[]
    def restore(record):
        receipt=record['document_receipt'];path=Path(receipt['file']);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            raw=request(receipt['url'])
            if hashlib.sha256(raw).hexdigest()!=receipt['sha256']:raise ValueError('Publisher bytes changed; do not overwrite the frozen source version')
            path.write_bytes(raw)
        assert sha256(path)==receipt['sha256']
        return str(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        written=list(pool.map(restore,[r for r in catalogue['records'] if r['status']=='archived_pending_field_review']))
    progress=json.loads(Path('reports/v10_clock/filing_progress.json').read_text())
    for source in progress['sources']:
        receipt=source['document_receipt'];target=Path(receipt['file'])
        matching=next(r for r in catalogue['records'] if r['security_id']=='000001' and r['report_year']==source['report_year'])
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(matching['document_receipt']['file']).read_bytes())
        assert sha256(target)==receipt['sha256']
    write_json(ROOT/'materialization.json',{'materialized_at':datetime.now(timezone.utc).isoformat(),'files':written,
        'historical_first_seen_proven':False,'published_receipts_rewritten':False,'used_for_training':False})
    print(json.dumps({'document_hashes_checked':len(written),'training_inputs_changed':False}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--materialize',action='store_true');args=parser.parse_args()
    if args.materialize:return materialize()
    output=Path('reports/v10_clock/filing_archive_catalogue.json')
    if output.exists():raise ValueError('Preserve published catalogue; use a new version for another acquisition')
    ROOT.mkdir(parents=True,exist_ok=True);config=json.loads(Path('configs/v10_preparation.json').read_text())
    directory=ROOT/'stock_directory.json'
    if not directory.exists():directory.write_bytes(request('https://www.cninfo.com.cn/new/data/szse_stock.json'))
    mapping={x['code']:x['orgId'] for x in json.loads(directory.read_text())['stockList']}
    symbols=config['filing_pilot_symbols'];years=config['filing_pilot_years'];records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(collect_issuer,code,mapping[code],years) for code in symbols]
        for future in futures:
            rows=future.result();records.extend(rows)
            for row in rows:print(json.dumps({'security':row['security_id'],'year':row['report_year'],'status':row['status']}),flush=True)
    archived=[x for x in records if x['status']=='archived_pending_field_review']
    write_json(output,{'stage':'archive_collection_only','planned_documents':len(symbols)*len(years),'archived_documents':len(archived),
        'query_window':'2020-01-01~2021-12-31','directory_url':'https://www.cninfo.com.cn/new/data/szse_stock.json',
        'directory_sha256':sha256(directory),'records':records,'complete_revision_history':False,'used_for_training':False,
        'used_in_clock_inputs':False,'note':'Current issuer directory locates disclosure records only, not historical constituents. Downloaded/dated is not field-verified or training-ready.'})
    print(json.dumps({'archived_documents':len(archived),'planned_documents':len(symbols)*len(years),'training_inputs_changed':False}),flush=True)
if __name__=='__main__':main()
