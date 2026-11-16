"""Reproduce the read-only filing pilot without overwriting original receipts."""
from datetime import datetime,timezone
import hashlib,json,subprocess,urllib.request
from pathlib import Path
from ..data import sha256
from ..study import write_json

URL='https://static.cninfo.com.cn/finalpage/2021-02-02/1209224370.PDF'
DIGEST='a502ec2550605604679113fd27a43fec2ffacc259cd7cbab1d2b7279a53bb25a'


def fetch(url,digest,path):
    if path.exists():
        if sha256(path)!=digest:raise ValueError('Existing source changed: '+str(path))
        return False
    with urllib.request.urlopen(url,timeout=30) as response:raw=response.read(20*1024*1024+1)
    if len(raw)>20*1024*1024 or hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('Upstream bytes changed; record a new source version, do not replace this one')
    path.write_bytes(raw);return True


def main():
    root=Path('data/prep_v10');root.mkdir(exist_ok=True);pdf=root/'000001_2020_annual.pdf';side=root/'000001_2020_annual.receipt.json'
    created=fetch(URL,DIGEST,pdf)
    if not side.exists():
        write_json(side,{'url':URL,'file':str(pdf),'bytes':pdf.stat().st_size,'sha256':DIGEST,
            'retrieved_at':datetime.now(timezone.utc).isoformat(),'archive_path_date':'2021-02-02',
            'historical_first_seen_proven':False,'used_for_training':False})
    receipt=json.loads(side.read_text());assert receipt['sha256']==DIGEST and receipt['used_for_training'] is False
    subprocess.run(['pdftotext','-layout',str(pdf),str(root/'000001_2020_annual.txt')],check=True)
    review=json.loads(Path('reports/prep_v10/membership_source_review.json').read_text())
    for item in review['read_files']:fetch(item['url'],item['sha256'],Path(item['file']))
    target=root/'membership_source_review.json'
    if not target.exists():write_json(target,review)
    else:assert json.loads(target.read_text())==review
    print(json.dumps({'filing_cached':True,'source_sha256':DIGEST,'historical_first_seen_proven':False,'training_started':False}))
if __name__=='__main__':main()
