"""Download current event sources into a NEW cache; never silently revise an archive."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from alphacontextlab.v3.data import prepare_events


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='data/v3_corporate_new');args=parser.parse_args()
    root=Path(args.root);raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
    sources={f'{s}-fhsp.html':f'https://fundf10.eastmoney.com/fhsp_{s}.html' for s in ['510300','510050','510500','159915']}
    pdfs={'split_2015.pdf':'https://www.sse.com.cn/disclosure/fund/announcement/c/2015-04-15/510500_20150415_2.pdf',
          'split_2022.pdf':'https://pdf.dfcfw.com/pdf/H2_AN202208291577764682_1.pdf'}
    for name,url in {**sources,**pdfs}.items():
        target=raw/name
        if target.exists():continue
        with urlopen(url,timeout=45) as response:payload=response.read()
        if name.endswith('.pdf') and not payload.startswith(b'%PDF'):raise ValueError('Not a primary PDF: '+url)
        target.write_bytes(payload)
    records=[{'file':name,'url':url,'sha256':hashlib.sha256((raw/name).read_bytes()).hexdigest()} for name,url in pdfs.items()]
    (root/'source_pdfs.json').write_text(json.dumps(records,indent=2)+'\n')
    result=prepare_events(root);print(json.dumps({'events':len(result['events']),'cache':str(root),'new_version_warning':'Current pages may differ from the published snapshot.'}))


if __name__=='__main__':main()
