"""Optional read-only Baostock probe; never changes the frozen training inputs.

Install the optional source package in an isolated target directory, not the fit
environment. An API publication date alone does not certify revision history.
"""
import argparse,json,socket
from pathlib import Path
from datetime import datetime,timezone
from ..data import sha256
from ..study import write_json


def run(output):
    socket.setdefaulttimeout(15)
    receipt={'started_at':datetime.now(timezone.utc).isoformat(),'status':'started',
        'source':'baostock','source_version':'0.9.4','reference':'https://github.com/microsoft/qlib/tree/main/scripts/data_collector/pit',
        'used_in_V9_model_inputs':False,'revision_history_certified':False,'orders_sent':0,'queries':[]}
    write_json(output,receipt)
    try:
        import baostock as bs
        session=bs.login()
        if session.error_code!='0':raise RuntimeError('Anonymous login failed: '+session.error_msg)
        root=Path('data/v9_sources/baostock_probe');root.mkdir(exist_ok=True)
        jobs=[('membership_2019',lambda:bs.query_hs300_stocks(date='2019-01-02')),
              ('membership_2024',lambda:bs.query_hs300_stocks(date='2024-01-02')),
              ('profit_000001_2020Q4',lambda:bs.query_profit_data(code='sz.000001',year=2020,quarter=4))]
        for name,query in jobs:
            response=query()
            if response.error_code!='0':raise RuntimeError(name+': '+response.error_msg)
            data=response.get_data();path=root/(name+'.csv');data.to_csv(path,index=False)
            receipt['queries'].append({'name':name,'rows':len(data),'columns':list(data.columns),'file':str(path),'sha256':sha256(path),
                'sample':data.head(2).to_dict('records')})
            write_json(output,receipt)
        bs.logout();receipt['status']='probe_complete_provisional_only'
        receipt['caveat']='Historical membership/pubDate API rows need version and original-filing checks; do not backdate current revised values.'
    except Exception as exc:
        receipt['status']='blocked';receipt['error']=str(exc)
    receipt['finished_at']=datetime.now(timezone.utc).isoformat();write_json(output,receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/v9/data_source_probe.json');args=parser.parse_args();run(args.output)
