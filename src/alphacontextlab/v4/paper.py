"""Local paper admission record. Never submits orders or starts background services."""
import argparse,json
from pathlib import Path
from ..data import sha256
from ..study import write_json
from .data import utcnow


def prepare(reports='reports/v4',state_root='outputs/v4_paper'):
    reports=Path(reports);root=Path(state_root);root.mkdir(parents=True,exist_ok=True)
    gate=json.loads((reports/'paper_gate.json').read_text());quality=json.loads((reports/'data_quality.json').read_text())
    # Structural eligibility is checked independently of a hand-edited status string.
    admitted=gate['status']=='eligible_for_local_paper' and quality['pit_universe_certified'] and quality['cash_event_ledger_certified']
    record={'created_at':utcnow(),'status':'awaiting_future_paper_session' if admitted else 'blocked',
            'source_report_sha256':sha256(reports/'paper_gate.json'),'research_source':json.loads((reports/'provenance.json').read_text())['source_commit'],
            'candidate_model':gate['selected_model'],'reasons':gate['reasons'],
            'live_orders_sent':0,'paper_orders_sent':0,'running_service':False,
            'next_requirement':'Certified PIT universe/events and executable ledger plus financial gates before paper sessions'}
    p=root/('admission-'+record['created_at'].replace(':','').replace('+','_')+'.json');write_json(p,record)
    return record,p


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports/v4');p.add_argument('--state',default='outputs/v4_paper')
    a=p.parse_args();record,path=prepare(a.reports,a.state);print(json.dumps({'record':str(path),**record},ensure_ascii=False,indent=2))
    if record['status']=='blocked':raise SystemExit(2)
if __name__=='__main__':main()
