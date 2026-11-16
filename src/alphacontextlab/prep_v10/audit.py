import argparse,hashlib,json,subprocess
from pathlib import Path
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v8.audit import equal,frame_equal
from .build import ROOT,decision_diagnosis
from .contracts import fact_available


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-documents',action='store_true');parser.add_argument('--receipt');a=parser.parse_args()
    config=json.loads(Path('configs/v10_preparation.json').read_text());ready=json.loads((ROOT/'READINESS.json').read_text())
    assert config['stage']=='preparation_only' and not config['training_allowed'] and not config['paper_allowed'] and not config['orders_allowed']
    assert not ready['ready_for_training'] and not ready['training_started'] and not ready['new_model_results'] and not ready['paper_admitted'] and ready['orders_sent']==0
    assert set(ready['blocking_reasons'])=={k for k in ready['training_prerequisites'] if not ready['checks'][k]}
    assert set(ready['all_pending_tasks'])=={k for k,v in ready['checks'].items() if not v}
    pilot=json.loads((ROOT/'filing_pilot.json').read_text());assert len(pilot['facts'])==3 and pilot['documents_downloaded']==1
    for fact in pilot['facts']:
        assert fact['document_value_verified'] and fact['value']==fact['current_cache_value']
        assert not fact['approved_for_feature_store'] and not fact['publication_verified']
        assert not fact_available(fact,'2021-02-03T00:00:00+08:00','historical_reconstruction')
    registry=json.loads((ROOT/'exposure_registry.json').read_text())
    for item in registry['entries']:assert sha256(item['config'])==item['sha256']
    assert not registry['2025_2026_is_new_blind_test'] and not registry['future_window_assigned']
    assert registry['V4_predictions_sha256']==sha256('reports/v4/predictions.csv.gz')
    diagnosis,table=decision_diagnosis(False);equal(diagnosis,json.loads((ROOT/'decision_diagnosis.json').read_text()))
    saved=pd.read_csv(ROOT/'decision_clock.csv',parse_dates=['entry_date','signal_date'],float_precision='round_trip')
    frame_equal(table,saved,['entry_date','signal_date'],['covered_weight','held_weighted_percentile','held_weight_outside_fresh_top20','held_H5_basket_gross_return'])
    inventory=json.loads((ROOT/'old_reports_integrity.json').read_text());names=list(inventory['files'])
    raw=subprocess.check_output(['git','cat-file','--batch'],input=''.join(inventory['base_commit']+':'+p+'\n' for p in names).encode());offset=0
    for name in names:
        end=raw.index(b'\n',offset);size=int(raw[offset:end].decode().split()[2]);offset=end+1
        assert hashlib.sha256(raw[offset:offset+size]).hexdigest()==inventory['files'][name]==sha256(name);offset+=size+1
    if a.local_documents:
        assert sha256(pilot['receipt']['file'])==pilot['receipt']['sha256']
        assert sha256(pilot['current_cache_file'])==pilot['current_cache_sha256']
        cache=pd.read_csv(pilot['current_cache_file']);row=cache[cache.REPORT_DATE.str.startswith('2020-12-31')].iloc[0]
        for fact in pilot['facts']:equal(float(row[fact['current_cache_field']]),fact['value'])
        # Verify the downloaded document and re-extract text; do not trust a
        # previously edited text cache or certify its historical publication.
        text=subprocess.check_output(['pdftotext','-layout',pilot['receipt']['file'],'-']).decode();pages=text.split('\f')
        assert '平安银行股份有限公司' in pages[0] and '2020' in pages[0]
        assert '基本每股收益' in pages[20] and '1.40' in pages[20]
        assert '15.16' in pages[19] and '9.58' in pages[20]
    result={'passed':True,'stage':'preparation_only','old_reports_unchanged':len(names),'decision_rows_verified':len(table),
        'local_document_verified':a.local_documents,'historical_publication_certified':False,'training_started':False,'paper_admitted':False,'orders_sent':0}
    if a.receipt:write_json(a.receipt,result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
