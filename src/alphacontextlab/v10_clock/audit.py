"""Public replay plus independent pooled/sleeve budget and fee conservation."""
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v8.audit import read,equal,frame_equal
from ..v8.study import path_metrics
from .ledger import simulate,apply_cost
from .study import inputs


def account_check(gross,held,accounts,sleeves,tape,symbols,clock,config):
    dates=pd.DatetimeIndex(gross.date);n=5 if clock=='stagger5' else 1
    w=held.pivot(index='date',columns='symbol',values='weight').reindex(index=dates,columns=symbols).fillna(0.).to_numpy()
    returns=tape.pivot(index='date',columns='symbol',values='next_open_return').reindex(index=dates,columns=symbols).fillna(0.).to_numpy()
    positions=sleeves.pivot(index=['date','sleeve'],columns='symbol',values='weight').reindex(pd.MultiIndex.from_product([dates,range(n)]),columns=symbols).fillna(0.).to_numpy().reshape(len(dates),n,len(symbols))
    account=accounts.set_index(['date','sleeve']).reindex(pd.MultiIndex.from_product([dates,range(n)]))
    old=np.zeros((n,len(symbols)));budgets=np.full(n,1/n)
    for i,(_,row) in enumerate(gross.iterrows()):
        s=positions[i];a=account.loc[dates[i]];np.testing.assert_allclose(s.sum(axis=0),w[i],atol=1e-10)
        np.testing.assert_allclose(a.budget_before,budgets,atol=1e-10)
        np.testing.assert_allclose(a.cash_before,budgets-old.sum(axis=1),atol=1e-10)
        assert (s>=-1e-12).all() and (s.sum(axis=1)<=budgets+1e-10).all()
        expected_active=i%5 if n==5 else (0 if i>=int(clock[-1]) and (i-int(clock[-1]))%5==0 else -1)
        for sleeve in range(n):
            assert bool(a.active.iloc[sleeve])==(sleeve==expected_active)
            if sleeve!=expected_active:np.testing.assert_allclose(s[sleeve],old[sleeve],atol=1e-10)
        change=s-old;net=float(abs(change.sum(axis=0)).sum());intent=float(abs(change).sum())
        equal(net,float(row.entry_turnover));equal(intent,float(row.gross_sleeve_turnover))
        sleeve_gross=s@returns[i];g=float(w[i]@returns[i]);equal(g,float(row.gross_return))
        np.testing.assert_allclose(a.sleeve_gross_contribution,sleeve_gross,atol=1e-10)
        after=s*(1+returns[i])/(1+g);budgets=(budgets+sleeve_gross)/(1+g)
        np.testing.assert_allclose(a.budget_after_return,budgets,atol=1e-10);equal(float(budgets.sum()),1.)
        np.testing.assert_allclose(a.cash_after_return,budgets-after.sum(axis=1),atol=1e-10)
        if i==len(gross)-1:
            equal(float(row.terminal_turnover),float(after.sum()));np.testing.assert_allclose(a.cash_after_terminal,budgets,atol=1e-10)
            assert (a.end_stock_exposure==0).all()
        else:
            equal(float(row.terminal_turnover),0.);np.testing.assert_allclose(a.end_stock_exposure,after.sum(axis=1),atol=1e-10)
        old=after


def phase_audit(root,c,base):
    root=Path(root);phase=root.name;pred,tape,symbols=inputs(phase,c);prov=json.loads((root/'provenance.json').read_text())
    assert prov['model_training'] is False and prov['executable_stock_ledger'] is False
    assert prov['config_sha256']==sha256('configs/v10_clock.json') and prov['protocol_seal_sha256']==sha256('reports/v10_clock/protocol_seal.json')
    frozen=subprocess.check_output(['git','show',prov['source_commit']+':configs/v10_clock.json'])
    assert hashlib.sha256(frozen).hexdigest()==prov['config_sha256'];equal(prov['inputs'],c['inputs'][phase])
    paths=read(root/'paths.csv.gz',['date','signal_date','exit_date']);held=read(root/'holdings.csv.gz',['date']);accounts=read(root/'sleeve_accounts.csv.gz',['date']);sleeves=read(root/'sleeve_positions.csv.gz',['date']);summary=read(root/'summary.csv');common=read(root/'common_window_summary.csv')
    old=read(Path('reports/v9')/phase/'screen_paths.csv.gz',['date']);count=0
    for method in c['methods']:
        q=pred[pred.method==('economic_h5' if method=='equal_weight' else method)]
        for clock in c['clocks']:
            def subset(frame):return frame[(frame.method==method)&(frame.clock==clock)]
            gross,w,a,s=simulate(q,tape,symbols,base,c,clock,method=='equal_weight')
            frame_equal(w,subset(held),['date','symbol'],['weight'])
            frame_equal(s,subset(sleeves),['date','sleeve','symbol'],['weight'])
            frame_equal(a,subset(accounts),['date','sleeve'],['budget_before','cash_before','target_exposure','budget_after_return','cash_after_return','cash_after_terminal','end_stock_exposure','sleeve_gross_contribution','active'])
            account_check(gross,subset(held),subset(accounts),subset(sleeves),tape,symbols,clock,c)
            for cost in c['cost_bps']:
                p=apply_cost(gross,cost);actual=subset(paths);actual=actual[actual.cost_bps==cost]
                frame_equal(p,actual,['date'],['gross_return','net_return','nav','entry_turnover','gross_sleeve_turnover','terminal_turnover','turnover','gross_exposure','entry_fee','terminal_fee','stale_held_marks'])
                for source,reported in [(p,summary),(p[p.session_index>=c['secondary_common_window_start_index']],common)]:
                    row=subset(reported);row=row[row.cost_bps==cost].iloc[0]
                    for name,value in path_metrics(source).items():
                        if value is None:assert pd.isna(row[name])
                        else:equal(value,float(row[name]))
                if clock=='phase0':
                    reference=old[(old.method==method)&(old.policy==('equal_weekly' if method=='equal_weight' else 'weekly_buffer'))&(old.cost_bps==cost)]
                    frame_equal(p,reference,['date'],['gross_return','net_return','nav','turnover'])
                count+=1
    gate=json.loads((root/'financial_gate.json').read_text());assert gate['primary_clock']=='stagger5' and gate['primary_method']=='economic_h5' and not gate['paper_admitted'] and gate['orders_sent']==0
    row=summary[(summary.method=='economic_h5')&(summary.clock=='stagger5')&(summary.cost_bps==20)].iloc[0]
    signal=json.loads((Path('reports/v9')/phase/'ic_summary.json').read_text())['economic_h5'];reasons=[]
    if signal['rank_ic']<base['minimum_ic'] or signal['ic_coverage']<base['minimum_ic_coverage'] or signal['passing_fold_fraction']<base['minimum_passing_fold_fraction'] or signal['ci95'][0]<=0:reasons.append('frozen_H5_signal_gate_failed')
    if not np.isfinite(row.net_sharpe) or row.net_sharpe<=c['minimum_net_sharpe']:reasons.append('net_sharpe_not_above_1')
    equal(gate['reasons'],reasons);assert gate['financial_passed']==(not reasons)
    return {'phase':phase,'curves_replayed':count,'V9_phase0_paths_equal':True,'sleeve_budget_and_cash_conservation':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt');p.add_argument('--local-filings',action='store_true');a=p.parse_args();c=json.loads(Path('configs/v10_clock.json').read_text());base=json.loads(Path(c['base_config']).read_text());result=[]
    seal=json.loads(Path('reports/v10_clock/protocol_seal.json').read_text());assert seal['config_sha256']==sha256('configs/v10_clock.json') and seal['primary_fixed_before_results'] and not seal['phase_winner_selection_allowed']
    for name,digest in seal['core_sources'].items():assert sha256(name)==digest
    for phase in ['development','evaluation']:
        root=Path('reports/v10_clock')/phase
        if root.exists():result.append(phase_audit(root,c,base))
    inventory=Path('reports/v10_clock/old_reports_integrity.json')
    if inventory.exists():
        inv=json.loads(inventory.read_text());names=list(inv['files']);raw=subprocess.check_output(['git','cat-file','--batch'],input=''.join(inv['base_commit']+':'+p+'\n' for p in names).encode());offset=0
        for name in names:
            end=raw.index(b'\n',offset);size=int(raw[offset:end].decode().split()[2]);offset=end+1
            assert hashlib.sha256(raw[offset:offset+size]).hexdigest()==inv['files'][name]==sha256(name);offset+=size+1
    filing_path=Path('reports/v10_clock/filing_progress.json')
    if filing_path.exists():
        filing=json.loads(filing_path.read_text());assert filing['documents_checked']==2 and filing['fields_checked']==6 and not filing['used_for_training'] and not filing['used_in_clock_inputs']
        for source in filing['sources']:
            public_meta=Path(source['public_response_file']);assert sha256(public_meta)==source['response_sha256']
            announcement=json.loads(public_meta.read_text())['announcement'];equal(announcement,source['announcement'])
            day=datetime.fromtimestamp(announcement['announcementTime']/1000,timezone.utc).astimezone(ZoneInfo('Asia/Shanghai')).date()
            assert str(day)==source['publication_date_china']==source['document_receipt']['archive_path_date']
            assert announcement['secCode']=='000001' and announcement['adjunctUrl']==source['document_receipt']['url'].split('static.cninfo.com.cn/')[1]
            if a.local_filings:
                receipt=source['document_receipt'];assert sha256(receipt['file'])==receipt['sha256']
                text=subprocess.check_output(['pdftotext','-layout',receipt['file'],'-']).decode();pages=text.split('\f')
                assert str(source['report_year']) in pages[0] and '平安银行股份有限公司' in pages[0]
                for fact in filing['facts']:
                    if fact['report_period'].startswith(str(source['report_year'])):
                        assert str(fact['value']) in pages[fact['pdf_page_one_based']-1]
        assert all(not f['approved_for_training'] and not f['used_in_clock_inputs'] for f in filing['facts'])
    archive_count=0;catalogue_path=Path('reports/v10_clock/filing_archive_catalogue.json')
    if catalogue_path.exists():
        catalogue=json.loads(catalogue_path.read_text());pilot=json.loads(Path('configs/v10_preparation.json').read_text())
        expected={(code,year) for code in pilot['filing_pilot_symbols'] for year in pilot['filing_pilot_years']}
        assert {(r['security_id'],r['report_year']) for r in catalogue['records']}==expected
        assert len(catalogue['records'])==len(expected)==catalogue['planned_documents']
        assert not catalogue['used_for_training'] and not catalogue['used_in_clock_inputs']
        for record in catalogue['records']:
            assert not record['approved_for_training'] and not record['field_values_verified']
            if record['status']=='archived_pending_field_review':
                archive_count+=1;announcement=record['announcement'];receipt=record['document_receipt']
                assert announcement['secCode']==record['security_id'] and receipt['url'].endswith(announcement['adjunctUrl'])
                day=datetime.fromtimestamp(announcement['announcementTime']/1000,timezone.utc).astimezone(ZoneInfo('Asia/Shanghai')).date()
                assert str(day)==record['publication_date_china']
                if a.local_filings:
                    assert sha256(receipt['file'])==receipt['sha256'] and Path(receipt['file']).read_bytes().startswith(b'%PDF')
        assert archive_count==catalogue['archived_documents']
    receipt={'passed':True,'phases':result,'local_filing_documents_verified':a.local_filings,'archive_documents':archive_count,
        'archive_field_validation_complete':False,'model_training':False,'paper_admitted':False,'orders_sent':0}
    if a.receipt:write_json(a.receipt,receipt)
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
