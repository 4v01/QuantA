"""Read-only V9 diagnosis, field-level filing pilot and explicit readiness gaps."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.metrics import block_mean_interval
from .contracts import fact_available

ROOT=Path('reports/prep_v10')


def pilot():
    raw=Path('data/prep_v10');receipt=json.loads((raw/'000001_2020_annual.receipt.json').read_text())
    assert sha256(receipt['file'])==receipt['sha256']
    pages=(raw/'000001_2020_annual.txt').read_text().split('\f')
    assert '平安银行股份有限公司' in pages[0] and '2020' in pages[0]
    specs=[('basic_eps','EPSJB',1.4,'CNY/share',21,'基本每股收益','1.40'),
           ('common_book_value_per_share','BPS',15.16,'CNY/share',20,'归属于本行普通股股东的每股净资产','15.16'),
           ('weighted_common_roe','ROEJQ',9.58,'percent',21,'加权平均净资产收益率','9.58')]
    cache_path='data/fusion/v5_current/fundamental/000001-financial.csv';cache=pd.read_csv(cache_path)
    row=cache[cache.REPORT_DATE.str.startswith('2020-12-31')].iloc[0];facts=[]
    for name,field,value,unit,page,label,token in specs:
        lines=[line for line in pages[page-1].splitlines() if label in line and token in line]
        assert lines and float(row[field])==value
        facts.append({'security_id':'000001.SZ','report_period':'2020-12-31','metric':name,'value':value,'unit':unit,
            'source_url':receipt['url'],'document_sha256':receipt['sha256'],'pdf_page_one_based':page,
            'publication_date_claimed':'2021-02-02T00:00:00+08:00','publication_precision':'date',
            'publication_evidence':'Dated CNINFO archive path and announcement index; official timestamp crosscheck pending',
            'candidate_available_at':'2021-02-03T00:00:00+08:00','first_seen_at':receipt['retrieved_at'],
            'document_value_verified':True,'original_document_verified':False,'publication_verified':False,
            'approved_for_feature_store':False,'revision_id':receipt['sha256']+':'+name,
            'current_cache_field':field,'current_cache_value':float(row[field]),'current_cache_update_date':row.UPDATE_DATE,
            'status':'candidate_document_fact_not_training_data'})
    result={'scope':'one original annual report, three fields; not a historical dataset certification',
        'receipt':receipt,'current_cache_file':cache_path,'current_cache_sha256':sha256(cache_path),
        'facts':facts,'documents_downloaded':1,'historically_eligible_facts':sum(fact_available(f,'2021-02-03T00:00:00+08:00','historical_reconstruction') for f in facts),
        'used_in_any_model':False,'official_publication_metadata_verified':False,
        'conclusion':'These three field values match the original document. UPDATE_DATE alone is not evidence that each field changed; no blanket backfill is authorized.'}
    write_json(ROOT/'filing_pilot.json',result)


def decision_diagnosis(write_outputs=True):
    r=Path('reports/v9/evaluation');paths=pd.read_csv(r/'screen_paths.csv.gz',parse_dates=['date','signal_date'],float_precision='round_trip')
    path=paths[(paths.method=='economic_h5')&(paths.policy=='weekly_buffer')&(paths.cost_bps==20)].copy()
    decisions=path[path.scheduled].copy()
    pred=pd.read_csv(r/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
    pred=pred[pred.method=='economic_h5'].copy();pred['percentile']=pred.groupby('date').prediction.rank(pct=True,method='average')
    held=pd.read_csv(r/'holdings.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
    held=held[(held.method=='economic_h5')&(held.policy=='weekly_buffer')&(held.cost_bps==20)]
    held=held.merge(decisions[['date','signal_date']],on='date',validate='many_to_one').merge(pred[['date','symbol','label','percentile']],left_on=['signal_date','symbol'],right_on=['date','symbol'],how='left',suffixes=('_entry','_signal'),validate='many_to_one')
    records=[]
    for day,g in held.groupby('date_entry'):
        valid=g.dropna(subset=['label','percentile']);mass=float(valid.weight.sum())
        records.append({'entry_date':day,'signal_date':g.signal_date.iloc[0],'covered_weight':mass,
            'held_weighted_percentile':float(valid.weight@valid.percentile/mass) if mass else np.nan,
            'held_weight_outside_fresh_top20':float(valid.loc[valid.percentile<=.8,'weight'].sum()),
            'held_H5_basket_gross_return':float(valid.weight@valid.label/mass) if mass else np.nan})
    table=pd.DataFrame(records)
    if write_outputs:table.to_csv(ROOT/'decision_clock.csv',index=False,float_format='%.17g')
    tails=pd.read_csv('reports/v9/quintile_daily.csv.gz',parse_dates=['date'],float_precision='round_trip')
    tails=tails[(tails.phase=='evaluation')&(tails.method=='economic_h5')&(tails.horizon==5)&(tails.quintile==5)]
    matched=tails[tails.date.isin(decisions.signal_date)]
    valid=table[table.covered_weight>=.99]
    result={'diagnostic_only':True,'candidate_not_changed':True,'scheduled_rebalances':len(decisions),
        'rebalances_with_no_current_forecast':int((~decisions.signal_date.isin(pred.date)).sum()),
        'fully_covered_H5_rebalances':len(valid),
        'top_quintile_all_days_mean_excess':float(tails.mean_excess.mean()),
        'top_quintile_actual_decision_days_mean_excess':float(matched.mean_excess.mean()),
        'actual_decision_tail_days':len(matched),
        'actual_decision_excess_ci95':block_mean_interval(matched.mean_excess,17,1000,4),
        'bootstrap_block_decisions':4,
        'mean_entry_fee_per_rebalance':float(decisions.entry_fee.mean()),
        'mean_held_H5_basket_gross_return':float(valid.held_H5_basket_gross_return.mean()),
        'mean_held_percentile':float(valid.held_weighted_percentile.mean()),
        'mean_held_weight_outside_fresh_top20':float(valid.held_weight_outside_fresh_top20.mean()),
        'same_holdings_gross_cumulative_return':float(np.prod(1+path.gross_return)-1),
        'net_cumulative_return':float(path.nav.iloc[-1]-1),
        'warning':'Overlapping H5 diagnostics, not new strategy results. Rank buffer can reduce turnover; these summaries do not prove removing it improves net profit.'}
    if write_outputs:write_json(ROOT/'decision_diagnosis.json',result)
    return result,table


def exposure_registry():
    records=[]
    for name in ['study','v2','v3','v4','v6','v7','v8','v9']:
        path=Path('configs')/(name+'.json');c=json.loads(path.read_text())
        records.append({'config':str(path),'sha256':sha256(path),'start':c.get('start'),'end':c.get('end'),
            'test_years':c.get('test_years'),'development_years':c.get('development_years'),
            'confirmation_year':c.get('confirmation_year'),'evaluation_years':c.get('evaluation_years',c.get('holdout_years'))})
    old=pd.read_csv('reports/v4/predictions.csv.gz',usecols=['date'])
    write_json(ROOT/'exposure_registry.json',{'entries':records,'V4_prediction_date_min':str(old.date.min()),'V4_prediction_date_max':str(old.date.max()),
        'V4_predictions_sha256':sha256('reports/v4/predictions.csv.gz'),'known_evaluation_through':'2026-10-30',
        '2025_2026_is_new_blind_test':False,'future_window_assigned':False,
        'rule':'Future validation must start after a new protocol seal and data readiness; already inspected histories remain retrospective.'})


def readiness():
    config=json.loads(Path('configs/v10_preparation.json').read_text());p=json.loads((ROOT/'filing_pilot.json').read_text())
    checks={'original_filing_pilot_complete':p['documents_downloaded']>=config['filing_pilot_required_documents'],
        'publication_metadata_verified':p['official_publication_metadata_verified'],
        'target_dataset_frozen':False,'historical_membership_verified':False,'stock_cash_event_ledger_verified':False,
        'economic_basis_contract_implemented':True,'source_document_pilot_available':True,
        'new_allocation_contract_integrated_and_replayed':False,'cross_regime_design_sealed':False,'prospective_window_sealed':False}
    training_checks=['original_filing_pilot_complete','publication_metadata_verified','target_dataset_frozen','historical_membership_verified','cross_regime_design_sealed']
    factor_ready=all(checks[k] for k in training_checks)
    executable_ready=factor_ready and checks['stock_cash_event_ledger_verified'] and checks['new_allocation_contract_integrated_and_replayed']
    write_json(ROOT/'READINESS.json',{'stage':'preparation_only','checks':checks,'training_prerequisites':training_checks,
        'ready_for_factor_research':factor_ready,'ready_for_training':factor_ready,'ready_for_executable_backtest':executable_ready,
        'ready_for_prospective_validation':executable_ready and checks['prospective_window_sealed'],
        'training_started':False,'new_model_results':False,'paper_admitted':False,'orders_sent':0,
        'blocking_reasons':[k for k in training_checks if not checks[k]],'all_pending_tasks':[k for k,v in checks.items() if not v]})


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/prep_v10');args=parser.parse_args();ROOT=Path(args.output)
    if ROOT.exists() and any(ROOT.iterdir()):raise ValueError('Use a new preparation output directory; do not overwrite published evidence')
    ROOT.mkdir(exist_ok=True);pilot();decision_diagnosis();exposure_registry();readiness()
    write_json(ROOT/'membership_source_review.json',json.loads(Path('data/prep_v10/membership_source_review.json').read_text()))
    baseline='095f7463b66c1a95f1a1c296cc73327b871702b6';names=subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'reports'],text=True).splitlines()
    write_json(ROOT/'old_reports_integrity.json',{'base_commit':baseline,'files':{name:sha256(name) for name in names}})
    print(json.dumps({'prepared':True,'training_started':False,'readiness':'blocked_pending_data_and_design'}))
if __name__=='__main__':main()
