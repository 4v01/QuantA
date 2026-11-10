"""Recompute published IC, incremental intervals, cost curves and admission gates."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from .data import FEATURES
from .metrics import daily_ic,signal_gate,path_statistics,paired_ic_increment,paper_gate


def close(a,b,path='root'):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Keys differ: '+path)
        for k in a:close(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('Lengths differ: '+path)
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not np.isclose(a,b,atol=2e-9,rtol=2e-8):raise ValueError(f'Numeric mismatch {path}: {a} != {b}')
    elif a!=b:raise ValueError('Mismatch '+path)


def run(reports='reports/v4',config_path='configs/v4.json'):
    r=Path(reports);config=json.loads(Path(config_path).read_text());prov=json.loads((r/'provenance.json').read_text())
    if sha256(config_path)!=prov['config_sha256']:raise ValueError('Config version changed')
    frozen=subprocess.check_output(['git','show',prov['source_commit']+':'+config_path])
    assert hashlib.sha256(frozen).hexdigest()==prov['config_sha256']
    assert sha256(r/'data_manifest.json')==prov['data_manifest_sha256']
    manifest=json.loads((r/'data_manifest.json').read_text());assert len(manifest['files'])==len(config['symbols'])*2
    versions=pd.read_csv(r/'data_versions.csv',dtype={'symbol':str})
    for item in manifest['files']:
        row=versions[versions.file==item['file']].iloc[0]
        assert row.sha256==item['sha256'] and row.symbol==item['symbol']
    pred=pd.read_csv(r/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
    assert not pred.duplicated(['model','date','symbol']).any()
    actual=pd.read_csv(r/'daily_ic.csv',parse_dates=['date']);computed=[]
    for (model,fold),g in pred.groupby(['model','fold']):
        d=daily_ic(g,g.prediction,config['min_cross_section']);d['model']=model;d['fold']=fold;computed.append(d)
    calculated=pd.concat(computed).sort_values(['model','fold','date']).reset_index(drop=True)
    expected=actual.sort_values(['model','fold','date']).reset_index(drop=True)
    assert len(calculated)==len(expected)
    assert calculated[['date','model','fold']].equals(expected[['date','model','fold']])
    np.testing.assert_allclose(calculated[['n','ic','rank_ic']],expected[['n','ic','rank_ic']],atol=1e-9,rtol=1e-8)
    signals={m:signal_gate(actual[actual.model==m],config) for m in config['models']}
    close(signals,json.loads((r/'signal_gates.json').read_text()),'signal_gates')
    increments={m:paired_ic_increment(actual[actual.model=='ridge'],actual[actual.model==m],config) for m in config['models'] if m!='ridge'}
    close(increments,json.loads((r/'incremental_ic.json').read_text()),'increments')
    # Selection and label cutoffs are recomputed from public validation paths/evidence.
    val=pd.read_csv(r/'validation_ic.csv',parse_dates=['date'])
    for selection in sorted((r/'evidence').glob('*-selection.json')):
        fold=selection.name.replace('-selection.json','');e=json.loads(selection.read_text());s=e['split']
        assert pd.Timestamp(s['train_max_label_exit'])<pd.Timestamp(s['validation_start'])
        assert pd.Timestamp(s['refit_max_label_exit'])<pd.Timestamp(s['test_start'])
        assert e['provenance']==prov
        for m,record in e['models'].items():
            chosen=record['selected']
            if 'candidates' in record:
                winner=max(record['candidates'],key=lambda x:x['absolute_validation_rank_ic']);close(winner,chosen)
            assert chosen['direction'] in [-1,1]
            if chosen['raw_validation_rank_ic'] is not None:assert (1 if chosen['raw_validation_rank_ic']>=0 else -1)==chosen['direction']
            got=val[(val.model==m)&(val.fold==fold)].rank_ic.mean()
            assert (pd.isna(got) and chosen['absolute_validation_rank_ic']==-1) or np.isclose(got,chosen['absolute_validation_rank_ic'],atol=1e-9)
    selected=str(val.groupby('model').rank_ic.mean().sort_values(ascending=False,kind='stable').index[0])
    assert selected==json.loads((r/'frozen_candidate.json').read_text())['model']
    paths=pd.read_csv(r/'screen_paths.csv.gz');table=pd.read_csv(r/'screen_summary.csv');yearly={};curves=0
    for (m,cost),g in paths.groupby(['model','cost_bps']):
        assert pd.to_datetime(g.date).is_monotonic_increasing and not g.date.duplicated().any()
        assert np.all(pd.to_datetime(g.signal_date)<pd.to_datetime(g.date))
        assert np.all(pd.to_datetime(g.exit_date)>pd.to_datetime(g.date))
        np.testing.assert_allclose(g.nav,np.cumprod(1+g.net_return),atol=1e-9,rtol=1e-8)
        stat=path_statistics(g);row=table[(table.model==m)&(table.cost_bps==cost)].iloc[0]
        close(stat,{k:row[k].item() if hasattr(row[k],'item') else row[k] for k in stat})
        if cost==config['screen_primary_cost_bps']:
            yearly[m]={str(y):path_statistics(z) for y,z in g.groupby(pd.to_datetime(g.date).dt.year)}
        curves+=1
    close(yearly,json.loads((r/'yearly_screen.json').read_text()))
    quality=json.loads((r/'data_quality.json').read_text())
    assert quality['pit_universe_certified'] is False and quality['cash_event_ledger_certified'] is False
    linear_pass=any(signals[m]['passed'] for m in ['ridge','elasticnet'])
    performance=table[(table.model==selected)&(table.cost_bps==config['screen_primary_cost_bps'])].iloc[0].to_dict()
    gate=paper_gate(linear_pass,signals[selected],performance,yearly[selected],quality,config)
    published=json.loads((r/'paper_gate.json').read_text());close(gate,{k:published[k] for k in gate})
    assert published['selected_model']==selected and published['linear_pass']==linear_pass
    # Every available evaluation calendar date has all five arms, not cherry-picked years.
    sets=[set(pred[pred.model==m].date) for m in config['models']];assert all(x==sets[0] for x in sets)
    counts=pred.groupby(['model','date']).size();assert counts.groupby(level=1).nunique().max()==1
    result={'status':'passed','source_commit':prov['source_commit'],'oof_model_rows':len(pred),
            'daily_ic_rows_recomputed':len(calculated),'screen_curves_recomputed':curves,
            'selected_model':selected,'paper_status':published['status'],'live_orders_sent':0}
    print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports/v4');p.add_argument('--config',default='configs/v4.json');a=p.parse_args();run(a.reports,a.config)
if __name__=='__main__':main()
