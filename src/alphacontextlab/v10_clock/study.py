import argparse,json,subprocess
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v8.study import path_metrics
from ..v4.metrics import path_statistics,block_mean_interval
from .ledger import simulate,apply_cost


def inputs(phase,config):
    root=Path('reports/v9')/phase
    for name,digest in config['inputs'][phase].items():
        if sha256(root/name)!=digest:raise ValueError('Frozen input changed: '+str(root/name))
    pred=pd.read_csv(root/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
    pred=pred[pred.evaluation_period].copy()
    tape=pd.read_csv(root/'market_tape.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip')
    symbols=json.loads((root/'provenance.json').read_text())['symbols']
    return pred,tape,symbols


def run(phase,out):
    out=Path(out);c=json.loads(Path('configs/v10_clock.json').read_text());base=json.loads(Path(c['base_config']).read_text())
    if out.exists() and any(out.iterdir()):raise ValueError('Use new output directory')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Freeze clean source/protocol first')
    seal=json.loads(Path('reports/v10_clock/protocol_seal.json').read_text());assert seal['config_sha256']==sha256('configs/v10_clock.json')
    pred,tape,symbols=inputs(phase,c);out.mkdir(parents=True)
    prov={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'phase':phase,
        'config_sha256':sha256('configs/v10_clock.json'),'base_config_sha256':sha256(c['base_config']),
        'protocol_seal_sha256':sha256('reports/v10_clock/protocol_seal.json'),'inputs':c['inputs'][phase],
        'source_report_commit':c['source_report_commit'],'symbols':symbols,'seed':17,'model_training':False,
        'versions':{name:version(name) for name in ['numpy','pandas','scipy']},'prospective_blind_test':False,'executable_stock_ledger':False}
    write_json(out/'provenance.json',prov);paths=[];weights=[];sleeves=[];accounts=[];summary=[];common=[]
    for method in c['methods']:
        p=pred[pred.method==('economic_h5' if method=='equal_weight' else method)]
        for clock in c['clocks']:
            print('REPLAY',phase,method,clock,flush=True)
            gross,w,a,s=simulate(p,tape,symbols,base,c,clock,method=='equal_weight')
            for frame in [w,a,s]:frame['method']=method;frame['clock']=clock
            weights.append(w);accounts.append(a);sleeves.append(s)
            for cost in c['cost_bps']:
                path=apply_cost(gross,cost);path['method']=method;path['clock']=clock;paths.append(path)
                summary.append({'method':method,'clock':clock,'cost_bps':cost,**path_metrics(path)})
                window=path[path.session_index>=c['secondary_common_window_start_index']]
                common.append({'method':method,'clock':clock,'cost_bps':cost,**path_metrics(window)})
    pd.concat(paths,ignore_index=True).to_csv(out/'paths.csv.gz',index=False,float_format='%.17g')
    pd.concat(weights,ignore_index=True).to_csv(out/'holdings.csv.gz',index=False,float_format='%.17g')
    pd.concat(sleeves,ignore_index=True).to_csv(out/'sleeve_positions.csv.gz',index=False,float_format='%.17g')
    pd.concat(accounts,ignore_index=True).to_csv(out/'sleeve_accounts.csv.gz',index=False,float_format='%.17g')
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False,float_format='%.17g');pd.DataFrame(common).to_csv(out/'common_window_summary.csv',index=False,float_format='%.17g')
    row=summary[(summary.method==c['primary_method'])&(summary.clock==c['primary_clock'])&(summary.cost_bps==20)].iloc[0]
    signal=json.loads((Path('reports/v9')/phase/'ic_summary.json').read_text())[c['primary_method']]
    reasons=[]
    if signal['rank_ic']<base['minimum_ic'] or signal['ic_coverage']<base['minimum_ic_coverage'] or signal['passing_fold_fraction']<base['minimum_passing_fold_fraction'] or signal['ci95'][0]<=0:reasons.append('frozen_H5_signal_gate_failed')
    if not np.isfinite(row.net_sharpe) or row.net_sharpe<=c['minimum_net_sharpe']:reasons.append('net_sharpe_not_above_1')
    write_json(out/'financial_gate.json',{'primary_method':c['primary_method'],'primary_clock':c['primary_clock'],'financial_passed':not reasons,
        'reasons':reasons,'IC_retrained_or_improved':False,'prospective_blind_test':False,'paper_admitted':False,'orders_sent':0,
        'structural_blocks':['unverified_financial_vintages_and_fixed_survivor_universe','fractional_return_screen_not_stock_cash_ledger']})
    print('COMPLETE',phase,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['development','evaluation','publish']);p.add_argument('--outputs',default='outputs/v10_clock/development');p.add_argument('--reports',default='reports/v10_clock/development');a=p.parse_args()
    if a.action=='publish':
        import shutil
        if Path(a.reports).exists():raise ValueError('Never overwrite evidence')
        shutil.copytree(a.outputs,a.reports)
    else:run(a.action,a.outputs)
if __name__=='__main__':main()
