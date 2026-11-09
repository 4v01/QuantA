"""Frozen validation-only selection, clean-commit final refits and traceable reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve
import torch

from .data import FEATURES, Transform, fetch, fold_indices, panel, sequences, sha256
from .models import make_model, parameter_count, predict, train
from .backtest import simulate, statistics, forecast_metrics, bootstrap_sharpe


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def provenance(config_path,data_manifest):
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if dirty:raise RuntimeError('Commit all source/config changes before experiments: '+dirty)
    return {'source_commit':source,'source_dirty':False,'config_sha256':sha256(config_path),
            'data_sha256':data_manifest['quotes_sha256'],'python':platform.python_version(),
            'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'torch':torch.__version__,
            'device':'cpu','threads':torch.get_num_threads()}


def ridge_fit(x,y,alpha):
    # Last observation and temporal mean; explicit intercept, train-centered targets.
    z=np.concatenate([x[:,-1,:],x.mean(1)],axis=1).astype(np.float64)
    mean=z.mean(0);target_mean=float(y.mean());zc=z-mean
    coef=solve(zc.T@zc+alpha*np.eye(z.shape[1]),zc.T@(y-target_mean),assume_a='pos')
    return coef,mean,target_mean


def ridge_predict(state,x):
    coef,mean,target_mean=state
    z=np.concatenate([x[:,-1,:],x.mean(1)],axis=1)
    return (z-mean)@coef+target_mean


def run(config_path,data_root,output_root):
    config=json.loads(Path(config_path).read_text());torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    frame,manifest=panel(data_root,config);x,meta=sequences(frame,config)
    prov=provenance(config_path,manifest);root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    write_json(root/'provenance.json',prov)
    for year in config['test_years']:
        train_ids,val_ids,test_ids=fold_indices(meta,year);refit_ids=np.r_[train_ids,val_ids]
        # Include every labeled interval available before test, including old train/val boundary purge.
        refit_ids=np.flatnonzero((meta.exit_date<pd.Timestamp(year,1,1)).to_numpy())
        transform=Transform().fit(x[train_ids]);tx,tt=transform.apply(x[train_ids]);vx,vt=transform.apply(x[val_ids])
        refit_transform=Transform().fit(x[refit_ids]);rx,rt=refit_transform.apply(x[refit_ids]);qx,qt=refit_transform.apply(x[test_ids])
        y=meta.target.to_numpy();test_meta=meta.iloc[test_ids].reset_index(drop=True)
        test_meta.to_csv(root/f'{year}-metadata.csv',index=False)
        split={'train_rows':len(train_ids),'validation_rows':len(val_ids),'refit_rows':len(refit_ids),'test_rows':len(test_ids),
               'train_last_exit':str(meta.iloc[train_ids].exit_date.max()),'validation_first_entry':str(meta.iloc[val_ids].entry_date.min()),
               'validation_last_exit':str(meta.iloc[val_ids].exit_date.max()),'test_first_entry':str(meta.iloc[test_ids].entry_date.min()),
               'test_last_exit':str(meta.iloc[test_ids].exit_date.max()),'training_transform':transform.state(),
               'refit_transform':refit_transform.state()}
        for arm in ['ridge','lstm','lstm_capacity','lstm_memory']:
            seeds=[config['seeds'][0]] if arm=='ridge' else config['seeds']
            for seed in seeds:
                destination=root/f'{year}-{arm}-{seed}';destination.mkdir(exist_ok=True)
                if (destination/'result.json').exists():
                    previous=json.loads((destination/'result.json').read_text())
                    if previous['provenance']!=prov:raise RuntimeError('Refusing resume under different provenance')
                    print('resume',destination.name,flush=True);continue
                started=time.time();candidates=[]
                try:
                    if arm=='ridge':
                        for alpha in config['ridge_candidates']:
                            state=ridge_fit(tx,y[train_ids],alpha);pred=ridge_predict(state,vx)
                            candidates.append({'alpha':alpha,'validation_rmse':forecast_metrics(y[val_ids],pred)['rmse']})
                        winner=min(candidates,key=lambda c:c['validation_rmse'])
                        write_json(destination/'selection.json',{'selected':winner,'candidates':candidates,'split':split,'provenance':prov})
                        state=ridge_fit(rx,y[refit_ids],winner['alpha']);prediction=ridge_predict(state,qx)
                        np.savez(destination/'model.npz',coef=state[0],mean=state[1],target_mean=state[2])
                        params=len(state[0])+1
                    else:
                        for candidate in config['neural_candidates']:
                            model=make_model(arm,len(FEATURES),config,seed)
                            history,best=train(model,tx,tt,y[train_ids],config,candidate,seed,(vx,vt,y[val_ids]))
                            candidates.append({'config':candidate,'history':history,'best':best,'validation_rmse':best['validation_rmse']})
                        winner=min(candidates,key=lambda c:c['validation_rmse'])
                        write_json(destination/'selection.json',{'selected':winner,'candidates':candidates,'split':split,'provenance':prov})
                        model=make_model(arm,len(FEATURES),config,seed);params=parameter_count(model)
                        history,_=train(model,rx,rt,y[refit_ids],config,winner['config'],seed,epochs=winner['best']['epoch'])
                        torch.save({'state_dict':model.state_dict(),'config':config,'arm':arm,'seed':seed,
                                    'transform':refit_transform.state(),'provenance':prov,'refit_history':history},destination/'model.pt')
                        prediction=predict(model,qx,qt)/100
                    checkpoint=destination/('model.npz' if arm=='ridge' else 'model.pt')
                    pd.DataFrame({'prediction':prediction}).to_csv(destination/'predictions.csv',index=False,float_format='%.12g')
                    result={'arm':arm,'seed':seed,'test_year':year,'parameters':params,'elapsed_seconds':time.time()-started,
                            'forecast_metrics':forecast_metrics(y[test_ids],prediction),'checkpoint_sha256':sha256(checkpoint),
                            'predictions_sha256':sha256(destination/'predictions.csv'),'provenance':prov}
                    write_json(destination/'result.json',result)
                    print(json.dumps({k:result[k] for k in ['arm','seed','test_year','elapsed_seconds','forecast_metrics']}),flush=True)
                except Exception as exc:
                    write_json(destination/'failure.json',{'error':repr(exc),'elapsed_seconds':time.time()-started,'provenance':prov})
                    raise


def report(config_path,output_root,report_root):
    config=json.loads(Path(config_path).read_text());root=Path(output_root);reports=Path(report_root);reports.mkdir(parents=True,exist_ok=True)
    # Completeness is mandatory. Never replace a complete report with partial output.
    expected=len(config['test_years'])*(1+3*len(config['seeds']))
    results=list(root.glob('*/result.json'))
    if len(results)!=expected:raise ValueError(f'Incomplete study: {len(results)}/{expected}')
    rows=[];forecast=[];traces=[]
    policies=[('cash',None),('csi300_buy_hold',None),('equal_buy_hold',None),('momentum20',None),('ridge',config['seeds'][0])]
    policies += [(arm,seed) for arm in ['lstm','lstm_capacity','lstm_memory'] for seed in config['seeds']]
    for name,seed in policies:
        for bps in config['cost_bps']:
            paths=[];markets=[]
            for year in config['test_years']:
                meta=pd.read_csv(root/f'{year}-metadata.csv',dtype={'symbol':str},parse_dates=['date','entry_date','exit_date'])
                if seed is None:
                    score=meta.ret20.to_numpy() if name=='momentum20' else np.zeros(len(meta))
                else:
                    directory=root/f'{year}-{name}-{seed}';score=pd.read_csv(directory/'predictions.csv').prediction.to_numpy()
                    record=json.loads((directory/'result.json').read_text())
                    if sha256(directory/'predictions.csv')!=record['predictions_sha256']:raise ValueError('Prediction hash mismatch')
                    if bps==config['primary_cost_bps']:forecast.append(record)
                policy=name if name in ['cash','csi300_buy_hold','equal_buy_hold'] else 'model'
                path=simulate(meta,score,config['symbols'],bps,policy);path['test_year']=year
                market=meta[meta.symbol=='510300'].sort_values('entry_date').target.to_numpy()
                metrics=statistics(path.net_return,market,config['annualization'])
                rows.append({'method':name,'seed':seed,'cost_bps':bps,'period':str(year),**metrics,
                             'annual_turnover':float(path.turnover.sum()*252/len(path)),'blocked_buys':int(path.blocked_buys.sum()),
                             'blocked_sells':int(path.blocked_sells.sum())})
                paths.append(path);markets.append(market)
            joined=pd.concat(paths,ignore_index=True);market=np.concatenate(markets)
            metrics=statistics(joined.net_return,market,config['annualization'])
            metrics['sharpe_block_bootstrap95']=bootstrap_sharpe(joined.net_return,config['bootstrap_repetitions'],config['bootstrap_block_length'])
            rows.append({'method':name,'seed':seed,'cost_bps':bps,'period':'pooled',**metrics,
                         'annual_turnover':float(joined.turnover.sum()*252/len(joined)),
                         'blocked_buys':int(joined.blocked_buys.sum()),'blocked_sells':int(joined.blocked_sells.sum())})
            if bps==config['primary_cost_bps']:
                joined['method']=name;joined['seed']=seed;traces.append(joined)
    table=pd.DataFrame(rows);table.to_csv(reports/'backtest_results.csv',index=False)
    pd.concat(traces,ignore_index=True).to_csv(reports/'daily_net_returns.csv',index=False,float_format='%.12g')
    write_json(reports/'forecast_results.json',forecast)
    write_json(reports/'provenance.json',json.loads((root/'provenance.json').read_text()))
    summaries=[]
    for method,group in table[(table.period=='pooled')&(table.cost_bps==config['primary_cost_bps'])].groupby('method',sort=False):
        summaries.append({'method':method,'seeds':len(group),'sharpe_mean':float(group.sharpe_rf_zero.mean()) if group.sharpe_rf_zero.notna().any() else None,
                          'sharpe_seed_std':float(group.sharpe_rf_zero.std(ddof=1)) if len(group)>1 else None,
                          'annual_return_mean':float(group.annualized_return.mean()),'max_drawdown_mean':float(group.max_drawdown.mean()),
                          'alpha_annual_mean':float(group.capm_alpha_annual_arithmetic.mean()),'annual_turnover_mean':float(group.annual_turnover.mean())})
    write_json(reports/'summary.json',summaries)
    lines=['# A 股 ETF 日频研究：首版真实结果','',
           '回顾性、探索性实验；无券商接口。四只固定存续 ETF，不代表全 A 股选股。',
           '每年重新选择参数并从头拟合；只按上一年度验证 RMSE 选择配置和轮数。测试收益没有参与模型或交易规则选择。',
           '主成本为单边成交额 5bp（佣金与滑点合计假设），现金日收益/无风险收益均为零。',
           '2021–2024 年年度折独立建仓并在最后可用开盘清仓，合并指标按实际评分区间计时，跨年未评分边界日不计入；并非连续实盘账户。','',
           '| 方法 | 种子数 | 净夏普均值 | 年化收益均值 | 最大回撤均值 | CAPM α 年化均值 | 年换手 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        sh='未定义' if s['sharpe_mean'] is None else f"{s['sharpe_mean']:.3f}"
        lines.append(f"| {s['method']} | {s['seeds']} | {sh} | {s['annual_return_mean']:.2%} | {s['max_drawdown_mean']:.2%} | {s['alpha_annual_mean']:.2%} | {s['annual_turnover_mean']:.1f} |")
    lines+=['','α 为净策略日收益对沪深300 ETF 同期复权开盘收益的单因子回归截距 ×252，不是多因子残差 α。',
            '逐种子、逐年度、0/5/10/20bp 成本敏感性、HAC α 区间和夏普区块 bootstrap 区间见 backtest_results.csv。',
            '多个方法/种子/年度比较未经多重检验校正；不得仅挑最高夏普宣称发现 α。',
            '预测 RMSE、R² 与相关系数见 forecast_results.json；预测 R² 与 CAPM 回归 R² 含义不同。','',
            '## 解释边界','',
            '- 数据来自第三方当前前复权日线，非逐时点复权档案；小数舍入及事后复权修订会影响历史特征。',
            '- 固定 ETF 篮子有存续及研究者选样偏差，没有历史全市场/退市股样本。',
            '- 连续名义权重模拟不含100份整手、最低佣金、订单簿、冲击与集合竞价真实成交保证。',
            '- 每日只在开盘调仓一次，新增持仓最早次日卖出，满足模型层面的 T+1；涨跌停以开盘价与前收的复权比值保守过滤，并非交易所精确涨跌停价。',
            '- 不存在实际融资、做空、实盘交易或保证收益。','',
            '## 复现','',f"代码提交：`{json.loads((root/'provenance.json').read_text())['source_commit']}`。",
            '每个最终拟合的配置、分区边界、预处理、候选历史、随机种子、检查点及预测 SHA256 已归档于 evidence/。',
            '原始行情及学习参数不上传；行情指纹见 data_manifest.json，下载后指纹不同时必须标记为新数据版本。']
    (reports/'REPORT.md').write_text('\n'.join(lines)+'\n')
    evidence=reports/'evidence';evidence.mkdir(exist_ok=True)
    for path in results:
        write_json(evidence/f'{path.parent.name}-result.json',json.loads(path.read_text()))
        selection=path.parent/'selection.json'
        write_json(evidence/f'{path.parent.name}-selection.json',json.loads(selection.read_text()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['fetch','run','report'])
    parser.add_argument('--config',default='configs/study.json');parser.add_argument('--data',default='data');parser.add_argument('--outputs',default='outputs');parser.add_argument('--reports',default='reports/v1')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if args.action=='fetch':
        result=fetch(config,args.data);print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.action=='run':run(args.config,args.data,args.outputs)
    else:report(args.config,args.outputs,args.reports)


if __name__=='__main__':main()
