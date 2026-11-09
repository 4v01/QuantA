"""Complete V3 ledgers/results, with failed scenarios retained and a fixed gate."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ..backtest import statistics,forecast_metrics,bootstrap_sharpe
from ..data import sha256
from ..study import write_json
from .statistics import paired_excess


def case_years(config):return [(y,'development') for y in config['development_years']]+[(config['confirmation_year'],'confirmation')]


def report(config_path='configs/v3.json',outputs='outputs/v3',reports='reports/v3'):
    c=json.loads(Path(config_path).read_text());base=Path(outputs);dest=Path(reports);dest.mkdir(parents=True,exist_ok=True)
    seal=json.loads((base/'champion.json').read_text());years=case_years(c)
    for year,mode in years:
        if not (base/mode/str(year)/'result.json').exists():raise ValueError('Incomplete V3; report unchanged')
    if (dest/'provenance.json').exists():
        old=json.loads((dest/'provenance.json').read_text())
        current_quotes={mode:json.loads((base/mode/'provenance.json').read_text())['data_sha256'] for mode in ['development','confirmation']}
        if old['source_commit']!=seal['source_commit'] or old['config_sha256']!=seal['config_sha256'] or old['corporate_actions_sha256']!=seal['corporate_actions_sha256'] or old['quotes_sha256']!=current_quotes:
            raise ValueError('Different V3 source/data report already exists; do not overwrite')
    evidence=dest/'evidence';evidence.mkdir(exist_ok=True);vdir=dest/'validation_paths';vdir.mkdir(exist_ok=True)
    rows=[];daily=[];forecast=[];pooled={};confirmation={};versions={};order_checks=[];failures=[];model_count=0;policy_count=0
    for year,mode in years:
        root=base/mode;folder=root/str(year);r=json.loads((folder/'result.json').read_text());s=json.loads((folder/'selection.json').read_text())
        if r['provenance']['source_commit']!=seal['source_commit'] or r['provenance']['config_sha256']!=seal['config_sha256']:raise ValueError('Mixed V3 source')
        versions[mode]=json.loads((root/'data_version.json').read_text())
        write_json(evidence/f'{year}-result.json',r);write_json(evidence/f'{year}-selection.json',s)
        for name in ['validation_paths.csv.gz','validation_benchmark.csv']:shutil.copyfile(folder/name,vdir/f'{year}-{name}')
        model_count+=r['forecast_candidates'];policy_count+=r['policy_validation_candidates']
        meta=pd.read_csv(folder/'metadata.csv',parse_dates=['entry_date','exit_date','label_exit']);pred=pd.read_csv(folder/'predictions.csv').to_numpy()
        valid=meta.label_valid&(meta.label_exit<pd.Timestamp(year+1,1,1))
        forecast.append({'year':year,'mode':mode,'return':forecast_metrics(meta.loc[valid,'label_return'],pred[valid,0]),
                         'downside_rmse':float(np.sqrt(np.mean((meta.loc[valid,'label_downside']-pred[valid,1])**2))),
                         'forecast_rows':int(valid.sum()),'account_rows':len(meta)})
        for case in r['scenarios']:
            family,scenario=case['family'],case['scenario'];folder_case=folder/family
            if case['status']!='success':
                failed={'mode':mode,'period':str(year),'family':family,'scenario':scenario,'status':'failed','error':case['error']}
                rows.append(failed);failures.append(failed);continue
            path_name=folder_case/f'{scenario}-path.csv';orders_name=folder_case/f'{scenario}-orders.csv'
            if sha256(path_name)!=case['path_sha256'] or sha256(orders_name)!=case['orders_sha256']:raise ValueError('Changed ledger')
            path=pd.read_csv(path_name);metrics=statistics(path.net_return,path.market_return)
            if scenario=='primary':metrics['sharpe_block95']=bootstrap_sharpe(path.net_return,c['bootstrap_repetitions'],c['bootstrap_block_length'])
            row={'mode':mode,'period':str(year),'family':family,'scenario':scenario,'status':'success',**metrics,
                 'annual_turnover':float(path.turnover.sum()*252/len(path)),'commission_cny':float(path.commission.sum()+path.terminal_commission.sum()),
                 'slippage_cny':float(path.slippage.sum()+path.terminal_slippage.sum()),'solver_fallbacks':int(path.solver_failures_cumulative.max()),
                 'mean_exposure':float(path.gross_exposure.mean()),'end_receivable_cny':float(path.receivable_after.iloc[-1])}
            rows.append(row)
            try:orders=pd.read_csv(orders_name)
            except pd.errors.EmptyDataError:orders=pd.DataFrame()
            if len(orders):
                if (orders.shares_after<0).any() or (orders.cash_after_all_trades< -1e-7).any():raise ValueError('Invalid inventory/cash')
                if not (pd.to_datetime(orders.signal_date)<pd.to_datetime(orders.date)).all():raise ValueError('Same-day signal/fill leakage')
                buy=orders[orders.filled>0]
                if ((buy.filled%c['lot_size'])!=0).any():raise ValueError('Non-lot buy')
            order_checks.append({'year':year,'family':family,'scenario':scenario,'orders_rows':len(orders),
                                 'status':'passed','orders_sha256':case['orders_sha256']})
            path['year']=year;path['mode']=mode;path['family']=family;path['scenario']=scenario;daily.append(path)
            if mode=='development':pooled.setdefault((family,scenario),[]).append(path)
            elif scenario=='primary':confirmation[family]=path
    for family in list(c['policies'])+c['baselines']:
      for scenario in c['scenarios']:
        parts=pooled.get((family,scenario),[])
        if len(parts)!=len(c['development_years']):
            rows.append({'mode':'development','period':'pooled','family':family,'scenario':scenario,'status':'failed','error':'Incomplete annual ledgers; not pooled'})
            continue
        path=pd.concat(parts,ignore_index=True);metrics=statistics(path.net_return,path.market_return)
        if scenario=='primary':metrics['sharpe_block95']=bootstrap_sharpe(path.net_return,c['bootstrap_repetitions'],c['bootstrap_block_length'])
        rows.append({'mode':'development','period':'pooled','family':family,'scenario':scenario,'status':'success',**metrics,
                     'annual_turnover':float(path.turnover.sum()*252/len(path)),
                     'commission_cny':float(path.commission.sum()+path.terminal_commission.sum()),
                     'slippage_cny':float(path.slippage.sum()+path.terminal_slippage.sum()),
                     'solver_fallbacks':int(sum(p.solver_failures_cumulative.max() for p in parts)),
                     'mean_exposure':float(path.gross_exposure.mean())})
    table=pd.DataFrame(rows);table.to_csv(dest/'results.csv',index=False)
    pd.concat(daily,ignore_index=True).to_csv(dest/'daily_paths.csv.gz',index=False,float_format='%.12g',compression={'method':'gzip','mtime':0})
    tests={};family=seal['family'];status='not_evaluable'
    if family in confirmation and all(name in confirmation for name in c['confirmation_controls']):
        for name in c['confirmation_controls']:
            tests[name]=paired_excess(confirmation[family].net_return,confirmation[name].net_return,
                                      c['bootstrap_repetitions'],c['bootstrap_block_length'],len(c['confirmation_controls']))
        sh=statistics(confirmation[family].net_return,confirmation[family].market_return)['sharpe_rf_zero']
        status='passed' if sh is not None and sh>0 and all(t['bonferroni_interval'][0]>0 for t in tests.values()) else 'not_passed'
    write_json(dest/'confirmation_test.json',{'status':status,'family':family,'tests':tests,'strict_prospective_blind_trial':False,
               'scope':'2026-01-01 through 2026-10-30 new-project retrospective check; market/corporate metadata not completely unseen.'})
    write_json(dest/'champion.json',seal);write_json(dest/'forecast_metrics.json',forecast);write_json(dest/'data_versions.json',versions)
    write_json(dest/'order_checks.json',order_checks);write_json(dest/'failed_scenarios.json',failures)
    planned_cases=len(years)*(len(c['policies'])+len(c['baselines']))*len(c['scenarios'])
    if sum(len(json.loads((base/mode/str(year)/'result.json').read_text())['scenarios']) for year,mode in years)!=planned_cases:raise ValueError('Missing planned scenarios')
    prov={'source_commit':seal['source_commit'],'config_sha256':seal['config_sha256'],'corporate_actions_sha256':seal['corporate_actions_sha256'],
          'final_forecast_fits':len(years),'forecast_candidate_fits':model_count,'policy_validation_paths':policy_count,
          'planned_scenarios':planned_cases,'successful_scenarios':len(order_checks),'failed_scenarios':len(failures),
          'quotes_sha256':{k:v['quotes_sha256'] for k,v in versions.items()}}
    write_json(dest/'provenance.json',prov)
    lines=['# QuantA V3：文献指导的组合与股数现金仿真','',
           f"正式源提交：`{seal['source_commit']}`；预封存家族：**{family}**。",'',
           '6个最终Ridge预测拟合，18个预测候选，342条政策验证路径；全部政策复用每年的同一预测。',
           '主场景为虚拟100万元、佣金2bp/最低5元、滑点3bp向不利价格档位取整。不是总费用固定5bp。',
           '现金和无风险日收益都取0；夏普及CAPM α采用这一假设，没有计入真实现金管理利息。',
           '2021–2025为已见开发回测；2026-01-01至2026-10-30为新增回顾性检查，不是前瞻盲测。公司行动及少量网页报价信息此前可见。','',
           '## 主场景全部结果','',
           '| 阶段 | 方法 | 净夏普 | 累计收益 | 折拼接年化收益 | 最大回撤 | 年换手 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for mode,period in [('development','pooled'),('confirmation',str(c['confirmation_year']))]:
        subset=table[(table['mode']==mode)&(table.period==period)&(table.scenario=='primary')]
        for r in subset.to_dict('records'):
            if r['status']!='success':lines.append(f"| {mode} | {r['family']} | 失败 | — | — | — | — |");continue
            sh='未定义' if pd.isna(r['sharpe_rf_zero']) else f"{r['sharpe_rf_zero']:.3f}"
            lines.append(f"| {mode} | {r['family']} | {sh} | {r['cumulative_return']:.2%} | {r['annualized_return']:.2%} | {r['max_drawdown']:.2%} | {r['annual_turnover']:.1f} |")
    lines+=['','legacy_discrete只是同输入/同现金股数账本的V2式控制，不是原V2结果重命名。旧V2报告未变，新旧不同数据定义与费用口径不能直接拼接。',
            '所有方法提前5个持有区间计划清仓，仍计算整段日历；独立年度折重置资金，拼接年化不是连续资金账户的实盘业绩。2026不足全年，年化不能当作2026已赚到的全年收益。','',
            '## 预设检验','',f"结论：**{status}**。三项参照的98.333%配对区块区间下界都为正且净夏普>0才算通过。",'',
            '| 参照 | 年化算术平均超额 | 三项Bonferroni区间 |','|---|---:|---|']
    for name,t in tests.items():
        lo,hi=t['bonferroni_interval'];lines.append(f"| {name} | {t['annualized_mean_excess']:.2%} | [{lo:.2%}, {hi:.2%}] |")
    lines+=['','## 预测解释指标','',
            '| 年度 | 收益RMSE（小数） | 收益R² |','|---:|---:|---:|']
    for f in forecast:lines.append(f"| {f['year']} | {f['return']['rmse']:.6f} | {f['return']['r2']:.6f} |")
    lines+=['','收益标签考虑登记权益、原始价格和拆分，现金不再投；预测R²不能直接与旧复权标签比较。负R²全部保留。','',
            '## 证据与限制','',
            f"计划{planned_cases}个回测场景，成功{len(order_checks)}、失败{len(failures)}；失败原因见failed_scenarios.json，未静默补新参数。",'',
            'results.csv和daily_paths.csv.gz含每年、全部政策、滑点/规模/零费敏感性及求解降级计数。',
            'validation_paths/公开57候选逐日收益，evidence/公开完整预测/政策选择和White式条件诊断；它不覆盖所有既往研究尝试，不作为全局显著性保证。',
            '逐单原始价格、模型参数和行情只在本地快照，公开文件记录各自SHA256。',
            '原始报价历史版本、分红精确公告时间尚未认证；虽然未来事件不重写过去特征，也不能把数据称为完整PIT。',
            '四只存续ETF样本有限；开盘滑点、ADV容量及涨跌停规则仍为仿真近似，没有盘口/集合竞价真实成交保证。',
            '单因子CAPM α相对现金分红经济总回报的沪深300ETF，不是多因子残差α。未通过确认时不改门槛，也不挑另一家族当胜者。','']
    (dest/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/v3.json');p.add_argument('--outputs',default='outputs/v3');p.add_argument('--reports',default='reports/v3');a=p.parse_args();report(a.config,a.outputs,a.reports)
