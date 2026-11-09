"""Complete, separately labeled discovery/confirmation tables; no model selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..backtest import statistics,bootstrap_sharpe,forecast_metrics
from ..study import write_json
from .execution import simulate
from .study import seeds


def confirmation_test(a,b,repetitions,block,seed=2026):
    a=np.asarray(a);b=np.asarray(b);rng=np.random.default_rng(seed);d=a-b;sampled=[]
    for _ in range(repetitions):
        start=rng.integers(0,len(d),size=int(np.ceil(len(d)/block)))
        ids=((start[:,None]+np.arange(block))%len(d)).reshape(-1)[:len(d)]
        sampled.append(float(d[ids].mean()*252))
    return {'annualized_arithmetic_excess_mean':float(d.mean()*252),
            'paired_block_bonferroni97_5':np.quantile(sampled,[.0125,.9875]).tolist(),
            'block_length':block,'bootstrap_repetitions':repetitions,'family_comparisons':2}


def report(config_path='configs/v2.json',outputs='outputs/v2',reports='reports/v2'):
    config=json.loads(Path(config_path).read_text());base=Path(outputs);dest=Path(reports);dest.mkdir(parents=True,exist_ok=True)
    seal=json.loads((base/'champion.json').read_text())
    year_modes=[(y,'development') for y in config['development_years']]+[(config['confirmation_year'],'confirmation')]
    for year,mode in year_modes:
        for arm in config['arms']:
            for seed in seeds(arm,config):
                if not (base/mode/f'{year}-{arm}-{seed}'/'result.json').exists():raise ValueError('Incomplete V2 evidence; report unchanged')
    if (dest/'provenance.json').exists():
        old=json.loads((dest/'provenance.json').read_text())
        current_data={mode:json.loads((base/mode/'provenance.json').read_text())['data_sha256'] for mode in ['development','confirmation']}
        if old['source_commit']!=seal['source_commit'] or old['config_sha256']!=seal['config_sha256'] or old['data_sha256']!=current_data:
            raise ValueError('Refusing to overwrite a different V2 report version')
    write_json(dest/'champion.json',seal)
    table=[];paths=[];forecasts=[];all_evidence=[];pooled={};confirmation_paths={};data_versions={}
    for year,mode in year_modes:
        root=base/mode;prov=json.loads((root/'provenance.json').read_text())
        if prov['source_commit']!=seal['source_commit'] or prov['config_sha256']!=seal['config_sha256']:raise ValueError('Unsealed code/config')
        data_versions[mode]=json.loads((root/'data_manifest.json').read_text())
        meta=pd.read_csv(root/f'{year}-test_metadata.csv',dtype={'symbol':str},parse_dates=['date','entry_date','exit_date','label_exit'])
        corr=np.load(root/'correlations.npy');prediction_map={}
        for arm in config['arms']:
            for seed in seeds(arm,config):
                directory=root/f'{year}-{arm}-{seed}'
                result=json.loads((directory/'result.json').read_text());selection=json.loads((directory/'selection.json').read_text())
                all_evidence.append((directory,result,selection))
                prediction=pd.read_csv(directory/'test_predictions.csv').to_numpy();prediction_map[(arm,seed)]=prediction
                valid=meta.label_valid&(meta.label_exit<pd.Timestamp(year+1,1,1))
                metrics=forecast_metrics(meta.loc[valid,'label_return'],prediction[valid,0])
                ranks=[]
                tmp=meta.loc[valid,['entry_date','symbol','label_return']].copy();tmp['prediction']=prediction[valid,0]
                for _,g in tmp.groupby('entry_date',sort=False):
                    if len(g)>=3 and g.prediction.std()>0 and g.label_return.std()>0:
                        ranks.append(float(np.corrcoef(rankdata(g.prediction),rankdata(g.label_return))[0,1]))
                forecasts.append({'mode':mode,'year':year,'arm':arm,'seed':seed,'return':metrics,
                                  'daily_cross_section_rank_ic_mean':float(np.mean(ranks)) if ranks else None,
                                  'rank_ic_dates':len(ranks),'forecast_rows':int(valid.sum()),'portfolio_rows':len(meta),
                                  'downside_rmse':float(np.sqrt(np.mean((prediction[valid,1]-meta.loc[valid,'label_downside'])**2)))})
        prediction_map[('champion_ensemble',-1)]=np.mean([prediction_map[(seal['arm'],s)] for s in seeds(seal['arm'],config)],axis=0)
        cases=[(arm,seed,profile,'model',prediction) for (arm,seed),prediction in prediction_map.items() for profile in config['profiles']]
        dummy=np.zeros((len(meta),2))
        cases += [(policy,-1,config['primary_profile'],policy,dummy) for policy in ['cash','csi300','equal_buy_hold','inverse_volatility','momentum60']]
        for arm,seed,profile,policy,prediction in cases:
            for cost in config['cost_bps']:
                path=simulate(meta,prediction,corr,config,profile,cost,policy)
                key=(arm,seed,profile,cost)
                values=statistics(path.net_return,path.market_return)
                if profile==config['primary_profile'] and cost==config['primary_cost_bps']:
                    values['sharpe_block95']=bootstrap_sharpe(path.net_return,config['bootstrap_repetitions'],config['bootstrap_block_length'])
                    if mode=='confirmation':confirmation_paths[arm,seed]=path
                table.append({'mode':mode,'period':str(year),'arm':arm,'seed':seed,'profile':profile,'cost_bps':cost,
                              **values,'annual_turnover':float(path.turnover.sum()*252/len(path)),
                              'mean_gross_exposure':float(path.gross_exposure.mean()),
                              'blocked_buys':int(path.blocked_buys.sum()),'blocked_sells':int(path.blocked_sells.sum())})
                path['year']=year;path['mode']=mode;path['arm']=arm;path['seed']=seed;path['profile']=profile;path['cost_bps']=cost
                paths.append(path)
                if mode=='development':pooled.setdefault(key,[]).append(path)
    expected=(len(config['development_years'])+1)*sum(len(seeds(arm,config)) for arm in config['arms'])
    if len(all_evidence)!=expected:raise ValueError('Incomplete V2 fits')
    for (arm,seed,profile,cost),group in pooled.items():
        joined=pd.concat(group,ignore_index=True);values=statistics(joined.net_return,joined.market_return)
        if profile==config['primary_profile'] and cost==config['primary_cost_bps']:
            values['sharpe_block95']=bootstrap_sharpe(joined.net_return,config['bootstrap_repetitions'],config['bootstrap_block_length'])
        table.append({'mode':'development','period':'pooled','arm':arm,'seed':seed,'profile':profile,'cost_bps':cost,**values,
                      'annual_turnover':float(joined.turnover.sum()*252/len(joined)),
                      'mean_gross_exposure':float(joined.gross_exposure.mean()),'blocked_buys':int(joined.blocked_buys.sum()),
                      'blocked_sells':int(joined.blocked_sells.sum())})
    df=pd.DataFrame(table);df.to_csv(dest/'results.csv',index=False)
    daily=pd.concat(paths,ignore_index=True);daily.to_csv(dest/'daily_paths.csv.gz',index=False,float_format='%.12g',compression={'method':'gzip','mtime':0})
    champion=confirmation_paths['champion_ensemble',-1]
    tests={name:confirmation_test(champion.net_return,confirmation_paths[name,-1].net_return,
                                 config['bootstrap_repetitions'],config['bootstrap_block_length']) for name in ['csi300','inverse_volatility']}
    champion_stats=statistics(champion.net_return,champion.market_return)
    passed=champion_stats['sharpe_rf_zero'] is not None and champion_stats['sharpe_rf_zero']>0 and all(v['paired_block_bonferroni97_5'][0]>0 for v in tests.values())
    write_json(dest/'confirmation_test.json',{'status':'passed' if passed else 'not_passed','primary_champion':seal['arm'],
                  'rule':'positive net Sharpe AND positive lower excess-return bound versus both predeclared benchmarks',
                  'tests':tests,'scope':'Retrospective uninspected-2025 confirmation, not prospective live evaluation; revised data limitations remain.'})
    write_json(dest/'forecast_metrics.json',forecasts);write_json(dest/'data_versions.json',data_versions)
    evidence=dest/'evidence';evidence.mkdir(exist_ok=True)
    for directory,result,selection in all_evidence:
        write_json(evidence/(directory.name+'-result.json'),result);write_json(evidence/(directory.name+'-selection.json'),selection)
    for year,mode in year_modes:
        write_json(evidence/f'{year}-factors.json',json.loads((base/mode/f'{year}-factor_search.json').read_text()))
    write_json(dest/'provenance.json',{'source_commit':seal['source_commit'],'config_sha256':seal['config_sha256'],
                  'final_fits':len(all_evidence),'candidate_fits':sum(len(s['candidates']) for _,_,s in all_evidence),
                  'failed_candidates':sum(c['status']=='failed' for _,_,s in all_evidence for c in s['candidates']),
                  'data_sha256':{k:v['quotes_sha256'] for k,v in data_versions.items()}})
    primary=df[(df.profile==config['primary_profile'])&(df.cost_bps==config['primary_cost_bps'])]
    lines=['# QuantA V2：假设研究、成本感知排序与独立留出检查','',
           'Grace CogAlpha/XALPHA 与 X 开源推荐流水线启发下的小规模独立实现；不是论文或生产系统复现。',
           f"正式源提交：`{seal['source_commit']}`；预先封存冠军：**{seal['arm']}**。",'',
           '10日开盘到开盘预测，主要每5个市场交易日调仓；单边5bp，无风险/现金收益为0。',
           '2021–2024为已参与研究设计的开发回测；2025在冠军封存后才下载，不能混为一个调参测试集。',
           '## 主协议结果','',
           '| 阶段 | 方法 | 种子数 | 净夏普均值 | 年化收益均值 | 最大回撤均值 | CAPM α年化均值 | 年换手 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for mode,period in [('development','pooled'),('confirmation',str(config['confirmation_year']))]:
        subset=primary[(primary['mode']==mode)&(primary.period==period)]
        for arm,g in subset.groupby('arm',sort=False):
            sh='未定义' if g.sharpe_rf_zero.isna().all() else f'{g.sharpe_rf_zero.mean():.3f}'
            lines.append(f'| {mode} | {arm} | {len(g)} | {sh} | {g.annualized_return.mean():.2%} | {g.max_drawdown.mean():.2%} | {g.capm_alpha_annual_arithmetic.mean():.2%} | {g.annual_turnover.mean():.1f} |')
    lines+=['','champion_ensemble 使用事前选定类别的全部预声明种子预测均值后决策；不是挑选测试表现最好的种子。',
            '其他神经方法的表格均值仅作描述，不代表可直接交易的种子组合。',
            'CAPM α为相对沪深300ETF、零无风险假设的单因子日截距×252，不能等同于多因子残差α或实盘盈利。','',
            '## 预设确认检验','',f"结论：**{'通过' if passed else '未通过'}**。正净夏普且对沪深300与逆波动参照两项年度均值超额收益区间下界均为正，才算通过。",'',
            '| 参照 | 年化算术平均超额 | 配对区块97.5%区间（两项Bonferroni） |','|---|---:|---|']
    for name,v in tests.items():
        lo,hi=v['paired_block_bonferroni97_5'];lines.append(f"| {name} | {v['annualized_arithmetic_excess_mean']:.2%} | [{lo:.2%}, {hi:.2%}] |")
    lines+=['','## 消融和复现','',
            'results.csv 包含全部年度、种子、0/5/10/20bp以及每日纯收益、每周纯收益、每周加风险、每周加风险和成本四个对照；模型预测完全复用，不按测试重新拟合。',
            'forecast_metrics.json 分别列出10日收益预测RMSE/R²、下行风险误差及逐日横截面RankIC；它们不能与V1单日预测指标或CSI300个股论文指标直接比较。',
            'evidence/ 保存全部候选验证分数、早停、训练/重拟合因子池、冗余拒绝、代码与参数指纹。每日全部成本/消融路径保存在daily_paths.csv.gz。',
            '决策消融的超参数先按主协议验证效用选定，再固定预测；不是每种策略分别调到最好的比较。','',
            '## 冠军类别的预测解释指标','',
            '| 年度 | 种子数 | 收益RMSE均值（小数） | 收益R²均值 | 日横截面RankIC均值 |',
            '|---:|---:|---:|---:|---:|']
    for year,_ in year_modes:
        selected=[r for r in forecasts if r['year']==year and r['arm']==seal['arm']]
        lines.append(f"| {year} | {len(selected)} | {np.mean([r['return']['rmse'] for r in selected]):.6f} | {np.mean([r['return']['r2'] for r in selected]):.6f} | {np.mean([r['daily_cross_section_rank_ic_mean'] for r in selected]):.6f} |")
    lines+=['','预测R²在开发年份均为负、2025也仅略大于零；较好的组合点估计不能替代对预测解释力与稳定性的检查。横截面只有4只ETF且标签重叠，不能与大股票池论文IC直接对标。','',
            '## 限制','',
            '- 仍是固定四只存续ETF和第三方当前前复权日线，不是逐时点完整股票市场数据；没有Level-2或逐笔交易，不能冒充MicroTSM。',
            '- 仍为连续名义权重、年度清仓重置及开盘成交近似；未解决整手、最低佣金、开盘深度和精确除权涨跌停价。',
            '- 因子搜索是有限白名单模板与训练内统计选择，没有运行LLM代理或无限代码搜索。X只借鉴候选/打分/过滤/持仓多样性思想，金融标签与目标全部重建。',
            '- 即使正收益或相对α，也不能自动宣称可部署。确认样本只有一个年度，bootstrap/HAC不涵盖模型搜索和所有数据修订风险。','']
    (dest/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/v2.json');p.add_argument('--outputs',default='outputs/v2');p.add_argument('--reports',default='reports/v2');a=p.parse_args();report(a.config,a.outputs,a.reports)
