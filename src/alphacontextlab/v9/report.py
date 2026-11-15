"""Result reporting after sealed evaluation; no model or policy substitution."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..study import write_json
from ..v4.metrics import block_mean_interval,path_statistics
from ..v8.report import sharpe_interval


def run():
    root=Path('reports/v9');config=json.loads(Path('configs/v9.json').read_text());seal=json.loads((root/'champion_seal.json').read_text())
    method,policy=seal['method'],seal['policy'];dev=pd.read_csv(root/'development/screen_summary.csv');ev=pd.read_csv(root/'evaluation/screen_summary.csv')
    signal=json.loads((root/'evaluation/ic_summary.json').read_text());h20=json.loads((root/'evaluation/secondary_H20_ic.json').read_text());metrics=json.loads((root/'evaluation/return_metrics.json').read_text())
    gate=json.loads((root/'evaluation/financial_gate.json').read_text());quality=json.loads((root/'evaluation/coverage.json').read_text());annual=[];attribution={}
    for phase in ['development','evaluation']:
        p=pd.read_csv(root/phase/'screen_paths.csv.gz',parse_dates=['date'],float_precision='round_trip')
        chosen=p[(p.method==method)&(p.policy==policy)&(p.cost_bps==20)]
        benchmark_policy='equal_weekly' if policy=='weekly_buffer' else ('equal_slow' if policy=='slow_buffer' else 'equal_slow_vol')
        benchmark=p[(p.method=='equal_weight')&(p.policy==benchmark_policy)&(p.cost_bps==20)]
        matched=chosen[['date','net_return']].merge(benchmark[['date','net_return']],on='date',suffixes=('_chosen','_benchmark'))
        delta=matched.net_return_chosen-matched.net_return_benchmark;gross=chosen.copy();gross['net_return']=gross.gross_return
        attribution[phase]={'method':method,'policy':policy,'benchmark':benchmark_policy,
            'net_sharpe_ci95':sharpe_interval(chosen.net_return),'paired_daily_return_difference':float(delta.mean()),
            'paired_daily_return_difference_ci95':block_mean_interval(delta,17,1000,20),
            'same_holdings_zero_cost_attribution':path_statistics(gross),
            'note':'Conditional intervals only; no global multiple-testing correction, and benchmark/exposure difference is not certified alpha.'}
        for (m,pol,cost,year),g in p.groupby(['method','policy','cost_bps',p.date.dt.year]):
            annual.append({'phase':phase,'method':m,'policy':pol,'cost_bps':int(cost),'year':int(year),**path_statistics(g),'mean_exposure':float(g.gross_exposure.mean())})
    write_json(root/'attribution.json',attribution);pd.DataFrame(annual).to_csv(root/'annual_results.csv',index=False)
    daily=pd.read_csv(root/'evaluation/daily_ic.csv',parse_dates=['date']);control=daily[(daily.method=='matched_control')&(daily.horizon==5)]
    increments={}
    for m in config['methods']:
        other=daily[(daily.method==m)&(daily.horizon==5)]
        paired=control[['date','rank_ic']].merge(other[['date','rank_ic']],on='date',suffixes=('_control','_other'))
        delta=paired.rank_ic_other-paired.rank_ic_control
        increments[m]={'paired_valid_days':int(delta.notna().sum()),'mean_delta_H5_RankIC':float(delta.mean()),'ci95':block_mean_interval(delta,17,1000,20)}
    write_json(root/'paired_ic_comparisons.json',increments)
    row=ev[(ev.method==method)&(ev.policy==policy)&(ev.cost_bps==20)].iloc[0]
    old=ev[(ev.method=='matched_control')&(ev.policy=='weekly_buffer')&(ev.cost_bps==20)].iloc[0];s=signal[method]
    table=['| 模型 | H5 IC | H5覆盖 | 半年折≥0.02 | H20 IC（次指标） | H20超额收益R² |','|---|---:|---:|---:|---:|---:|']
    for m in config['methods']:
        z=signal[m];table.append(f"| {m} | {z['rank_ic']:.4f} | {z['ic_coverage']:.1%} | {z['passing_fold_fraction']:.1%} | {h20[m]['rank_ic']:.4f} | {metrics[m]['excess_H20_r2']:.5f} |")
    portfolios=['| 模型/政策 | 开发净夏普 | 后段净夏普 | 后段累计净收益 | 最大回撤 | 年换手 | 平均仓位 |','|---|---:|---:|---:|---:|---:|---:|']
    for m,pol in [(m,pol) for m in config['methods'] for pol in config['policies']]+[('equal_weight',pol) for pol in config['benchmarks']]:
        d=dev[(dev.method==m)&(dev.policy==pol)&(dev.cost_bps==20)].iloc[0];e=ev[(ev.method==m)&(ev.policy==pol)&(ev.cost_bps==20)].iloc[0]
        portfolios.append(f'| {m}/{pol}{"（封存）" if (m,pol)==(method,policy) else ""} | {d.net_sharpe:.3f} | {e.net_sharpe:.3f} | {e.cumulative_return:.2%} | {e.max_drawdown:.2%} | {e.annual_turnover:.1f} | {e.mean_exposure:.1%} |')
    sources={phase:json.loads((root/phase/'provenance.json').read_text())['source_commit'] for phase in ['development','evaluation']}
    posthoc=''
    if (root/'group_signal_summary.json').exists():
        groups=json.loads((root/'group_signal_summary.json').read_text())['phases']
        coverage=json.loads((root/'news_coverage_shift.json').read_text())['periods']
        quintiles=json.loads((root/'quintile_summary.json').read_text())['rows']
        qt=[x for x in quintiles if x['phase']=='evaluation' and x['method']==method and x['horizon']==5]
        risk=json.loads((root/'risk_structure.json').read_text())
        lines=['| 分支 | 开发H5 IC | 后段H5 IC |','|---|---:|---:|']
        for name in ['fundamental','price_volume','alternative','fusion']:
            lines.append(f"| {name} | {groups['development'][name]['rank_ic']:.4f} | {groups['evaluation'][name]['rank_ic']:.4f} |")
        bins=['| 预测分位（5最高） | 平均H5毛收益/bp | 相对同日全池超额/bp | 超额95%区间/bp |','|---|---:|---:|---|']
        for z in qt:bins.append(f"| {z['quintile']} | {z['mean_return']*10000:.2f} | {z['mean_excess']*10000:.2f} | [{z['excess_ci95'][0]*10000:.2f}, {z['excess_ci95'][1]*10000:.2f}] |")
        posthoc=f'''## 进一步归因：IC为何没有变成收益

以下是看完评价后的诊断，不用于重选候选、构造新交易规则或声称因果贡献。

{chr(10).join(lines)}

基本面分支信号弱化，价量分支保持，融合未超过价量的IC点估计；不能从此直接推断移除某组就会改善净夏普。按可得日期计，每股每自然日标题强度从{coverage[0]['mean_headlines_per_stock_calendar_day']:.2f}降到{coverage[1]['mean_headlines_per_stock_calendar_day']:.2f}，无标题的股票自然日比例从{coverage[0]['zero_headline_stock_calendar_day_fraction']:.1%}升到{coverage[1]['zero_headline_stock_calendar_day_fraction']:.1%}。覆盖漂移确实存在，但新闻分支IC没有同时下降，故不能把亏损简单归因于新闻来源切换。

{chr(10).join(bins)}

最高分组的平均毛收益幅度很小，最高两组也没有清晰单调优势。全截面正IC可以同时反映识别落后者，而只多策略必须能买到足够强的赢家。分位统计使用全部有效预测日、有重叠H5标签；它不是缓冲持仓结果，不能直接乘频率年化或当可交易α。20bp/边新仓往返费用约40bp，缓冲能降低实际成本，但不能凭这些微弱均值证明覆盖费用。

相同封存持仓即使零费，后段毛收益夏普也只有{attribution['evaluation']['same_holdings_zero_cost_attribution']['net_sharpe']:.3f}。这是收益结构问题，不只是费率。58股的后段收益相关矩阵描述性参与率约{risk['correlation_participation_ratio']:.2f}、第一主成分解释{risk['first_component_variance_fraction']:.1%}，说明股票数量不等于独立风险维度；该参与率不是有效试验数，也不能直接当夏普上限。

下一步应先获得历史成分/原始披露版本、覆盖更多市场状态的逐期预测，再围绕“可交易赢家的收益幅度与风险”校准各分支，而非让每组标准化后天然同幅。还需用跨市场状态的开发区间检验政策选择：本轮2020–2021开发改善未迁移到后段，不能继续只凭该开发期夏普选复杂度。上述方向尚未验证；后段较好的低仓位消融也未被提升为新冠军。

'''
    text=f'''# QuantA V9：公开量化作者核查、经济因子与逐期校准

2026-11-16北京时间。所有区间已见，属于回顾性研究；开发2020–2021，公开封存后才运行2022–2024。原58股、原数据快照与每边20bp主费用保持一致，不用更换后段赢家、降低费用或换主指标达标。

## 结论

预先封存 **{method}/{policy}**，后段净夏普 **{row.net_sharpe:.3f}**、累计净收益 **{row.cumulative_return:.2%}**、最大回撤 **{row.max_drawdown:.2%}**。财务门槛 **{gate['financial_passed']}**，原因 `{gate['reasons']}`。数据/成交结构阻断仍存在，**模拟盘未启动，订单0**。

封存模型H5 RankIC={s['rank_ic']:.4f}，95%区块区间[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]，有效覆盖{s['ic_coverage']:.1%}，半年折达标比例{s['passing_fold_fraction']:.1%}。匹配控制的后段净夏普{old.net_sharpe:.3f}，封存方案点估计差{row.net_sharpe-old.net_sharpe:+.3f}。点估计改善不自动代表统计显著或可执行盈利。

## 核查了哪些公开来源

[Robert Carver](https://github.com/robcarver17)的[履历](https://www.systematicmoney.org/bio)支持其AHL从业背景；其系统现位于[pst-group/pysystemtrade](https://github.com/pst-group/pysystemtrade)。阅读预测缩放与缓冲实现，本轮采用预测/仓位分离及慢频成本对照，没有复制GPL代码或复刻其期货系统。

[Artur Sepp](https://github.com/ArturSepp)的[QuantInvestStrats](https://github.com/ArturSepp/QuantInvestStrats)提供风险、绩效和基准归因实现；本轮核查相关文档，采用明确频率、条件区间及收益/风险分开报告。并不将分析工具说成可直接移植的α。

[Stefan Jansen](https://github.com/stefan-jansen/machine-learning-for-trading)属于技术作者/咨询背景，核查其价值、质量、动量与多期限公司特征案例，采用因子量纲和期限一致性思路。没有复制其市场、长短交易或比赛收益。

[完整核查与身份依据](../../docs/PRACTITIONER_GITHUB_REVIEW.md)及practitioner_sources.json保留三个固定revision、许可和阅读文件哈希；外部代码未被复制/执行。额外数据来源探查使用单独隔离的Baostock客户端，不属于上述项目代码移植，也没有进入V9拟合输入。

## 当前方法的局限与实际修正

1. **短验证窗口导致硬归零。** 恢复6个月有符号选模；将可靠性硬门控作为单独消融。原V7 matched_control的逐股预测与周频缓冲收益路径均由审计校验，不靠换弱基线。
2. **因子量纲。** 从每股EPS/BPS水平改为价格相对估值代理，另加年度ROE、现金盈利比、报表年龄及长期动量/风险特征。TTM EPS仅使用当日已知的YTD、上年全年、上年同期分量，缺失保持缺失；股份口径/拆并股匹配未经认证，不称完整PIT。
3. **校准占用选模历史。** 用2019起严格逐期预测形成过去12个月已成熟H20标签校准集，保证每条校准预测来自当时已经冻结的模型；不再牺牲完整选模窗口，也不对负斜率翻向。
4. **H5优势与成本错配。** 保留H5主指标，增加H20训练、20交易日持有和H20风险/收益配置消融。没有将次指标H20冒充主指标。

四个学习模型仍用30/40/30名义信号权重。classic_value_quality是明确独立的50%基本面/50%价量基准，不是静默更改融合权重。经济特征以一组加入，未隔离到单一特征的因果贡献；任何收益差不能单独归功于EPS/价格或某位作者。

## 更底层的数据问题

原财务快照中，公告2018–2024记录的UPDATE_DATE相对NOTICE_DATE滞后中位数 **{quality['revision_delay_days_median']:.0f}天**，超过180天比例 **{quality['revision_delay_above_180_days_fraction']:.1%}**。当前输入面板可得报表的报告期年龄中位数 **{quality['known_report_age_days_median']:.0f}天**，P90为{quality['known_report_age_days_p90']:.0f}天。年度ROE缺失 **{quality['economic_feature_missing_fraction']['roe_annual']:.1%}**。

这说明“当前API历史表”不等于“当时披露版本”。本轮加入年龄特征和缺失指示，但**没有修复原始版本缺失**，也没有把修订后数值按更早公告回填。PIT与股份口径仍是数据阻断。可选参考[Qlib PIT采集器](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/pit)探查Baostock历史成分/财务来源；本环境匿名登录发生网络接收错误，回执data_source_probe.json为blocked，没有导入任何新数据。即使可连通，pubDate仍需与原始公告/修订历史核验。

UPDATE_DATE滞后本身不证明每个字段都改过；它表示当前表不足以认证更早的逐字段可得值。

## 后段信号与可解释性

{chr(10).join(table)}

R²按H20截面去均值收益的1-SSE/SST计算，明确不是绝对收益R²、不是IC²。每条预测保留两期限标签、模型指纹及校准窗口。

## 同费用全部组合结果

{chr(10).join(portfolios)}

所有数字主费用为20bp/边；10/30bp全路径、逐年、持仓与费用均公开。空仓零波动夏普为undefined，不填成0或成功。等权基准独立于α分数强弱；slow_net_edge采用风险惩罚，并非与equal_slow_vol完全同风险，基准差不可直接解释为纯α。

封存政策后段夏普条件20日区块区间为{attribution['evaluation']['net_sharpe_ci95']}；相对所列基准的日均收益差区间为{attribution['evaluation']['paired_daily_return_difference_ci95']}。全部研究历史未做多重试验校正，不据此单独声称显著。相同持仓零费用归因不参与选模或准入。

{posthoc}
## 工程修复、追溯与已知问题

首次开发缺预测日误清仓，使原对照路径偏离V7；在任何V9候选封存与后段运行前修复为全模型/基准统一保持原仓位，完整重跑。原始尝试和未封存候选保留在development_attempt1_correction.json及本地证据包；不是隐藏失败或修改费用/模型选优。新增审计要求V7对照路径数值一致。

开发来源 `{sources['development']}`；封存/后段来源 `{sources['evaluation']}`。公开预测含2019预热，指标与开发选优仅2020–2021。60组checkpoint与逐期标签成熟规则可重放。

固定幸存者股票池、财务修订和股份口径、历史新闻来源切换、比例后复权收益缺股数现金撮合认证仍存在。市场方向、行业集中、只有58股的横截面宽度与重复用历史选优，仍限制泛化判断。任何财务门槛结论都不自动解除这些阻断。

[冻结协议](../../docs/V9_PROTOCOL.md) · [候选封存](champion_seal.json) · [复现](../../docs/V9_REPRODUCE.md)

![V9后段回顾性评价](evaluation.png)
'''
    (root/'REPORT.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=pd.read_csv(root/'evaluation/screen_paths.csv.gz',parse_dates=['date']);fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained',sharex=True)
    bench=attribution['evaluation']['benchmark']
    for m,pol,label in [(method,policy,'sealed: '+method+'/'+policy),('matched_control','weekly_buffer','V7 matched control'),('equal_weight',bench,'benchmark: '+bench)]:
        path=p[(p.method==m)&(p.policy==pol)&(p.cost_bps==20)];axes[0].plot(path.date,path.nav,label=label)
        nav=path.nav.to_numpy();drawdown=nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1
        axes[1].plot(path.date,drawdown,label=label,alpha=.8)
    axes[0].set_title('QuantA V9 — retrospective 2022–2024, 20bp per side');axes[0].set_ylabel('Net NAV');axes[0].legend(fontsize=8);axes[0].axhline(1,color='grey',lw=.6)
    axes[1].set_ylabel('Drawdown');axes[1].set_ylim(-.45,.02)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(root/'evaluation.png',dpi=160);fig.savefig(root/'evaluation.svg');plt.close(fig)
    print(json.dumps({'method':method,'policy':policy,'financial_passed':gate['financial_passed'],'net_sharpe':float(row.net_sharpe)}),flush=True)
if __name__=='__main__':run()
