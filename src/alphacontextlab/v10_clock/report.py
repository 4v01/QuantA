import json
from pathlib import Path
import pandas as pd
import numpy as np
from ..study import write_json
from ..v4.metrics import block_mean_interval,path_statistics
from ..v8.report import sharpe_interval


def run():
    root=Path('reports/v10_clock');c=json.loads(Path('configs/v10_clock.json').read_text());rows=[];attribution={}
    for phase in ['development','evaluation']:
        p=pd.read_csv(root/phase/'paths.csv.gz',parse_dates=['date'],float_precision='round_trip')
        def take(method,clock):return p[(p.method==method)&(p.clock==clock)&(p.cost_bps==20)].copy()
        main=take('economic_h5','stagger5');control=take('economic_h5','phase0');benchmark=take('equal_weight','stagger5')
        comparisons={}
        for name,other in [('phase0',control),('equal_weight_stagger5',benchmark)]:
            merged=main[['date','net_return']].merge(other[['date','net_return']],on='date',suffixes=('_main','_other'))
            delta=merged.net_return_main-merged.net_return_other
            comparisons[name]={'paired_days':len(delta),'mean_daily_return_difference':float(delta.mean()),'daily_return_difference_ci95':block_mean_interval(delta,17,1000,20)}
        gross=main.copy();gross['net_return']=gross.gross_return
        attribution[phase]={'paired_comparisons':comparisons,'net_sharpe_ci95':sharpe_interval(main.net_return),
            'same_holdings_zero_cost_attribution':path_statistics(gross),
            'total_gross_sleeve_turnover':float(main.gross_sleeve_turnover.sum()),'total_external_entry_turnover':float(main.entry_turnover.sum()),
            'netting_reduction':float(main.netting_reduction.sum()),
            'note':'Conditional intervals on already-seen history; no full-research multiple-testing correction. Gross attribution is not a new candidate.'}
        for (m,clock,cost,year),g in p.groupby(['method','clock','cost_bps',p.date.dt.year]):
            rows.append({'phase':phase,'method':m,'clock':clock,'cost_bps':int(cost),'year':int(year),**path_statistics(g),'mean_exposure':float(g.gross_exposure.mean())})
    write_json(root/'attribution.json',attribution);pd.DataFrame(rows).to_csv(root/'annual_results.csv',index=False)
    dev=pd.read_csv(root/'development/summary.csv');evaluation=pd.read_csv(root/'evaluation/summary.csv');common=pd.read_csv(root/'evaluation/common_window_summary.csv')
    def row(frame,method,clock):return frame[(frame.method==method)&(frame.clock==clock)&(frame.cost_bps==20)].iloc[0]
    main=row(evaluation,'economic_h5','stagger5');control=row(evaluation,'economic_h5','phase0');bench=row(evaluation,'equal_weight','stagger5')
    table=['| 固定时钟 | 开发净夏普 | 后段净夏普 | 后段累计净收益 | 后段最大回撤 | 后段年换手 |','|---|---:|---:|---:|---:|---:|']
    for clock in c['clocks']:
        d=row(dev,'economic_h5',clock);e=row(evaluation,'economic_h5',clock)
        table.append(f'| {clock}{"（预先固定主方案）" if clock=="stagger5" else ""} | {d.net_sharpe:.3f} | {e.net_sharpe:.3f} | {e.cumulative_return:.2%} | {e.max_drawdown:.2%} | {e.annual_turnover:.2f} |')
    gate=json.loads((root/'evaluation/financial_gate.json').read_text());signal=json.loads(Path('reports/v9/evaluation/ic_summary.json').read_text())['economic_h5']
    filings=json.loads((root/'filing_progress.json').read_text());sources={p:json.loads((root/p/'provenance.json').read_text())['source_commit'] for p in ['development','evaluation']}
    catalogue=json.loads((root/'filing_archive_catalogue.json').read_text()) if (root/'filing_archive_catalogue.json').exists() else None
    catalogue_note=(f"固定8家公司×2年的原始全文采集队列已归档{catalogue['archived_documents']}/{catalogue['planned_documents']}份，保留当前获取时间、官方索引和文档指纹；逐字段人工核对仍只完成上述两份六项。下载齐全不等于原始数据验收通过，其余字段、单位、口径与修订差异待核。当前发行人目录仅用来定位公告，不作为历史投资股票池。" if catalogue else '')
    text=f'''# 固定V9预测的五组错峰实验

没有重新训练或改动融合权重。主方案在任何本轮结果计算前公开固定为economic_h5/stagger5；全部五个单相位只是诊断，未选择相位赢家。2020–2021和2022–2024均是已见历史，不是新的盲测。

## 主结果

每边20bp，2022–2024：原phase0净夏普 **{control.net_sharpe:.3f}**、累计净收益 **{control.cumulative_return:.2%}**；五组错峰净夏普 **{main.net_sharpe:.3f}**、累计净收益 **{main.cumulative_return:.2%}**。累计结果差{(main.cumulative_return-control.cumulative_return)*100:+.2f}个百分点。等权错峰基准净夏普{bench.net_sharpe:.3f}、累计{bench.cumulative_return:.2%}。

财务门槛 **{gate['financial_passed']}**，原因`{gate['reasons']}`。H5 IC仍是原预测的{signal['rank_ic']:.4f}，覆盖{signal['ic_coverage']:.1%}、半年折达标{signal['passing_fold_fraction']:.1%}，**时钟没有提升模型IC**。模拟盘未启动、订单0。

{chr(10).join(table)}

所有相位、两个固定预测对照、独立等权与10/20/30bp路径均公开。没有用后段最好相位替换主方案。主结果包含首周逐组建仓；事先规定的启动五会话后次窗口中，phase0/错峰净夏普分别为{row(common,'economic_h5','phase0').net_sharpe:.3f}/{row(common,'economic_h5','stagger5').net_sharpe:.3f}，不取代主结果。

## 改善与局限

本段已见历史上，错峰减轻了单一调仓时钟的影响，但距离净夏普>1仍远。主方案平均仓位{main.mean_exposure:.1%}；年换手{main.annual_turnover:.2f}对原方案{control.annual_turnover:.2f}，没有明显节费。合计净额相抵换手为{attribution['evaluation']['netting_reduction']:.12g}；每天通常仅一组调仓，本轮不把持仓时钟变化说成跨组净额节费。

主方案夏普的20会话区块条件区间为{attribution['evaluation']['net_sharpe_ci95']}；与phase0配对日收益差区间为{attribution['evaluation']['paired_comparisons']['phase0']['daily_return_difference_ci95']}。未对整个既往研究的多重试验做校正，不宣称独立发现稳定α。相同持仓零费毛夏普为{attribution['evaluation']['same_holdings_zero_cost_attribution']['net_sharpe']:.3f}，完整归因见attribution.json。

## 资金与费用核验

五组各从20%基金资金的现金开始，每天只调对应组。组预算随自身盈亏漂移，不能每天重置；基金只按合并目标和旧持仓的净权重变化收费。保存逐日组预算、现金、持仓及总账户，核验资金守恒和终点退出。

费用仍是V9比例份额、收益前按全基金同比例承担的研究筛查，不能解释为真实股票现金账本。phase0在三个费用下必须精确重现V9原路径，全部对照和基准也遵守同一费用规则。没有通过平均五条净值或夏普替代合并账户。

## 原始财报进展（与实验隔离）

已核验{filings['documents_checked']}份平安银行2019/2020年报、{filings['fields_checked']}个数值，并通过巨潮公告详情POST接口核对证券代码、公告ID、日期和相同文档路径。

{catalogue_note}

2019原文EPS1.54、普通股每股净资产14.07、加权ROE11.30%；2020为1.40、15.16、9.58%。这六项与当前缓存一致，不能推及所有字段均未修订。公告日期按日期精度记录，不把午夜时间戳当成精确发布时刻，实际获取时间保留为当前。

公告接口的大小字段单位及历史原始字节一致性尚未核实，完整修订链仍缺失。因此这些值留在证据暂存区，**未批准进入训练，也未用于本轮冻结输入**。详见filing_progress.json及原始公告元数据快照；不通过NOTICE_DATE批量回填当前财务表。

## 追溯与下一步

两阶段运行来源为 `{sources['development']}` / `{sources['evaluation']}`。配置逐文件绑定V9预测、市场回放带、IC和原对照路径哈希；所有旧报告按发布前基准核验未改。

时钟修正只解决部分损失，仍需要原始财务版本、历史成分、更广市场状态与收益单位校准。固定幸存者股票池、供应商后复权收益及非可执行账本继续阻断模拟盘。下一步优先扩展数据工程样本并核验完整版本，而非事后挑选单相位或把零费结果作为主成绩。

[事先冻结协议](../../docs/V10_CLOCK_PROTOCOL.md) · [协议指纹](protocol_seal.json) · [复现](../../docs/V10_CLOCK_REPRODUCE.md)

![固定预测时钟比较](evaluation.png)
'''
    (root/'REPORT.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=pd.read_csv(root/'evaluation/paths.csv.gz',parse_dates=['date']);p=p[p.cost_bps==20]
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained',sharex=True)
    phases=p[(p.method=='economic_h5')&p.clock.str.startswith('phase')].pivot(index='date',columns='clock',values='nav')
    axes[0].fill_between(phases.index,phases.min(axis=1),phases.max(axis=1),color='grey',alpha=.15,label='phase0–4 diagnostic range (not a strategy)')
    for method,clock,label in [('economic_h5','phase0','V9 phase0'),('economic_h5','stagger5','fixed stagger5'),('equal_weight','stagger5','equal-weight stagger5')]:
        q=p[(p.method==method)&(p.clock==clock)];axes[0].plot(q.date,q.nav,label=label)
        nav=q.nav.to_numpy();axes[1].plot(q.date,nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1,label=label)
    axes[0].set_title('Frozen V9 forecasts — retrospective 2022–2024, 20bp/side');axes[0].set_ylabel('Net NAV');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Drawdown')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(root/'evaluation.png',dpi=160);fig.savefig(root/'evaluation.svg');plt.close(fig)
    print(json.dumps({'primary_net_sharpe':float(main.net_sharpe),'primary_cumulative_return':float(main.cumulative_return),'financial_passed':gate['financial_passed']}))
if __name__=='__main__':run()
