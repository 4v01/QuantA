"""Post-evaluation matched comparisons and cost attribution; no retuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..study import write_json
from ..v4.metrics import path_statistics,paired_ic_increment


def generate(root='reports/v7'):
    r=Path(root);c=json.loads(Path('configs/v7.json').read_text());seal=json.loads((r/'champion_seal.json').read_text());champion=seal['method']
    dev=json.loads((r/'development/ic_summary.json').read_text());test=json.loads((r/'holdout/ic_summary.json').read_text());gate=json.loads((r/'holdout/financial_gate.json').read_text())
    ic=pd.read_csv(r/'holdout/daily_ic.csv',parse_dates=['date']);ic=ic[ic.group=='fusion'];paths=pd.read_csv(r/'holdout/screen_paths.csv.gz',parse_dates=['date','signal_date']);table=pd.read_csv(r/'holdout/screen_summary.csv')
    comparisons={m:paired_ic_increment(ic[ic.method=='legacy_matched'],ic[ic.method==m],{'seed':c['seed'],'signal_gate':{'bootstrap_repetitions':c['bootstrap_repetitions']}}) for m in c['methods'] if m!='legacy_matched'}
    write_json(r/'paired_ic_comparisons.json',comparisons)
    attribution=[]
    for method in c['methods']+['equal_weight']:
        policies=['weekly'] if method=='equal_weight' else ['weekly','buffer']
        for policy in policies:
            p=paths[(paths.method==method)&(paths.policy==policy)&(paths.cost_bps==20)].copy()
            gross=p.copy();gross['net_return']=gross.gross_return
            a=path_statistics(p);b=path_statistics(gross)
            attribution.append({'method':method,'policy':policy,'net':a,'same_holdings_zero_cost_attribution':b,
                'cumulative_cost_wedge':b['cumulative_return']-a['cumulative_return'],
                'note':'Identical fractional holdings, diagnostic only. Zero cost was not an eligible selection or admission setting.'})
    write_json(r/'cost_attribution.json',attribution)
    rows=[]
    for phase in ['development','holdout']:
        d=pd.read_csv(r/phase/'daily_ic.csv');d=d[d.group=='fusion']
        for (m,fold),x in d.groupby(['method','fold']):
            rows.append({'phase':phase,'method':m,'fold':fold,'mean_rank_ic':float(x.rank_ic.mean()),'valid_days':int(x.rank_ic.notna().sum())})
    pd.DataFrame(rows).to_csv(r/'fold_ic.csv',index=False,float_format='%.17g')
    contribution=[]
    for phase in ['development','holdout']:
        for p in sorted((r/phase/'evidence').glob('*/selection.json')):
            fold,method=p.parent.name.split('-',1);record=json.loads(p.read_text())
            for group,s in record['groups'].items():
                contribution.append({'phase':phase,'fold':fold,'method':method,'group':group,'nominal_weight':c['weights'][group],
                    'reliability':s['confidence'],'effective_signal_multiplier':c['weights'][group]*s['confidence'],
                    'validation_ic':s['selected']['validation']['mean'],'direction':s['selected']['direction'],'target':s['target']})
    pd.DataFrame(contribution).to_csv(r/'group_contributions.csv',index=False,float_format='%.17g')
    decisions=[]
    for m in c['methods']:
        p=paths[(paths.method==m)&(paths.policy=='weekly')&(paths.cost_bps==20)]
        dates=set(p.signal_date.iloc[::c['rebalance_sessions']]);z=ic[(ic.method==m)&ic.date.isin(dates)]
        decisions.append({'method':m,'decision_day_ic':float(z.rank_ic.mean()),'valid_decision_days':int(z.rank_ic.notna().sum()),'all_day_ic':test[m]['rank_ic']})
    write_json(r/'decision_day_ic.json',decisions)
    lines=['# QuantA V7：底层方法修正与Kaggle公开思路迁移','',
       '2026-11-16（北京时间）。2020–2021为已知开发资料；候选封存后才运行2022–2024后段。价格以前已见、当前财务和历史新闻版本未完整PIT认证，因此后段仍是回顾性检查。',
       '新增FinMultiTime来源仅取非空title与datetime，不用生成摘要；58股、后段43,231条去重标题，旧CMIN/报价/财务版本和58股交集不变。来源变化会带来分布漂移，所有后段方法用同一份数据。','',
       '## 实际改造','',
       '七个预先固定对照检验单窗口翻向、弱组等幅放大、训练历史过窄、财务目标期限以及市场上下文/排序目标。全部主要IC仍为H5，名义融合30/40/30；可靠性会降低有效贡献，其系数全部公开，不重分配。',
       'Kaggle作者代码启发了时序切分、市场截面聚合和多期限目标。小型LambdaRank是本项目独立排名改造，不声称复制比赛冠军。盘口、匿名特征、提前可得响应标签与排行榜选模没有移植。',
       '','## 开发与封存','',f'开发平均IC最高且覆盖合格的候选为 **{champion}**；其开发IC={dev[champion]["rank_ic"]:.4f}，四折通过0.02比例={dev[champion]["passing_fold_fraction"]:.1%}。封存文件和开发结果在GitHub推送后才运行后段，未看后段更换候选。',
       '开发指标不作为后段成功证明。旧式匹配对照在本轮也使用H20 purge，所以不可把V7旧式数值当V6原协议重跑。','',
       '## 2022–2024同协议结果','',
       '| 方法 | RankIC | 日ICIR | 半年折≥0.02 | 净夏普：周频 | 净夏普：缓冲 | 缓冲累计收益 |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for m in c['methods']:
        a=table[(table.method==m)&(table.policy=='weekly')&(table.cost_bps==20)].iloc[0];b=table[(table.method==m)&(table.policy=='buffer')&(table.cost_bps==20)].iloc[0];x=test[m]
        lines.append(f'| {m} | {x["rank_ic"]:.4f} | {x["ic_ir"]:.3f} | {x["passing_fold_fraction"]:.1%} | {a.net_sharpe:.3f} | {b.net_sharpe:.3f} | {b.cumulative_return:.2%} |')
    lines+=['','每边20bp为固定主费用，另有10/30bp。全部IC覆盖100%；其他费用/暴露/换手见case曲线。分数份额/供应商后复权收益筛查不是可执行股数现金账户，未认证盘口、税费和公司行动；不称实盘。','',
       '## 对目标的判定','',f'封存候选IC={test[champion]["rank_ic"]:.4f}，95%区块区间[{test[champion]["ci95"][0]:.4f}, {test[champion]["ci95"][1]:.4f}]，六折有五折达到0.02。**IC部分满足本轮预设门槛，净夏普未达到>1。** 财务门槛结论：{gate["passed"]}；原因：'+', '.join(gate['reasons'])+'。',
       '旧式匹配对照在后段也达到IC门槛，均值甚至略高于封存候选；不能把开发改善或时期变化说成已证明后段算法胜出。signed_stable等消融后段点估计较高，但它们未被预先选为候选，不能看完结果替换冠军。','',
       '| 相对旧式匹配对照 | 配对ΔIC | 95%区块区间 |','|---|---:|---|']
    for m,z in comparisons.items():lines.append(f'| {m} | {z["mean_delta_rank_ic"]:.4f} | [{z["block_ci95"][0]:.4f}, {z["block_ci95"][1]:.4f}] |')
    a=next(x for x in attribution if x['method']==champion and x['policy']=='buffer');net=a['net'];gross=a['same_holdings_zero_cost_attribution']
    eq=table[(table.method=='equal_weight')&(table.policy=='weekly')&(table.cost_bps==20)].iloc[0]
    lines+=['','## 为什么IC合格仍未盈利','',
       f'封存方案缓冲年换手{net["annual_turnover"]:.1f}；同持仓毛收益归因累计{gross["cumulative_return"]:.2%}，扣固定主费用后累计{net["cumulative_return"]:.2%}，净最大回撤{net["max_drawdown"]:.2%}。这个零费归因不用于选模/准入，也不是可用的成本设置。全池等权净累计{eq.cumulative_return:.2%}。',
       f'同持仓零费归因的夏普也只有{gross["net_sharpe"]:.3f}，所以问题不只是费用，毛信号与持仓规则本身也没有达到夏普1。降低假设费率不会被当成解决方案。',
       'IC衡量相对排序，并不保证全仓只多的Top20%具有足够绝对收益；市场方向、尾部收益、持有期和交易成本都影响利润。缓冲大幅减少换手，但仍未创造足以覆盖费用的优势。决策日IC另存，不能把全部日频IC当作每个交易日都实际变现。',
       '可靠性缩小的是组信号贡献；随后Top20%排序仍会充分投资，故较低分数幅度不自动降低总仓位。这是当前预测—组合接口的具体错配，需要未来把收益幅度、成本与现金决策连起来，而不是继续只优化名次。',
       'Kaggle启发的上下文/排名树也没有提供可靠胜出证据；竞赛相关性或NDCG不能替代成本后的组合目标。财务近似年化不是TTM，慢变量H20训练也未普遍改善后段；全部负结果保留。','',
       '## 下一步更值得精进的方向','',
       '1. 从“秩分数等幅混合”进一步转向过去验证期可校准的预期收益幅度，检验预期收益是否超过往返成本与不确定性，再决定是否交易。这个方向需要新封存试验，不在已用后段重调来报成功。',
       '2. 把选股、持有期限与成本目标统一；比较低换手持仓/交易缓冲时同时约束风险和暴露，不靠降低20bp费用或只展示牛市。',
       '3. 分离市场方向与相对选股的贡献，加入现金基线与等风险对照；当前充分投资的只多组合不是纯α。',
       '4. 补历史披露版本、行业/报表期口径、实体别名和更长新闻档案，先核验信息增量，再增加网络容量。',
       '以上为基于证据的待验证方向，未声称已经解决。**模拟盘未启动，订单0。** 即便部分IC门槛通过，数据/真实成交结构仍有阻断。','',
       '## 证据与复现','',
       '[Kaggle方法核查](../../docs/KAGGLE_METHOD_REVIEW.md) · [冻结协议](../../docs/V7_PROTOCOL.md) · [候选封存](champion_seal.json) · [复现](../../docs/V7_REPRODUCE.md)',
       '','全部七方法、两种持仓政策、三种费用、每折参数/方向/可靠性/模型指纹、逐股票预测和逐日路径保留。区块区间为条件诊断，不是覆盖整个研究历史的多重检验校正。']
    (r/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(r/'trial_ledger.json',{'methods':7,'development_folds':4,'later_folds':6,'ridge_candidates_per_group':3,
      'ridge_group_candidates_per_fold':60,'ranker_candidates_per_fold':1,'validation_candidates_total':610,
      'final_group_models':210,'selected_ridge_training_refits_for_calibration':200,'checkpoint_sets':70,
      'screen_curves':90,'seed':17,'primary_champion_presealed':champion,'primary_cost_bps':20,'primary_policy':'buffer',
      'after_later_results_retuning':False,'zero_fee_attribution_used_for_selection':False,'strict_pit':False,'paper_admitted':False})
    import os
    os.environ.setdefault("MPLCONFIGDIR","/tmp/quanta-matplotlib")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    for m in ['legacy_matched',champion,'signed_stable','kaggle_rank']:
        d=ic[ic.method==m].sort_values('date');axes[0].plot(d.date,d.rank_ic.rolling(120,min_periods=60).mean(),label=m)
    axes[0].axhline(.02,color='black',linestyle='--');axes[0].set_title('2022-2024 | 120-session mean fusion RankIC')
    for m in ['legacy_matched',champion,'equal_weight']:
        policy='weekly' if m=='equal_weight' else 'buffer';d=paths[(paths.method==m)&(paths.policy==policy)&(paths.cost_bps==20)]
        axes[1].plot(d.date,d.nav,label=m+' net')
        if m==champion:axes[1].plot(d.date,np.cumprod(1+d.gross_return),linestyle='--',label=m+' same holdings gross')
    axes[1].set_title('2022-2024 | Adjusted-return screen, 20 bp per side')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.savefig(r/'later_evaluation.png',dpi=140);fig.savefig(r/'later_evaluation.svg');plt.close(fig)
    with (r/'REPORT.md').open('a') as f:f.write('\n![后段IC与成本归因](later_evaluation.png)\n')
    print(json.dumps({'status':'passed','champion':champion,'gross_attribution':gross,'net':net},indent=2))
if __name__=='__main__':generate()
