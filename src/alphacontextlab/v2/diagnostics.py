"""Post-evaluation visualizations and decomposition, without further selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def diagnose(root='reports/v2'):
    root=Path(root);table=pd.read_csv(root/'results.csv');seal=json.loads((root/'champion.json').read_text())
    chosen=table[(table.arm==seal['arm'])&(table.cost_bps==5)&(((table['mode']=='development')&(table.period=='pooled'))|((table['mode']=='confirmation')&(table.period=='2025')))]
    aggregate=chosen.groupby(['mode','period','profile'],sort=False)[['sharpe_rf_zero','annualized_return','max_drawdown','annual_turnover']].mean().reset_index()
    aggregate.to_csv(root/'champion_decision_ablations.csv',index=False)
    daily=pd.read_csv(root/'daily_paths.csv.gz');concentration=[]
    focus=daily[(daily.profile=='weekly_risk_cost')&(daily.cost_bps==5)&daily.arm.isin(['champion_ensemble','csi300','inverse_volatility'])]
    for (mode,arm),g in focus.groupby(['mode','arm'],sort=False):
        log=np.log1p(g.net_return.to_numpy());total=float(log.sum());top=float(np.sort(log)[-5:].sum())
        concentration.append({'mode':mode,'arm':arm,'days':len(g),'max_daily_net_gain':float(g.net_return.max()),
            'max_daily_net_loss':float(g.net_return.min()),'top5_positive_log_growth':top,'total_log_growth':total,
            'top5_share_of_log_growth':top/total if total>0 else None})
    (root/'return_concentration.json').write_text(json.dumps({'scope':'Post-hoc concentration diagnostic only; no days deleted or metrics reselected.',
                                                             'records':concentration},indent=2)+'\n')
    lines=['# V2 诊断与后续优先级','',
           '以下为全部预设评估结束后的描述性分析，不用于本版再次选择模型或交易规则。','',
           '## 同一预测下的交易决策消融','',
           f"固定封存类别：{seal['arm']}。每个阶段复用完全相同预测，只改变预声明的调仓与价值函数。",'',
           '| 阶段 | 决策 | 净夏普 | 年化收益 | 最大回撤 | 年换手 |','|---|---|---:|---:|---:|---:|']
    for r in aggregate.to_dict('records'):
        sh='未定义' if pd.isna(r['sharpe_rf_zero']) else f"{r['sharpe_rf_zero']:.3f}"
        lines.append(f"| {r['mode']} | {r['profile']} | {sh} | {r['annualized_return']:.2%} | {r['max_drawdown']:.2%} | {r['annual_turnover']:.1f} |")
    lines+=['','主协议仍为weekly_risk_cost，不能把测试表中最好的另一行改成预封存冠军。',
            '超参数统一按主协议验证效用选择后，才固定预测做这组消融；不是为每一种决策分别调优后的最佳策略竞赛。','',
            '## 自优化究竟验证了什么','',
            '- 候选按可解释机制分组，因子变异、交叉与拒绝均有日志；选择只看对应训练分区。它证明研究环节可审计，不等于证明演化模块带来超额收益。',
            '- 冠军是基础Ridge，说明本轮验证证据没有支持用更复杂模型替代它；不能仅挑某一年或某个种子反向包装结论。',
            '- 静态/演化因子臂改变了输入数量与组合，并非严格参数量匹配的因果消融；部分初始因子与基础特征重叠，Ridge等效正则会随重复特征变化。',
            '- 组合风险是单资产预测下行风险加历史相关矩阵的代理，不是已校准的联合尾部概率；四只股票ETF高度相关，多样性收益受到限制。',
            '- 路径集中度见return_concentration.json：不删除任何交易日，只列前5个正向日对净对数增长的贡献；开发期净增长很小，不能因2025较好就忽略早期大回撤。','',
            '## 先完成的下一步','',
            '1. 获取可审计逐时点复权/分红折算和完整停复牌资料，补整手、资金规模、最低佣金与真实开盘撮合。当前数据/执行近似不能被正回测替代。',
            '2. 在新冻结实验中，把风险和成本单独对照，检验风险目标、滞回和较低调仓频率是否跨时期稳定；2025已经用过，不能继续称它为未见确认集。',
            '3. 因子库先对基础特征做去重，再增加同数量随机选择/主题选择对照、跨时期稳定性和相关约束，区分容量、正则变化与真正的因子信息增益。',
            '4. 只有获得Level-2/逐笔数据后再做MicroTSM式微观结构实验；不为追求外部论文指标虚构特征。',
            '5. 新研究必须预先封存，并在尚未分析的新时期/资产域确认；不在这次测试结果上循环改到满意。','']
    (root/'DIAGNOSTICS.md').write_text('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    data=daily[(daily.profile=='weekly_risk_cost')&(daily.cost_bps==5)]
    fig,axes=plt.subplots(2,2,figsize=(12,7),gridspec_kw={'height_ratios':[2,1]})
    for column,mode in enumerate(['development','confirmation']):
        for arm in ['champion_ensemble','csi300','inverse_volatility','momentum60']:
            group=data[(data['mode']==mode)&(data.arm==arm)]
            nav=np.r_[1.,np.cumprod(1+group.net_return)]
            dates=pd.to_datetime([group.entry_date.iloc[0]]+group.exit_date.tolist())
            label='sealed champion: '+seal['arm'] if arm=='champion_ensemble' else arm
            line=axes[0,column].plot(dates,nav,label=label,lw=1.3)[0]
            axes[1,column].plot(dates,nav/np.maximum.accumulate(nav)-1,color=line.get_color(),lw=1.1)
        axes[0,column].set_title('Development 2021-2024' if mode=='development' else 'Uninspected-2025 confirmation')
        axes[0,column].set_ylabel('Net NAV');axes[1,column].set_ylabel('Drawdown');axes[0,column].legend(fontsize=8)
        for ax in axes[:,column]:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
    fig.suptitle('QuantA V2 | 5bp one-way cost | annual folds reset, no real orders')
    fig.tight_layout();fig.savefig(root/'equity_and_drawdown.png',dpi=160);fig.savefig(root/'equity_and_drawdown.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports/v2');a=p.parse_args();diagnose(a.reports)
