"""Public-data-only post hoc decomposition, not additional strategy selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def diagnose(root='reports/v3'):
    root=Path(root);table=pd.read_csv(root/'results.csv');seal=json.loads((root/'champion.json').read_text());family=seal['family']
    base=table[(table.period=='2026')&(table.family==family)&(table.status=='success')]
    base.to_csv(root/'champion_sensitivity.csv',index=False)
    primary=table[(table.scenario=='primary')&(table.status=='success')]
    lines=['# V3：哪些改进有效，哪些没有证据','',
           '本分析在全部预声明实验结束后生成；不更换冠军，不新增参数搜索，不删除极端交易日。','',
           '## 有效提升主要在风险与交易强度','',
           f'固定冠军：{family}。所有方法使用相同预测、原始价格、分红权益和现金股数账本。',
           '各家族的gamma按相同验证网格分别选择，所以这是完整流程比较，不是严格固定gamma的单一组件因果消融。','',
           '| 阶段 | 方法 | 年化波动 | 平均股票暴露 | 净夏普 | 最大回撤 | 年换手 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for mode,period in [('development','pooled'),('confirmation','2026')]:
        for name in [family,'legacy_discrete','sample_mv','lw_mv','inverse_volatility']:
            selected=primary[(primary['mode']==mode)&(primary.period==period)&(primary.family==name)]
            if len(selected)==0:continue
            row=selected.iloc[0]
            lines.append(f"| {mode} | {name} | {row.annualized_volatility:.2%} | {row.mean_exposure:.2%} | {row.sharpe_rf_zero:.3f} | {row.max_drawdown:.2%} | {row.annual_turnover:.1f} |")
    lines+=['','较低股票暴露本身能减少回撤；不能仅凭回撤较低就称为同风险收益更优。确认检验区间是主要结论，漂亮点估计不能替代它。','',
            '## 成本与账户规模','',
            '| 场景 | 2026累计收益 | 净夏普 | 最大回撤 | 佣金（元） | 滑点（元） |',
            '|---|---:|---:|---:|---:|---:|']
    for r in base.to_dict('records'):
        lines.append(f"| {r['scenario']} | {r['cumulative_return']:.2%} | {r['sharpe_rf_zero']:.3f} | {r['max_drawdown']:.2%} | {r['commission_cny']:.2f} | {r['slippage_cny']:.2f} |")
    lines+=['','不同资金规模下费用金额不能直接相减比较；参数没有按这些场景重新选择。零费情景仍保留整手、收盘定量订单和公司行动。','',
            '## 文献的本项目证据','',
            '- 凸优化使连续权重、历史风险和成本在同一问题中表达，数值求解和现金/股数可检验；它没有自动创造新的预测信息。',
            '- Ledoit-Wolf单独加入未显示普遍优势；本例资产数仅4，不能用高维估计论文结论替代本轮对照。',
            '- 部分调仓与成本惩罚减少换手时也可能错失上涨；方差管理的风险/收益权衡必须同时看。Cederburg等的反面证据因此被保留。',
            '- 2026预测R²仍为负，说明仅凭当前8项日线输入的10日收益点预测没有解释力优势。没有发现可靠新α。',
            '- 57候选验证诊断的p值仅针对当前有限候选及验证资料，不是对研究历史的全局校正；不计算依赖未知有效试验数的精确DSR。','',
            '## 下一轮更有效的投入','',
            '1. 优先获取可认证的公告可得时刻/报价版本和更真实成交资料，验证当前固定滑点与ADV代理，而不是把账本仿真称为真实撮合。',
            '2. 新增经过可得性审计的信息或更广资产域，检验是否带来预测增益；4只高度相关ETF的日线信号空间有限。',
            '3. 用新冻结的等风险、固定gamma和更长时段对照区分收益信号、现金比例及风险管理贡献；不能回头重调已用2026以制造确认。',
            '4. 若进行新前瞻检查，应在真正未来区间到来前封存；当前2026检查为回顾性，不能升级宣传为盲测。','']
    (root/'DIAGNOSTICS.md').write_text('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    daily=pd.read_csv(root/'daily_paths.csv.gz');data=daily[daily.scenario=='primary']
    fig,axes=plt.subplots(2,2,figsize=(12,7),gridspec_kw={'height_ratios':[2,1]})
    for column,mode in enumerate(['development','confirmation']):
        for name in [family,'legacy_discrete','csi300','equal_weekly']:
            group=data[(data['mode']==mode)&(data.family==name)];nav=np.r_[1.,np.cumprod(1+group.net_return)]
            dates=pd.to_datetime([group.entry_date.iloc[0]]+group.exit_date.tolist())
            line=axes[0,column].plot(dates,nav,label=name,lw=1.2)[0]
            axes[1,column].plot(dates,nav/np.maximum.accumulate(nav)-1,color=line.get_color(),lw=1.1)
        axes[0,column].set_title('Development 2021-2025' if mode=='development' else '2026 Jan-Oct retrospective check')
        axes[0,column].legend(fontsize=8);axes[0,column].set_ylabel('Net NAV');axes[1,column].set_ylabel('Drawdown')
        for ax in axes[:,column]:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
    fig.suptitle('QuantA V3 | integer shares, dividends, minimum commissions | fictional accounts')
    fig.tight_layout();fig.savefig(root/'equity_and_drawdown.png',dpi=160);fig.savefig(root/'equity_and_drawdown.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports/v3');a=p.parse_args();diagnose(a.reports)
