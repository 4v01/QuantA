"""Post-hoc descriptive diagnostics; never changes fitted models or allocation rules."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import forecast_metrics
from .study import write_json


def diagnose(config_path='configs/study.json',outputs='outputs',reports='reports/v1'):
    config=json.loads(Path(config_path).read_text());outputs=Path(outputs);reports=Path(reports)
    results=json.loads((reports/'forecast_results.json').read_text());rows=[]
    for r in results:
        selection=json.loads((outputs/f"{r['test_year']}-{r['arm']}-{r['seed']}"/'selection.json').read_text())
        winner=selection['selected']
        rows.append({'method':r['arm'],'seed':r['seed'],'year':r['test_year'],'parameters':r['parameters'],
                     'selected_epochs':None if r['arm']=='ridge' else winner['best']['epoch'],
                     'validation_rmse':winner['validation_rmse'],**r['forecast_metrics'],
                     'excluded_nontradable_intervals':r['forecast_excluded_nontradable_intervals']})
    zero=[]
    for year in config['test_years']:
        meta=pd.read_csv(outputs/f'{year}-metadata.csv',dtype={'symbol':str});target=meta[meta.forecast_valid].target.to_numpy()
        zero.append({'year':year,'observations':len(target),'zero_return_forecast':forecast_metrics(target,np.zeros_like(target))})
    write_json(reports/'zero_forecast_reference.json',{'scope':'Post-hoc no-fit arithmetic reference, never used to select models or rules.','years':zero})
    pd.DataFrame(rows).to_csv(reports/'forecast_diagnostics.csv',index=False)
    table=pd.read_csv(reports/'backtest_results.csv');primary=table[(table.period=='pooled')&(table.cost_bps==config['primary_cost_bps'])]
    daily=pd.read_csv(reports/'daily_net_returns.csv')
    paired=[];rng=np.random.default_rng(2026)
    def sharpe(r):
        sd=np.std(r,ddof=1)
        return float(np.mean(r)/sd*np.sqrt(252)) if sd>0 else np.nan
    for seed in config['seeds']:
        a=daily[(daily.method=='lstm_memory')&(daily.seed==seed)].net_return.to_numpy()
        for control in ['lstm','lstm_capacity']:
            b=daily[(daily.method==control)&(daily.seed==seed)].net_return.to_numpy();deltas=[]
            for _ in range(config['bootstrap_repetitions']):
                start=rng.integers(0,len(a),size=int(np.ceil(len(a)/20)))
                ids=((start[:,None]+np.arange(20))%len(a)).reshape(-1)[:len(a)]
                delta=sharpe(a[ids])-sharpe(b[ids])
                if np.isfinite(delta):deltas.append(delta)
            paired.append({'seed':seed,'comparison':'lstm_memory - '+control,'delta_sharpe':sharpe(a)-sharpe(b),
                           'paired_block_bootstrap95':np.quantile(deltas,[.025,.975]).tolist()})
    write_json(reports/'memory_paired_diagnostics.json',{'scope':'Post-hoc conditional descriptive comparison; no multiple-testing correction or model reselection.',
                                                       'block_length':20,'pairs':paired})
    lines=['# 结果诊断与下一步','',
           '这是完整测试结果生成后的描述性诊断，不参与本版模型、种子、交易阈值或持仓选择。','',
           '## 预测与配置','',
           '| 方法 | 参数量 | 选中轮数范围 | 各年度/种子预测 R² 范围 |',
           '|---|---:|---:|---:|']
    df=pd.DataFrame(rows)
    for method,g in df.groupby('method',sort=False):
        epochs='不适用' if g.selected_epochs.isna().all() else f'{int(g.selected_epochs.min())}–{int(g.selected_epochs.max())}'
        lines.append(f'| {method} | {int(g.parameters.iloc[0])} | {epochs} | {g.r2.min():.4f} 至 {g.r2.max():.4f} |')
    lines+=['','R² 接近零或为负表示对单日收益的平方误差优势不足；不能把它与电影评分 R² 或策略收益率混为一谈。',
            '逐年度零收益预测的无需训练参照见 zero_forecast_reference.json。预测误差中排除不可交易开盘区间，组合收益保留这些日期；排除数已逐拟合披露。','',
            '## 记忆模块消融','',
            '| 种子 | 对照 | 记忆模块夏普差 | 配对区块95%区间 |','|---:|---|---:|---|']
    for p in paired:
        lo,hi=p['paired_block_bootstrap95']
        lines.append(f"| {p['seed']} | {p['comparison']} | {p['delta_sharpe']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines+=['','这些区间只反映固定策略在这段历史上的时间重采样，不包含搜索选择、数据修订或市场结构变化的不确定性；3个训练种子不是3份独立市场历史。','',
            '## 成本与稳定性','',
            '| 方法 | 零成本夏普均值 | 5bp夏普均值 | 20bp夏普均值 |','|---|---:|---:|---:|']
    for method in primary.method.unique():
        values=[]
        for cost in [0,5,20]:
            v=table[(table.method==method)&(table.period=='pooled')&(table.cost_bps==cost)].sharpe_rf_zero.mean()
            values.append('未定义' if pd.isna(v) else f'{v:.3f}')
        lines.append('| '+method+' | '+' | '.join(values)+' |')
    lines+=['','## 下一版按证据优先安排','',
            '1. **先补数据与执行。** 获取逐时点复权/分红折算、完整历史停复牌和交易日历，加入整手、最低佣金、资金规模及开盘成交限制。当前结果不能支持直接部署。',
            '2. **降低预测任务与交易成本的错配。** 另立冻结协议对照5日/20日收益或风险预测、固定较低换手再平衡与成本感知持仓；标签重叠时延长purge。只在训练/验证选择阈值，保留零收益/买入持有参照。',
            '3. **先验证便宜方法再扩网络。** 加入趋势与波动目标、收缩协方差/风险预算、树模型和强正则线性模型，在相同信息和预算下比较。',
            '4. **记忆机制继续作为待验证假设。** 分别检验离散化信息损失、桶冲突、门控、记忆容量与市场状态漂移；不能在同一测试时期反复调到最好后宣称独立发现。',
            '5. **预留确认。** 本版未获取2025年之后行情；未来升级先冻结方案，再使用此前未分析的时期确认，并控制多重尝试。当前复权数据仍有事后修订，确认版需改用合格的逐时点数据。','']
    (reports/'DIAGNOSTICS.md').write_text('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for method,seed in [('csi300_buy_hold',None),('equal_buy_hold',None),('momentum20',None),('ridge',17),('lstm',17),('lstm_capacity',17),('lstm_memory',17)]:
        subset=daily[daily.method==method]
        subset=subset[subset.seed.isna()] if seed is None else subset[subset.seed==seed]
        equity=np.r_[1.,np.cumprod(1+subset.net_return.to_numpy())]
        dates=pd.to_datetime([subset.entry_date.iloc[0]]+subset.exit_date.tolist())
        label=method if seed is None else f'{method} (seed {seed})'
        line=axes[0].plot(dates,equity,label=label,lw=1.2)[0]
        axes[1].plot(dates,equity/np.maximum.accumulate(equity)-1,color=line.get_color(),lw=1)
    axes[0].set_title('A-share ETF research | 5bp one-way cost | fixed seed 17, not best-seed selection')
    axes[0].set_ylabel('Net NAV');axes[1].set_ylabel('Drawdown');axes[0].legend(ncol=2,fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(reports/'equity_and_drawdown.png',dpi=160);fig.savefig(reports/'equity_and_drawdown.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/study.json');p.add_argument('--outputs',default='outputs');p.add_argument('--reports',default='reports/v1');a=p.parse_args()
    diagnose(a.config,a.outputs,a.reports)
