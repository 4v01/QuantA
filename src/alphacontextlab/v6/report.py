"""Post-fit synthesis and paired comparisons, never used to select follow-up variants."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from ..study import write_json
from ..v4.metrics import paired_ic_increment


def generate(root='reports/v6'):
    r=Path(root);config=json.loads(Path('configs/v6.json').read_text());variants=['baseline','entity_filter','entity_tfidf']
    summaries={v:json.loads((r/('followup_'+v)/'ic_summary.json').read_text()) for v in variants}
    frames={v:pd.read_csv(r/('followup_'+v)/'daily_ic.csv',parse_dates=['date']) for v in variants}
    increments={}
    for v in variants[1:]:
        increments[v]={g:paired_ic_increment(frames['baseline'][frames['baseline'].model==g],frames[v][frames[v].model==g],
              {'seed':config['seed'],'signal_gate':{'bootstrap_repetitions':config['bootstrap_repetitions']}}) for g in ['alternative','fusion']}
    write_json(r/'followup_paired_comparisons.json',increments)
    rows=[];direction=[]
    for v in ['pilot']+['followup_'+x for x in variants]:
        d=pd.read_csv(r/v/'daily_ic.csv');t=pd.read_csv(r/v/'screen_summary.csv');s=json.loads((r/v/'ic_summary.json').read_text())
        for group,m in s.items():
            stat=t[(t.model==group)&(t.cost_bps==20)].iloc[0]
            rows.append({'case':v,'model':group,**m,'net_sharpe_proxy_20bp':stat.net_sharpe,'cumulative_return_proxy':stat.cumulative_return})
        for p in sorted((r/v/'evidence').glob('*-selection.json')):
            e=json.loads(p.read_text());fold=p.name.replace('-selection.json','')
            for group,selection in e['selection']['groups'].items():
                direction.append({'case':v,'fold':fold,'group':group,**selection['selected'],
                                  'oof_rank_ic':float(d[(d.fold==fold)&(d.model==group)].rank_ic.mean())})
    # JSON preserves the interval lists without misleading CSV string parsing.
    write_json(r/'comparison.json',rows);pd.DataFrame(direction).to_csv(r/'direction_diagnostics.csv',index=False,float_format='%.17g')
    pilot=json.loads((r/'pilot/ic_summary.json').read_text());out=['# V6：GitHub历史来源、首次融合IC与自主优化复核','',
       '本轮实际采用CMIN-CN原始历史新闻：58只共同股票、433,482条去重标题，固定作者仓库提交。检索核查8个开源项目，区分数据集与仅有采集器的工具。',
       '来源、配置和旧V5严格框架均保留。V6是按新闻日期与财报公告/更新日重建的**回顾性研究，不是严格PIT或前瞻盲测**；另类30%明确改为历史中国公司新闻，基本面30%/价量40%不变。','',
       '## 首次融合：2020年','',
       '| 组 | RankIC | 日ICIR |','|---|---:|---:|']
    for g,m in pilot.items():out.append(f"| {g} | {m['rank_ic']:.4f} | {m['ic_ir']:.3f} |")
    out+=['','各均值区间跨零；融合IC0.0037未超过价量0.0068。基本面也跨期反向，不能将全部弱表现归因于新闻。',
          '','## 根据首轮IC选定的方向','',
          '新闻组IC为负、与价量预测相关近零，原始公司新闻文件又含其它公司主题条目。因此封存一个方向：实体关联过滤与字符文本表示，并以仅过滤作为消融。',
          '2020原结果和改进计划在提交`8c50a43f2e3b97bda49f7305cf50385900427c55`公开后，才运行2021三臂；价格与基本面预测在三臂逐行一致，费用、样本和权重相同。',
          '2021行情已见于V4，因此本轮是后续回顾性复核，不把它称为绝对未见的盲测。','',
          '## 同一2021样本下的结果','',
          '| 新闻编码 | 新闻组IC | 融合IC | 融合日ICIR | 融合净夏普* | 累计收益* |','|---|---:|---:|---:|---:|---:|']
    labels={'baseline':'原方案','entity_filter':'仅实体过滤','entity_tfidf':'实体过滤+TF-IDF'}
    for v in variants:
        m=summaries[v];t=pd.read_csv(r/('followup_'+v)/'screen_summary.csv');z=t[(t.model=='fusion')&(t.cost_bps==20)].iloc[0]
        out.append(f"| {labels[v]} | {m['alternative']['rank_ic']:.4f} | {m['fusion']['rank_ic']:.4f} | {m['fusion']['ic_ir']:.3f} | {z.net_sharpe:.3f} | {z.cumulative_return:.2%} |")
    out+=['','*每边20bp分数份额/供应商后复权收益筛查，非可执行现金账本。每个阶段独立初始资金，阶段内半年间不重置，不把2020和2021拼成实盘账户。10/30bp及全部组、等权结果见各case目录。',
          '','| 相对2021原融合 | 配对ΔIC | 95%区块区间 |','|---|---:|---|']
    for v,d in increments.items():
        z=d['fusion'];lo,hi=z['block_ci95'];out.append(f"| {labels[v]} | {z['mean_delta_rank_ic']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    out+=['','**改进未获支持。** 过滤使新闻组点估计稍升，但融合未升；字符TF-IDF的两项IC和收益筛查均退化。区间仍宽，不将最高点估计挑作确认成功。',
          '','## 为什么仍低，以及下一步方向','',
          '- 组内预测有效性弱、符号随验证窗口变化；同样方差的标准化分数不等于同样可靠。固定30/40/30会让弱组也提供足量噪声。后续优先研究基于多个过去验证窗口的可靠性校准，并明确记录其对有效贡献的影响，不偷偷改权重。',
          '- 基本面是慢变量，却预测H5收益，且当前表的部分更新晚于原公告；六字段含YTD、行业差异。后续先比较披露时点、报告期对齐、行业处理与不同预测期限，再给复杂模型更多容量。',
          '- 完整公司名过滤会漏掉简称，词典难表达语义，字符文本可能学到公司/风格噪声。下一步需要可审查的历史别名与事件类型；不能在看到2021收益后手挑别名把本轮结果改好。',
          '- 新闻档案只到2021，股票池为存续交集；新数据源解决了获取问题，没有消除样本选择、修订链和历史时刻的不确定性。更长数据必须另立版本，不能把缺档日填成中性消息。',
          '','这些是后续待验证方向，**本轮没有声称实现稳定IC≥0.02或净夏普>1**。没有启动模拟盘、没有订单。',
          '','## 复核材料','',
          '[来源benchmark](../../docs/OPEN_SOURCE_DATA_BENCHMARK.md) · [冻结协议](../../docs/V6_PROTOCOL.md) · [首轮IC](pilot/REPORT.md) · [改进封存](../../configs/v6_improvement.json) · [复现说明](../../docs/V6_REPRODUCE.md)',
          '','每个case公开逐股票预测、逐日IC、参数/符号选择、校准、数据指纹、源提交与检查点哈希。原始新闻、财务/报价与自有模型在本地快照。旧V1–V5和首轮报告不覆盖。']
    (r/'REPORT.md').write_text('\n'.join(out)+'\n')
    write_json(r/'trial_ledger.json',{'cases':['pilot']+['followup_'+x for x in variants],
      'folds':8,'groups_per_fold':3,'alpha_candidates_per_group':3,'validation_candidates':72,
      'final_group_refits':24,'selected_training_refits_for_calibration':24,'seed':17,
      'improvement_directions':1,'followup_variants_including_component_ablation':3,
      'no_2021_based_retuning_or_champion_replacement':True,'strict_pit':False,'paper_admitted':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for v in variants:
        d=frames[v];z=d[d.model=='fusion'].sort_values('date');axes[0].plot(z.date,z.rank_ic.rolling(40,min_periods=20).mean(),label=v)
        t=pd.read_csv(r/('followup_'+v)/'screen_paths.csv.gz',parse_dates=['date']);t=t[(t.model=='fusion')&(t.cost_bps==20)];axes[1].plot(t.date,t.nav,label=v)
    axes[0].axhline(.02,color='black',linestyle='--');axes[0].set_title('2021 | 40-session mean fusion RankIC');axes[1].set_title('2021 | Adjusted-return screen, 20 bp/side')
    for ax in axes:
        ax.legend(fontsize=8);ax.grid(alpha=.2);ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.tick_params(axis='x',labelrotation=20)
    fig.savefig(r/'followup.png',dpi=140);fig.savefig(r/'followup.svg');plt.close(fig)
    with (r/'REPORT.md').open('a') as f:f.write('\n![2021同协议对照](followup.png)\n')
    print(json.dumps({'status':'passed','cases':4,'comparison_rows':len(rows)},indent=2))
if __name__=='__main__':generate()
