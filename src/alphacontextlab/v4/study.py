"""Semiannual purged rolling study. Source/config/data frozen before all formal fits."""
import argparse,json,pickle,subprocess,platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from .data import FEATURES,load,fetch
from .metrics import daily_ic,ic_summary,signal_gate,path_statistics,paired_ic_increment,paper_gate
from .models import fit_fold
from .screen import screen_path


def folds(config):
    for year in config['test_years']:
        for month in [1,7]:
            begin=pd.Timestamp(year=year,month=month,day=1)
            if begin>pd.Timestamp(config['end']):continue
            val=begin-pd.DateOffset(months=config['validation_months'])
            train=val-pd.DateOffset(months=config['train_months'])
            end=min(begin+pd.DateOffset(months=config['test_months']),pd.Timestamp(config['end'])+pd.Timedelta(days=1))
            yield f'{year}H{1 if month==1 else 2}',train,val,begin,end


def split(panel,train,val,begin,end,config):
    base=panel.eligible & panel.label.notna() & panel.target.notna()
    count=panel[base].groupby('date').symbol.count();good_dates=count[count>=config['min_cross_section']].index
    base=base & panel.date.isin(good_dates)
    a=panel[base&(panel.date>=train)&(panel.date<val)&(panel.exit_date<val)].copy()
    b=panel[base&(panel.date>=val)&(panel.date<begin)&(panel.exit_date<begin)].copy()
    r=panel[base&(panel.date>=train)&(panel.date<begin)&(panel.exit_date<begin)].copy()
    q=panel[panel.eligible&(panel.date>=begin)&(panel.date<end)].copy()
    if min(len(a),len(b),len(q))<500:raise ValueError('Insufficient fold data')
    assert a.exit_date.max()<b.date.min() and r.exit_date.max()<q.date.min()
    return a,b,r,q


def run(config_path,root,out):
    config=json.loads(Path(config_path).read_text());out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory; no silent resume/overwrite')
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    if dirty:raise ValueError('Commit source/config before experiments')
    panel,raw,adjusted,calendar,manifest,quality=load(config,root)
    out.mkdir(parents=True,exist_ok=True)
    prov={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'config_sha256':sha256(config_path),'data_manifest_sha256':sha256(Path(root)/'manifest.json'),
          'packages':{p:version(p) for p in ['akshare','lightgbm','gplearn','scikit-learn','numpy','pandas','scipy']},
          'python':platform.python_version(),'seed':config['seed'],'source_dirty':False}
    write_json(out/'provenance.json',prov);write_json(out/'data_quality.json',quality);write_json(out/'data_manifest.json',manifest)
    pd.DataFrame(manifest['files']).to_csv(out/'data_versions.csv',index=False)
    all_predictions=[];ics=[];validation_ics=[];selection=[];factor_ics=[]
    # One predeclared candidate budget; failed folds fail the experiment, never disappear.
    for name,train,val,begin,end in folds(config):
        print('START',name,flush=True);a,b,r,q=split(panel,train,val,begin,end,config)
        pred,vpred,models,records=fit_fold(a,b,r,q,config)
        dst=out/'folds'/name;dst.mkdir(parents=True)
        split_record={k:str(v) for k,v in {'train_start':train,'validation_start':val,'test_start':begin,'test_end_exclusive':end,
                                         'train_max_label_exit':a.exit_date.max(),'refit_max_label_exit':r.exit_date.max()}.items()}
        split_record.update({'training_rows':len(a),'validation_rows':len(b),'refit_rows':len(r),'test_rows':len(q)})
        write_json(dst/'selection.json',{'split':split_record,'models':records,'provenance':prov})
        with (dst/'models.pkl').open('wb') as f:pickle.dump(models,f)
        write_json(dst/'checkpoint.json',{'sha256':sha256(dst/'models.pkl'),'provenance':prov})
        for arm in config['models']:
            z=q[['date','symbol','label','target']].copy();z['prediction']=pred[arm];z['model']=arm;z['fold']=name;all_predictions.append(z)
            d=daily_ic(q,pred[arm],config['min_cross_section']);d['model']=arm;d['fold']=name;ics.append(d)
            vd=daily_ic(b,vpred[arm],config['min_cross_section']);vd['model']=arm;vd['fold']=name;validation_ics.append(vd)
            s=ic_summary(vd,config);selection.append({'fold':name,'model':arm,'validation_mean_rank_ic':s['mean_rank_ic']})
            print(name,arm,'OOF RankIC',round(float(d.rank_ic.mean()),6),flush=True)
        for feature in FEATURES:
            d=daily_ic(q,q[feature].to_numpy(),config['min_cross_section']);d['factor']=feature;d['fold']=name;factor_ics.append(d)
    predictions=pd.concat(all_predictions,ignore_index=True);daily=pd.concat(ics,ignore_index=True)
    valdaily=pd.concat(validation_ics,ignore_index=True);factor_daily=pd.concat(factor_ics,ignore_index=True)
    predictions.to_csv(out/'predictions.csv.gz',index=False,float_format='%.17g')
    daily.to_csv(out/'daily_ic.csv',index=False,float_format='%.12g');valdaily.to_csv(out/'validation_ic.csv',index=False,float_format='%.12g')
    factor_daily.to_csv(out/'factor_ic.csv.gz',index=False,float_format='%.12g')
    pd.DataFrame(selection).to_csv(out/'validation_selection.csv',index=False)
    write_json(out/'trial_ledger.json',{'folds':len(list(folds(config))),'seed':[config['seed']],
       'linear_candidates_per_fold':6,'lightgbm_candidates_per_fold':2,'gp_ridge_alphas_per_fold':3,
       'gp_programs_per_fold':config['gp_population']*config['gp_generations'],'x_extra_search':0,
       'stage_policy':'Advanced arms are exploratory even if linear gate fails; no paper promotion by bypassing the gate'})
    results(config,out,predictions,daily,valdaily,adjusted,calendar,quality)
    print('COMPLETE',str(out),flush=True)


def results(config,out,predictions,daily,valdaily,adjusted,calendar,quality):
    signals={m:signal_gate(daily[daily.model==m],config) for m in config['models']}
    linear_pass=any(signals[m]['passed'] for m in ['ridge','elasticnet'])
    # This is an exploratory retrospective choice based on validation IC only.
    val_scores=valdaily.groupby('model').rank_ic.mean().sort_values(ascending=False,kind='stable')
    selected=str(val_scores.index[0]);paths=[];summary=[];yearly={}
    for arm in config['models']+['equal_weight']:
        p=predictions[predictions.model==('ridge' if arm=='equal_weight' else arm)]
        for cost in config['screen_oneway_cost_bps']:
            path=screen_path(p,adjusted,calendar,config['symbols'],config,cost,arm=='equal_weight')
            path['model']=arm;path['cost_bps']=cost;paths.append(path);stat=path_statistics(path)
            summary.append({'model':arm,'cost_bps':cost,**stat})
            if cost==config['screen_primary_cost_bps']:
                yearly[arm]={str(year):path_statistics(g) for year,g in path.groupby(pd.to_datetime(path.date).dt.year)}
    curves=pd.concat(paths,ignore_index=True);curves.to_csv(out/'screen_paths.csv.gz',index=False,float_format='%.12g')
    table=pd.DataFrame(summary);table.to_csv(out/'screen_summary.csv',index=False,float_format='%.12g')
    write_json(out/'yearly_screen.json',yearly);write_json(out/'signal_gates.json',signals)
    increments={m:paired_ic_increment(daily[daily.model=='ridge'],daily[daily.model==m],config) for m in config['models'] if m!='ridge'}
    write_json(out/'incremental_ic.json',increments)
    selected_performance=table[(table.model==selected)&(table.cost_bps==config['screen_primary_cost_bps'])].iloc[0].to_dict()
    gate=paper_gate(linear_pass,signals[selected],selected_performance,yearly[selected],quality,config)
    gate.update({'selected_model':selected,'selected_by':'mean_validation_rank_ic_only','linear_pass':linear_pass,
                 'backtest_type':'continuous_out_of_fold_fractional_adjusted_return_screen_not_executable_ledger',
                 'thresholds':config['paper_gate']})
    write_json(out/'paper_gate.json',gate)
    write_json(out/'frozen_candidate.json',{'model':selected,'validation_scores':{str(k):float(v) for k,v in val_scores.items()},
                                           'never_selected_on_oof_sharpe':True,'gate':gate['status']})


def report(config_path,out,reports):
    config=json.loads(Path(config_path).read_text());out=Path(out);dest=Path(reports)
    if dest.exists() and any(dest.iterdir()):raise ValueError('Do not overwrite published reports')
    dest.mkdir(parents=True)
    import shutil
    names=['predictions.csv.gz','data_manifest.json','provenance.json','data_quality.json','data_versions.csv','trial_ledger.json','daily_ic.csv','validation_ic.csv',
           'factor_ic.csv.gz','validation_selection.csv','screen_paths.csv.gz','screen_summary.csv','yearly_screen.json',
           'signal_gates.json','incremental_ic.json','paper_gate.json','frozen_candidate.json']
    for n in names:shutil.copy2(out/n,dest/n)
    folds_dir=dest/'evidence';folds_dir.mkdir()
    for p in sorted((out/'folds').glob('*/selection.json')):
        shutil.copy2(p,folds_dir/(p.parent.name+'-selection.json'))
        shutil.copy2(p.parent/'checkpoint.json',folds_dir/(p.parent.name+'-checkpoint.json'))
    gate=json.loads((dest/'paper_gate.json').read_text());signals=json.loads((dest/'signal_gates.json').read_text())
    table=pd.read_csv(dest/'screen_summary.csv');inc=json.loads((dest/'incremental_ic.json').read_text())
    lines=['# QuantA V4：AKShare 价量信号与滚动研究','',f"正式拟合源提交：`{json.loads((dest/'provenance.json').read_text())['source_commit']}`。",'',
           '固定60只长期上市存续A股；2014–2026原始及后复权快照。2021–2026均为回顾性开发，不是新盲测。',
           '每半年：30个月训练、6个月验证，H=5，标签为下个开盘到第6个开盘的供应商后复权收益代理。退出日严格早于切分边界。',
           '27个基础价量因子采用同日截面秩；没有用全样本标准化。缺失日保留，模型行缺因子则排除并计数。',
           '方向与参数只看验证集；线性信号门槛为有向OOF RankIC≥0.02、至少75%折≥0.02、20日区块95%均值区间下界>0。',
           'ICIR=日RankIC均值/样本标准差，不把重叠5日标签按sqrt252放大。', '',
           '## 信号与增量比较','', '| 方法 | 平均RankIC | 日ICIR | 达标折比例 | 95%区块区间 | 稳定性门槛 |', '|---|---:|---:|---:|---|---|']
    for m,s in signals.items():
        lo,hi=s['mean_rank_ic_ci95'];lines.append(f"| {m} | {s['mean_rank_ic']:.4f} | {s['daily_ic_ir']:.3f} | {s['fold_pass_fraction']:.1%} | [{lo:.4f}, {hi:.4f}] | {'通过' if s['passed'] else '未通过'} |")
    lines+=['','所有方向均在每折验证确定；不对样本外IC取绝对值，不因某年差而删年。高级方法即使在线性门槛失败时也只作为探索诊断。', '',
            '| 相对Ridge | 配对ΔRankIC | 95%区块区间 |','|---|---:|---|']
    for m,s in inc.items():
        lo,hi=s['block_ci95'];lines.append(f"| {m} | {s['mean_delta_rank_ic']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines+=['','## 连续样本外收益筛查','',
            '全部OOF信号日连续拼接持仓，跨年不重置资金，5日调仓，前20%等权，只做多；末尾平仓计费。',
            '主要情景为每边20bp的假设总费用，另列10/30bp压力；不是准确历史税费、整手/最低佣金/竞价成交现金账本。',
            '开盘目标权重是分数份额因子组合约定；停牌缺报价以旧价估值，不能把它称为可执行成交。后复权收益不是已认证分红现金总回报。',
            '全样本指全部滚动样本外日期，不是训练集拟合曲线。原始AKShare报价/模型留在本地快照，公开其版本哈希。','',
            '| 方法 | 每边成本bp | 净夏普 | 累计收益 | 年化收益 | 最大回撤 | 年换手 |','|---|---:|---:|---:|---:|---:|---:|']
    for _,r in table.iterrows():lines.append(f"| {r.model} | {int(r.cost_bps)} | {r.net_sharpe:.3f} | {r.cumulative_return:.2%} | {r.annualized_return:.2%} | {r.max_drawdown:.2%} | {r.annual_turnover:.1f} |")
    lines+=['','## 模拟盘状态','',f"验证IC选定候选：**{gate['selected_model']}**；状态：**{gate['status']}**。",'',
            '阻断原因：'+', '.join(gate['reasons'])+'。','',
            '没有发送模拟或真实订单，没有启动常驻交易任务。门槛失败时不得强行标记为进入模拟盘；local paper命令保留可审查的拦截记录。','',
            '## X机制与舆情边界','',
            'X启发的辅助臂用同一LightGBM分数，增加固定波动/流动性秩惩罚，复用候选→排序→过滤的思路。是独立金融适配，不是移植X原始权重，也没有其用户互动数据。',
            'AKShare新闻只做前向采集，published_at和first_seen_at都必须早于决策时刻。词典情绪只是可解释代理；本轮没有可信历史首次可见档案，未加入历史训练，不伪造历史舆情增益。','',
            '## 已知问题与下一步','',
            '固定存续股票池有幸存者/人工选择偏差；当期后复权历史会修订；样本内选择和多重试验使漂亮OOF点估计仍有选择偏差。',
            '置信区间是各臂的描述性诊断，不是覆盖整个研究历史的多重检验校正；不能把每半年重新定向当作固定信号方向稳健性。',
            '进入模拟盘前需要历史可得股票池、公司行动及真实可执行账本；先通过信号稳定性与净夏普门槛，再封存未来纸面评估。',
            '完整折参数、早停轮数、GP公式、数据文件SHA256、随机种子和源提交见同目录证据。旧V1/V2/V3报告保留。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fetch','run','report']);p.add_argument('--config',default='configs/v4.json')
    p.add_argument('--data',default='data/v4_akshare');p.add_argument('--outputs',default='outputs/v4');p.add_argument('--reports',default='reports/v4')
    a=p.parse_args()
    if a.action=='fetch':fetch(json.loads(Path(a.config).read_text()),a.data)
    elif a.action=='run':run(a.config,a.data,a.outputs)
    else:report(a.config,a.outputs,a.reports)
if __name__=='__main__':main()
