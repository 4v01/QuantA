"""Post-fit plots and score/coverage diagnostics; never used by selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from ..study import write_json
from ..data import sha256


def generate(reports='reports/v4',news_root='data/v4_news',outputs='outputs/v4'):
    r=Path(reports);config=json.loads(Path('configs/v4.json').read_text());p=pd.read_csv(r/'predictions.csv.gz',dtype={'symbol':str})
    metrics={}
    for model,g in p.groupby('model'):
        z=g.dropna(subset=['target','prediction']);metrics[model]={'rows':len(z),
            'rank_target_rmse':float(np.sqrt(mean_squared_error(z.target,z.prediction))),
            'rank_target_r2':float(r2_score(z.target,z.prediction)),
            'target_definition':'cross-sectional future hfq proxy return percentile minus 0.5, not raw return R2'}
    write_json(r/'score_metrics.json',metrics)
    factor=pd.read_csv(r/'factor_ic.csv.gz');summary=factor.groupby('factor').agg(mean_rank_ic=('rank_ic','mean'),std_rank_ic=('rank_ic','std'),days=('rank_ic','count'))
    summary['daily_ic_ir']=summary.mean_rank_ic/summary.std_rank_ic
    summary.sort_values('mean_rank_ic',ascending=False).to_csv(r/'factor_summary.csv',float_format='%.12g')
    write_json(r/'coverage.json',{'eligible_oof_stock_dates':len(p[p.model=='ridge']),
        'oof_dates':int(p.date.nunique()),'cross_section_min':int(p[p.model=='ridge'].groupby('date').size().min()),
        'cross_section_max':int(p[p.model=='ridge'].groupby('date').size().max()),
        'unlabeled_oof_rows_per_model':int(p[p.model=='ridge'].label.isna().sum()),
        'failure_attempts':[{'source':'f0e9bb3732e5fcebc2b36072e9fcc1dd96e00c00','stage':'first-fold validation, before OOF prediction',
             'reason':'Constant ElasticNet candidate made correlation undefined; fixed invalid-candidate handling without changing config'}]})
    sources=[]
    for f in sorted(Path(news_root).glob('*.json')):
        d=json.loads(f.read_text());sources.append({'file':f.name,'sha256':sha256(f),'rows':len(d['records']),
            'fetched_at':d['fetched_at'],'earliest_published':min(x['published_at'] for x in d['records']),
            'latest_published':max(x['published_at'] for x in d['records'])})
    write_json(r/'news_acquisition.json',{'sources':sources,'used_in_historical_models':False,
       'availability':'max(published_at, first_seen_at); all collected after evaluation cutoff',
       'note':'Current small public snapshot, not a complete archive. No sentiment performance claimed.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=pd.read_csv(r/'screen_paths.csv.gz',parse_dates=['date']);paths=paths[paths.cost_bps==config['screen_primary_cost_bps']]
    ic=pd.read_csv(r/'daily_ic.csv',parse_dates=['date']);fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    for m in config['models']+['equal_weight']:
        g=paths[paths.model==m];axes[0].plot(g.date,g.nav,label=m)
    axes[0].set_title('V4 | Continuous OOF adjusted-return screen | 20 bp per side');axes[0].set_ylabel('Proxy NAV');axes[0].legend(ncol=3,fontsize=8);axes[0].grid(alpha=.2)
    for m in config['models']:
        g=ic[ic.model==m].sort_values('date');axes[1].plot(g.date,g.rank_ic.rolling(120,min_periods=60).mean(),label=m)
    axes[1].axhline(.02,color='black',linestyle='--',label='0.02 threshold');axes[1].set_title('120-session rolling mean OOF RankIC | validation-only direction');axes[1].set_ylabel('Mean RankIC');axes[1].legend(ncol=3,fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(r/'oof_screen.png',dpi=140);fig.savefig(r/'oof_screen.svg');plt.close(fig)
    with (r/'REPORT.md').open('a') as f:
        f.write('\n## 附加诊断\n\n![OOF收益与IC](oof_screen.png)\n\nscore_metrics.json保留截面秩目标RMSE/R²，不能与旧收益率标签的R²直接比较；factor_summary.csv为基础因子的原始有向OOF IC，无事后翻转，不用于重挑本轮模型。\n')
    print(json.dumps({'status':'passed','score_models':len(metrics),'factor_count':len(summary),'news_files':len(sources)},indent=2))
