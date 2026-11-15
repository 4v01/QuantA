"""Post-evaluation attribution, explicitly excluded from candidate selection."""
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import sha256
from ..study import write_json
from ..v4.metrics import daily_ic,block_mean_interval
from ..v7.study import split,ic_summary
from .data import load
from .models import predict


def risk_structure():
    tape=pd.read_csv('reports/v9/evaluation/market_tape.csv.gz',dtype={'symbol':str},parse_dates=['date'])
    x=tape[tape.date>='2022-01-01'].pivot(index='date',columns='symbol',values='close_return').dropna()
    correlation=x.corr().to_numpy();eigen=np.linalg.eigvalsh(correlation).clip(min=0)
    write_json('reports/v9/risk_structure.json',{'posthoc':True,'used_to_replace_candidate':False,'names':x.shape[1],'sessions':len(x),
        'correlation_participation_ratio':float(eigen.sum()**2/(eigen@eigen)),
        'first_component_variance_fraction':float(eigen[-1]/eigen.sum()),
        'note':'Descriptive return-correlation dimensions, not a count of independent strategy trials, a causal estimate, or a Sharpe bound.'})


def run():
    root=Path('reports/v9');config=json.loads(Path('configs/v9.json').read_text());base=json.loads(Path(config['base_config']).read_text())
    method=json.loads((root/'champion_seal.json').read_text())['method'];tail=[];tail_summary=[]
    for phase in ['development','evaluation']:
        p=pd.read_csv(root/phase/'predictions.csv.gz',dtype={'symbol':str},parse_dates=['date'],float_precision='round_trip');p=p[p.evaluation_period]
        for m,g in p.groupby('method'):
            for horizon,column in [(5,'label'),(20,'label20')]:
                for day,q in g.groupby('date'):
                    q=q.dropna(subset=[column,'prediction']).copy()
                    if len(q)<base['min_cross_section'] or q.prediction.std()<1e-12:continue
                    q['bucket']=np.ceil(q.prediction.rank(pct=True,method='average')*5).clip(1,5).astype(int)
                    q['excess']=q[column]-q[column].mean()
                    for bucket,z in q.groupby('bucket'):
                        tail.append({'phase':phase,'method':m,'horizon':horizon,'date':day,'quintile':int(bucket),'names':len(z),
                            'mean_return':float(z[column].mean()),'mean_excess':float(z.excess.mean())})
    tails=pd.DataFrame(tail);tails.to_csv(root/'quintile_daily.csv.gz',index=False,float_format='%.17g')
    for (phase,m,horizon,bucket),g in tails.groupby(['phase','method','horizon','quintile']):
        tail_summary.append({'phase':phase,'method':m,'horizon':int(horizon),'quintile':int(bucket),'days':len(g),
            'mean_return':float(g.mean_return.mean()),'mean_excess':float(g.mean_excess.mean()),
            'excess_ci95':block_mean_interval(g.mean_excess,17,1000,20)})
    write_json(root/'quintile_summary.json',{'posthoc':True,'used_to_replace_candidate':False,'rows':tail_summary,
        'note':'Prediction-sorted quintiles, not buffered holdings. Conditional overlapping H5/H20 return diagnostics; no new trading rule or causal claim.'})
    panel,adj,calendar,symbols,manifest,quality=load();groups=[]
    for phase,years in [('development',config['development_years']),('evaluation',config['evaluation_years'])]:
        for year in years:
            for half in [1,2]:
                fold=f'{year}H{half}';name=fold+'-'+method;public=root/phase/'evidence'/name;checkpoint=Path('outputs/v9')/phase/'evidence'/name/'models.pkl'
                assert sha256(checkpoint)==json.loads((public/'checkpoint.json').read_text())['sha256']
                with checkpoint.open('rb') as f:models=pickle.load(f)
                _,_,_,q,_=split(panel,year,half,config['train_months'],base)
                q.loc[q.exit_date>pd.Timestamp(f'{max(years)}-12-31'),'label']=np.nan
                for group,score in predict(q,models,base).items():
                    d=daily_ic(q,score,base['min_cross_section']);d['phase']=phase;d['fold']=fold;d['group']=group;groups.append(d)
    gd=pd.concat(groups,ignore_index=True);gd.to_csv(root/'group_daily_ic.csv',index=False,float_format='%.17g')
    write_json(root/'group_signal_summary.json',{'posthoc':True,'method':method,'used_to_replace_candidate':False,
        'phases':{phase:{group:ic_summary(g,base) for group,g in d.groupby('group')} for phase,d in gd.groupby('phase')}})
    news=pd.read_csv('data/v7/news.csv.gz',dtype={'symbol':str},parse_dates=['research_available_date']);rows=[]
    for start,end in [('2018-01-01','2021-12-31'),('2022-01-01','2024-12-31')]:
        days=pd.date_range(start,end);s=news[(news.research_available_date>=start)&(news.research_available_date<=end)]
        counts=s.groupby(['research_available_date','symbol']).size().reindex(pd.MultiIndex.from_product([days,symbols]),fill_value=0)
        rows.append({'start':start,'end':end,'headlines':len(s),'calendar_stock_days':len(counts),
            'mean_headlines_per_stock_calendar_day':float(counts.mean()),'zero_headline_stock_calendar_day_fraction':float((counts==0).mean())})
    write_json(root/'news_coverage_shift.json',{'posthoc':True,'used_to_replace_candidate':False,'periods':rows,
        'note':'Calendar-day availability counts, including weekends; not a causal decomposition of performance. Same frozen title-only cache.'})
    risk_structure()
    print(json.dumps({'posthoc_diagnostics_completed':True,'candidate_unchanged':True,'quintile_daily_rows':len(tails),'group_ic_rows':len(gd)}),flush=True)
if __name__=='__main__':run()
