import json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from alphacontextlab.v7.data import read_title_archive,aligned_financial
from alphacontextlab.v7.models import validation_diagnostics,group_spec,fit_method
from alphacontextlab.v7.screen import choose_target,simulate
from alphacontextlab.v7.study import split,ic_summary
from alphacontextlab.v6.data import FINANCIAL,NEWS_FEATURES,financial_features
from alphacontextlab.v4.data import FEATURES


def config():return json.loads((Path(__file__).parents[1]/'configs/v7.json').read_text())


def test_generated_summaries_cannot_enter_archive_features(tmp_path):
    records=[{'datetime':'2022-01-03 14:00:00','title':'公司增长','summary':'未来收益999'},
             {'datetime':'2022-01-03 14:00:00','title':'公司增长','summary':'另一个摘要'},
             {'datetime':'2022-01-04','title':float('nan'),'summary':'不能当成标题'}]
    paths=[]
    for k in [0,1]:
        p=tmp_path/f'{k}.zip'
        with zipfile.ZipFile(p,'w') as z:z.writestr('dir/600000.SH_x.jsonl','\n'.join(json.dumps(r,ensure_ascii=False) for r in records))
        paths.append(p)
        for r in records:r['summary']='被修改的任意未来文本'
    a,counts=read_title_archive(paths[0],{'600000'});b,_=read_title_archive(paths[1],{'600000'})
    pd.testing.assert_frame_equal(a,b);assert len(a)==1 and a.research_available_date.iloc[0]==pd.Timestamp('2022-01-04')
    assert counts['missing_title']==1


def test_no_unannounced_financial_period_in_alignment():
    rows=[]
    for period,notice,value in [('2020-03-31','2020-04-30',2.),('2020-06-30','2020-08-30',3.)]:
        r={v:value for v in FINANCIAL.values()};r.update({'REPORT_DATE':period,'NOTICE_DATE':notice,'UPDATE_DATE':notice});rows.append(r)
    d=pd.DataFrame(rows);days=pd.to_datetime(['2020-05-01','2020-08-01','2020-08-31']);base=financial_features(d,days)
    a=aligned_financial(d,days,base);assert a.aligned_eps.iloc[0]==8. and a.aligned_eps.iloc[1]==8. and a.aligned_eps.iloc[2]==6.


def test_stability_and_reliability_use_signed_past_blocks(monkeypatch):
    c=config();x=np.repeat([.08,-.08,.08,-.08],25)
    monkeypatch.setattr('alphacontextlab.v7.models.daily_ic',lambda *args:pd.DataFrame({'rank_ic':x}))
    d=validation_diagnostics(None,None,c);assert d['reliability']==0 and d['stable_score']<0
    x[:]=.08;d=validation_diagnostics(None,None,c);assert 0<d['reliability']<1


def test_flat_scores_go_to_cash_not_alphabetical_stocks():
    c=config();c['min_cross_section']=3
    target,missing=choose_target([0.,0.,0.,0.],np.array([.5,.5,0,0]),c)
    assert target.sum()==0 and not missing


def test_buffer_keeps_holdings_inside_predeclared_exit_band():
    c=config();c.update({'min_cross_section':5,'top_fraction':.2,'buffer_exit_fraction':.4})
    scores=np.arange(10,0,-1,dtype=float);old=np.zeros(10);old[[2,3]]=.5
    plain,_=choose_target(scores,old,c,False);buffer,_=choose_target(scores,old,c,True)
    assert set(np.flatnonzero(plain))=={0,1} and set(np.flatnonzero(buffer))=={2,3}
    assert abs(buffer-old).sum()<abs(plain-old).sum()


def test_twenty_day_targets_are_purged_for_every_method():
    c=config();dates=pd.bdate_range('2018-01-01','2021-12-31');n=len(dates)
    d=pd.DataFrame({'date':np.repeat(dates,40),'symbol':list(map(str,range(40)))*n,'eligible':True,'target':.1,'target20':.1,
                    'exit20':np.repeat(pd.Series(dates).shift(-21),40)})
    a,b,r,q,info=split(d,2021,1,18,c)
    assert a.exit20.max()<pd.Timestamp(info['validation_start']) and b.exit20.max()<pd.Timestamp(info['test_start']) and r.exit20.max()<q.date.min()


def test_undefined_ic_is_reported_as_coverage_loss():
    d=pd.DataFrame({'rank_ic':[np.nan]*20,'fold':['a']*20});m=ic_summary(d,config())
    assert m['rank_ic'] is None and m['ic_coverage']==0 and m['coverage_adjusted_ic']==0


def test_method_forecasts_ignore_test_labels_and_do_not_flip_signed_models():
    c=config();c.update({'ridge_alphas':[1.],'min_cross_section':10});c['ranker'].update({'max_rounds':10,'early_stopping':2,'min_child_samples':30});rng=np.random.default_rng(3)
    from alphacontextlab.v7.data import ALIGNED,CONTEXT
    columns=list(dict.fromkeys(FEATURES+list(FINANCIAL)+NEWS_FEATURES+ALIGNED+CONTEXT))
    def frame(start,days):
        x=rng.normal(size=(days*40,len(columns)));d=pd.DataFrame(x,columns=columns)
        d['date']=np.repeat(pd.bdate_range(start,periods=days),40);d['symbol']=list(map(str,range(40)))*days
        d['target']=x[:,0]*.1+rng.normal(0,.01,len(x));d['target20']=d.target*.5;d['label']=d.target;return d
    a=frame('2018-01-01',30);b=frame('2019-01-01',20);q=frame('2020-01-01',10);r=pd.concat([a,b])
    for method in ['signed_stable','reliable','aligned_fund','kaggle_rank']:
        p,_,_,s=fit_method(a,b,r,q,method,c);other=q.copy();other[['target','target20','label']]*=-1000
        p2,_,_,_=fit_method(a,b,r,other,method,c)
        for group in p:np.testing.assert_allclose(p[group],p2[group],atol=1e-10)
        assert all(x['selected']['direction']==1 for x in s.values())
        np.testing.assert_allclose(p['fusion'],sum(c['weights'][g]*p[g] for g in c['weights']))


def test_cash_screen_has_zero_returns_and_turnover():
    c=config();c.update({'min_cross_section':3,'top_fraction':.2});days=pd.bdate_range('2020-01-01',periods=20);symbols=list('abcd')
    adj={s:pd.DataFrame({'date':days,'open':np.linspace(10,20,len(days))}) for s in symbols}
    p=pd.DataFrame({'date':np.repeat(days[:-1],4),'symbol':symbols*19,'prediction':0.})
    result=simulate(p,adj,days,symbols,c,20)
    assert (result.net_return==0).all() and result.turnover.sum()==0 and (result.nav==1).all()
