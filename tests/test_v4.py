import copy,json
import numpy as np
import pandas as pd
import pytest
from alphacontextlab.v4.data import symbol_factors,cross_rank,FEATURES,normalize
from alphacontextlab.v4.metrics import daily_ic,signal_gate,paper_gate
from alphacontextlab.v4.study import split,folds
from alphacontextlab.v4.screen import screen_path
from alphacontextlab.v4.sentiment import available_records
from alphacontextlab.v4.paper import prepare
from alphacontextlab.v4.models import x_rerank


def config():
    from pathlib import Path
    return json.loads((Path(__file__).parents[1]/'configs/v4.json').read_text())


def quotes(n=200,symbol='600000'):
    dates=pd.bdate_range('2019-01-01',periods=n);rng=np.random.default_rng(9);p=np.exp(np.cumsum(rng.normal(0,.01,n)))*10
    return pd.DataFrame({'date':dates,'symbol':symbol,'open':p,'close':p*1.001,'high':p*1.01,'low':p*.99,
                         'volume':np.arange(n)+1000.,'amount':p*10000,'turnover':1.})


def test_factor_future_prices_cannot_change_past():
    q=quotes();cut=150;other=q.copy();other.loc[cut:,['open','close','high','low']]*=1.7
    a=symbol_factors(q,q,pd.DatetimeIndex(q.date),5);b=symbol_factors(other,other,pd.DatetimeIndex(q.date),5)
    pd.testing.assert_frame_equal(a[FEATURES].iloc[:cut],b[FEATURES].iloc[:cut])
    assert not np.allclose(a.label.iloc[145:150],b.label.iloc[145:150])


def test_label_next_open_and_purge_not_close_to_close():
    q=quotes();f=symbol_factors(q,q,pd.DatetimeIndex(q.date),5)
    assert f.label.iloc[100]==pytest.approx(q.open.iloc[106]/q.open.iloc[101]-1)
    assert f.exit_date.iloc[100]==q.date.iloc[106]
    assert f.label.tail(6).isna().all()


def test_cross_rank_uses_same_date_only():
    d=pd.DataFrame({'date':['a','a','b','b'],'v':[1.,2.,3.,4.]});other=d.copy();other.loc[2:,'v']*=-100
    pd.testing.assert_series_equal(cross_rank(d,['v']).v.iloc[:2],cross_rank(other,['v']).v.iloc[:2])


def test_missing_session_is_not_deleted_or_forward_filled_into_factors():
    q=quotes();calendar=pd.DatetimeIndex(q.date);sparse=q.drop(100)
    f=symbol_factors(sparse,sparse,calendar,5)
    assert len(f)==200 and not f.known_tradable.iloc[100]
    assert np.isnan(f.ret1.iloc[101])


def test_signal_gate_rejects_large_absolute_daily_ic_with_zero_signed_mean():
    c=config();dates=pd.bdate_range('2020-01-01',periods=240)
    d=pd.DataFrame({'date':dates,'rank_ic':np.tile([.3,-.3],120),'ic':0.,'fold':np.repeat(['a','b','c','d'],60)})
    assert not signal_gate(d,c)['passed']
    assert abs(signal_gate(d,c)['mean_rank_ic'])<1e-10


def test_ic_is_cross_sectional_spearman_not_pooled_time_correlation():
    d=pd.DataFrame({'date':np.repeat(pd.to_datetime(['2020-01-01','2020-01-02']),4),'symbol':list('abcd')*2,
                    'label':[1.,2.,3.,4.,100.,101.,102.,103.]})
    z=daily_ic(d,[4,3,2,1,103,102,101,100],4)
    assert np.allclose(z.rank_ic,-1)


def test_split_removes_overlapping_labels_and_refit_has_no_test_labels():
    c=config();dates=pd.bdate_range('2017-01-01','2021-06-30');n=len(dates)
    d=pd.DataFrame({'date':np.repeat(dates,60),'symbol':list(map(str,range(60)))*n,'eligible':True,'label':.01,'target':.1,
                    'exit_date':np.repeat(pd.Series(dates).shift(-6),60)})
    _,start,val,begin,end=next(folds(c));a,b,r,q=split(d,start,val,begin,end,c)
    assert a.exit_date.max()<val and b.exit_date.max()<begin and r.exit_date.max()<begin
    assert q.date.min()>=begin


def test_screen_future_scores_cannot_rewrite_past_positions_and_nav():
    c=config();c.update({'symbols':['a','b'],'min_cross_section':2,'top_fraction':.5,'rebalance_sessions':1})
    days=pd.bdate_range('2020-01-01',periods=30)
    adjusted={s:pd.DataFrame({'date':days,'open':np.linspace(10,12 if s=='a' else 9,30)}) for s in c['symbols']}
    p=pd.DataFrame({'date':np.repeat(days[:-1],2),'symbol':['a','b']*29,'prediction':[1.,0.]*29})
    other=p.copy();other.loc[other.date>=days[15],'prediction']*=-1
    a=screen_path(p,adjusted,days,c['symbols'],c,20);b=screen_path(other,adjusted,days,c['symbols'],c,20)
    pd.testing.assert_frame_equal(a.iloc[:14],b.iloc[:14]);assert a.nav.iloc[-1]!=b.nav.iloc[-1]
    np.testing.assert_allclose(a.nav,np.cumprod(1+a.net_return))


def test_high_sharpe_cannot_bypass_linear_or_data_gate():
    gate=paper_gate(False,{'passed':True},{'net_sharpe':3.},{'2025':{'cumulative_return':.5}},
                    {'pit_universe_certified':False,'cash_event_ledger_certified':False},config())
    assert gate['status']=='blocked' and 'linear_signal_stability_gate_failed' in gate['reasons']


def test_paper_cannot_be_enabled_by_only_editing_status(tmp_path):
    r=tmp_path/'reports';r.mkdir()
    (r/'paper_gate.json').write_text(json.dumps({'status':'eligible_for_local_paper','selected_model':'ridge','reasons':[]}))
    (r/'data_quality.json').write_text(json.dumps({'pit_universe_certified':False,'cash_event_ledger_certified':False}))
    (r/'provenance.json').write_text(json.dumps({'source_commit':'x'}))
    record,_=prepare(r,tmp_path/'state');assert record['status']=='blocked' and record['paper_orders_sent']==0


def test_news_requires_both_publication_and_first_seen():
    records=[{'published_at':'2020-01-01T00:00:00Z','first_seen_at':'2026-11-10T00:00:00Z'}]
    assert available_records(records,'2025-01-01T00:00:00Z')==[]
    assert len(available_records(records,'2026-11-11T00:00:00Z'))==1
    with pytest.raises(ValueError):available_records(records,'2026-11-11')


def test_x_reranker_future_date_independence():
    d=pd.DataFrame({'date':['a','a','b','b'],'vol20':[.1,.2,.3,.4],'amount_ratio20':[.2,.1,.4,.3]})
    a=x_rerank(d,np.array([1.,2.,3.,4.]));b=x_rerank(d,np.array([1.,2.,-3.,-4.]));np.testing.assert_equal(a[:2],b[:2])


def test_all_model_arms_ignore_test_labels():
    from alphacontextlab.v4.models import fit_fold
    c=config();c.update({'min_cross_section':10,'ridge_alphas':[1.],'elasticnet_alphas':[.001],
                         'lightgbm_leaves':[15],'lightgbm_max_rounds':8,'lightgbm_early_stopping':3,
                         'gp_population':20,'gp_generations':1,'gp_training_rows':300})
    rng=np.random.default_rng(12)
    def sample(start,days):
        x=rng.normal(size=(days*60,len(FEATURES)));d=pd.DataFrame(x,columns=FEATURES)
        d['date']=np.repeat(pd.bdate_range(start,periods=days),60);d['symbol']=list(map(str,range(60)))*days
        d['target']=x[:,0]*.1+rng.normal(0,.01,len(x));d['label']=d.target
        return d
    a=sample('2018-01-01',40);b=sample('2019-01-01',10);q=sample('2020-01-01',10);r=pd.concat([a,b])
    p,_,_,_=fit_fold(a,b,r,q,c);q2=q.copy();q2[['label','target']]*=-1000
    p2,_,_,_=fit_fold(a,b,r,q2,c)
    for arm in c['models']:np.testing.assert_allclose(p[arm],p2[arm],rtol=1e-12,atol=1e-12)


def test_constant_validation_candidate_is_invalid_not_a_fabricated_ic():
    from alphacontextlab.v4.models import validation_score
    d=pd.DataFrame({'date':np.repeat(pd.to_datetime(['2020-01-01']),60),'symbol':list(map(str,range(60))),'label':np.linspace(-.1,.1,60)})
    score,direction,raw=validation_score(d,np.ones(60),config())
    assert score==-1 and raw is None and direction==1
