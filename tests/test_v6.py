from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from alphacontextlab.v6.data import parse_news,news_features,financial_features,NEWS_FEATURES,FINANCIAL
from alphacontextlab.v6.models import percentile,fit,GROUPS
from alphacontextlab.v6.study import splits


def config():return json.loads((Path(__file__).parents[1]/'configs/v6.json').read_text())


def test_headerless_news_preserves_first_article_and_delays_date_only_news(tmp_path):
    p=tmp_path/'sample.xlsx';pd.DataFrame([['2020-01-02','增长','来源'],['2020-01-03','亏损','来源']]).to_excel(p,index=False,header=False)
    d=parse_news(p,'600000');assert len(d)==2 and d.title.iloc[0]=='增长'
    assert d.research_available_date.iloc[0]==pd.Timestamp('2020-01-03')
    f=news_features(d,pd.date_range('2020-01-01','2020-01-05'))
    assert f.loc['2020-01-02','news_count_1']==0 and f.loc['2020-01-03','news_count_1']>0


def test_future_headlines_cannot_rewrite_past_features():
    d=pd.DataFrame({'published_date':pd.to_datetime(['2020-01-01','2020-02-01']),
                     'research_available_date':pd.to_datetime(['2020-01-02','2020-02-02']),'title':['增长','亏损']})
    cal=pd.date_range('2020-01-01','2020-02-10');a=news_features(d,cal);d.loc[1,'title']='增长'*20;b=news_features(d,cal)
    pd.testing.assert_frame_equal(a.loc[:'2020-02-01'],b.loc[:'2020-02-01'])


def test_financial_update_date_prevents_using_later_restatement_early():
    row={v:float(i+1) for i,v in enumerate(FINANCIAL.values())};row.update({'REPORT_DATE':'2019-12-31','NOTICE_DATE':'2020-03-01','UPDATE_DATE':'2020-08-01'})
    d=pd.DataFrame([row]);cal=pd.to_datetime(['2020-03-02','2020-08-01','2020-08-02']);f=financial_features(d,cal)
    assert f.iloc[:2].isna().all().all();assert f.iloc[2].notna().all()


def test_older_period_late_update_does_not_overwrite_newer_report():
    rows=[]
    for period,notice,update,value in [('2019-12-31','2020-03-01','2020-09-01',1.),('2020-03-31','2020-04-30','2020-04-30',2.)]:
        r={v:value for v in FINANCIAL.values()};r.update({'REPORT_DATE':period,'NOTICE_DATE':notice,'UPDATE_DATE':update});rows.append(r)
    f=financial_features(pd.DataFrame(rows),pd.to_datetime(['2020-09-03']));assert (f.iloc[0]==2.).all()


def test_percentile_is_same_date_only_and_direction_not_test_selected():
    f=pd.DataFrame({'date':['a','a','b','b']});a=percentile(f,[1,2,3,4]);b=percentile(f,[1,2,-3,-4]);np.testing.assert_equal(a[:2],b[:2])


def test_group_models_ignore_test_labels_and_keep_fixed_weights():
    c=config();c.update({'ridge_alphas':[1.],'min_cross_section':10});rng=np.random.default_rng(19)
    def sample(start,days):
        cols=list(dict.fromkeys(x for group in GROUPS.values() for x in group));x=rng.normal(size=(days*40,len(cols)));d=pd.DataFrame(x,columns=cols)
        d['date']=np.repeat(pd.bdate_range(start,periods=days),40);d['symbol']=list(map(str,range(40)))*days
        d['target']=x[:,0]*.1+x[:,7]*.1+rng.normal(0,.01,len(x));d['label']=d.target;return d
    a=sample('2018-01-01',30);b=sample('2019-01-01',10);q=sample('2020-01-01',10);r=pd.concat([a,b])
    p,_,_,_=fit(a,b,r,q,c);q2=q.copy();q2[['label','target']]*=-1000;p2,_,_,_=fit(a,b,r,q2,c)
    for group in p:np.testing.assert_allclose(p[group],p2[group],atol=1e-10)
    np.testing.assert_allclose(p['fusion'],sum(c['weights'][g]*p[g] for g in c['weights']))


def test_entity_grounded_documents_exclude_other_firms_and_future_news():
    d=pd.DataFrame({'research_available_date':pd.to_datetime(['2020-01-02','2020-01-02','2020-02-02']),
                    'title':['浦发银行盈利','民生银行亏损','浦发银行未来增长']})
    calendar=pd.date_range('2020-01-01','2020-02-10')
    a=news_features(d,calendar,True,'浦发银行',True)
    assert a.loc['2020-01-02','news_count_1']==pytest.approx(np.log(2))
    assert a.loc['2020-01-01','news_document']==''
    assert '民生' not in a.loc['2020-01-02','news_document'] and '未来' not in a.loc['2020-02-01','news_document']
    d.loc[2,'title']='浦发银行完全不同的未来标题';b=news_features(d,calendar,True,'浦发银行',True)
    pd.testing.assert_frame_equal(a.loc[:'2020-02-01'],b.loc[:'2020-02-01'])


def test_text_vocabulary_is_fitted_only_on_training_documents():
    from alphacontextlab.v6.models import text_estimator
    rng=np.random.default_rng(11);d=pd.DataFrame(rng.normal(size=(30,len(NEWS_FEATURES))),columns=NEWS_FEATURES)
    d['news_document']=['浦发银行盈利增长']*15+['浦发银行业绩改善']*15
    model=text_estimator(1.).fit(d,np.arange(30)*.01)
    vocabulary=dict(model.steps[0][1].named_transformers_['text'].vocabulary_)
    held=d.iloc[:2].copy();held['news_document']='测试独有外星事件';model.predict(held)
    assert '外星' not in vocabulary and vocabulary==model.steps[0][1].named_transformers_['text'].vocabulary_
