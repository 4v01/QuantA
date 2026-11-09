import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from alphacontextlab.data import sha256
from alphacontextlab.v2.data import samples,split
from alphacontextlab.v2.execution import candidate_weights,choose,simulate
from alphacontextlab.v2.factors import FactorSearch
from alphacontextlab.v2.models import LSTM,fit_lstm

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def config():return json.loads((ROOT/'configs/v2.json').read_text())


def save_quotes(path,quotes):
    quotes.to_csv(path/'quotes.csv',index=False,float_format='%.12g')
    (path/'manifest.json').write_text(json.dumps({'quotes_sha256':sha256(path/'quotes.csv')}))


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory,config):
    path=tmp_path_factory.mktemp('v2_quotes');dates=pd.date_range('2018-01-01',periods=900,freq='B');rng=np.random.default_rng(7);rows=[]
    for symbol in config['symbols']:
        price=3*np.exp(np.cumsum(rng.normal(0,.01,len(dates))))
        for adjustment in [0,1]:
            for j,date in enumerate(dates):
                rows.append({'symbol':symbol,'date':date,'adjustment':adjustment,'open':price[j]*.999,
                             'close':price[j],'high':price[j]*1.01,'low':price[j]*.99,'volume':int(rng.integers(1000,3000))})
    q=pd.DataFrame(rows);save_quotes(path,q)
    cfg=dict(config);cfg['known_full_session_halts']={};cfg['known_no_open_execution']={}
    x,meta,corr,_=samples(path,cfg)
    return path,q,cfg,x,meta,corr


def test_ten_session_label_and_purge(synthetic):
    _,q,c,x,meta,corr=synthetic
    row=meta.iloc[0];f=q[(q.symbol==row.symbol)&(q.adjustment==1)].set_index('date')
    assert row.label_return==pytest.approx(f.loc[row.label_exit,'open']/f.loc[row.entry_date,'open']-1)
    segment=f.loc[row.entry_date:row.label_exit].open.pct_change().dropna()
    assert len(segment)==10
    assert row.label_downside==pytest.approx(np.sqrt((segment.clip(upper=0)**2).sum()))
    train,val,val_labels,test,refit=split(meta,2021)
    assert meta.iloc[train].label_exit.max()<meta.iloc[val].entry_date.min()
    assert meta.iloc[val_labels].label_exit.max()<meta.iloc[test].entry_date.min()
    assert meta.iloc[refit].label_exit.max()<meta.iloc[test].entry_date.min()
    assert len(test)>int(meta.iloc[test].label_valid.sum())  # retain final intervals without long-horizon labels


def test_future_quotes_do_not_change_past_features_correlation_or_factor_selection(synthetic,tmp_path):
    _,q,c,x,meta,corr=synthetic;cut=pd.Timestamp('2021-01-01');changed=q.copy();future=changed.date>=cut
    for column in ['open','close','high','low','volume']:changed.loc[future,column]*=3
    save_quotes(tmp_path,changed);xx,mm,cc,_=samples(tmp_path,c)
    before=(meta.date<cut).to_numpy();other=(mm.date<cut).to_numpy()
    np.testing.assert_equal(x[before],xx[other]);count=(pd.DatetimeIndex(sorted(q.date.unique()))<cut).sum()
    np.testing.assert_equal(corr[:count],cc[:count])
    train,*_=split(meta,2021);train2,*_=split(mm,2021)
    first=FactorSearch(c['factor_search']).fit(x[train],meta.iloc[train].label_return.to_numpy(),meta.iloc[train].date.dt.year.to_numpy())
    second=FactorSearch(c['factor_search']).fit(xx[train2],mm.iloc[train2].label_return.to_numpy(),mm.iloc[train2].date.dt.year.to_numpy())
    assert first.state()==second.state()


def test_factor_search_bound_and_redundancy(synthetic):
    _,q,c,x,meta,corr=synthetic;train,*_=split(meta,2021)
    search=FactorSearch(c['factor_search']).fit(x[train],meta.iloc[train].label_return.to_numpy(),meta.iloc[train].date.dt.year.to_numpy())
    assert len(search.ledger)==24 and 1<=len(search.selected)<=6
    transformed=search.transform(x[train],'evolved_ridge')
    assert transformed.shape[-1]==8+len(search.selected)
    assert np.isfinite(transformed).all()
    for a in range(len(search.selected)):
        for b in range(a):assert abs(np.corrcoef(transformed[:,-1,8+a],transformed[:,-1,8+b])[0,1])<.9


def test_candidate_isolation_and_order_invariance():
    torch.set_num_threads(2);m=LSTM(3,4,17);x=np.random.default_rng(3).normal(size=(6,20,3)).astype('float32')
    actual=m.predict(x);order=np.array([3,2,1,0,5,4])
    np.testing.assert_allclose(m.predict(x[order])[np.argsort(order)],actual,atol=1e-7)
    np.testing.assert_allclose(m.predict(x[:1]),actual[:1],atol=1e-7)


def test_cost_aware_value_prefers_no_churn():
    mu=np.array([.0004,.0003]);down=np.zeros(2);pre=np.array([0.,1.]);corr=np.eye(2)
    plain={'risk_penalty':0.,'cost_aware':False};costly={'risk_penalty':0.,'cost_aware':True}
    assert np.array_equal(choose(mu,down,corr,pre,plain,.0005)[0],[1,0])
    assert np.array_equal(choose(mu,down,corr,pre,costly,.0005)[0],[0,1])
    assert candidate_weights(4).shape==(11,4)
    assert np.all(candidate_weights(4).sum(1)<=1)


def test_weekly_holding_calendar_keeps_every_day(synthetic):
    _,q,c,x,meta,corr=synthetic;_,_,_,ids,_=split(meta,2021);part=meta.iloc[ids]
    p=np.zeros((len(part),2));p[:,0]=.01
    path=simulate(part,p,corr,c,'weekly_return',5)
    assert len(path)==part.entry_date.nunique()
    assert path.rebalanced.iloc[0]
    expected=part.groupby('entry_date').session.first().to_numpy()%5==0;expected[0]=True
    assert np.array_equal(path.rebalanced.to_numpy(),expected)
    assert (path.turnover>=0).all() and np.isfinite(path.net_return).all()


def test_lstm_restores_best_validation_epoch(config):
    c=dict(config);c.update(max_epochs=4,patience=1,batch_size=8)
    rng=np.random.default_rng(8);x=rng.normal(size=(24,6,3)).astype('float32');y=rng.normal(0,.01,size=(24,2));y[:,1]=abs(y[:,1])
    m=LSTM(3,4,17);history,best=fit_lstm(m,x[:16],y[:16],c,c['neural_candidates'][0],17,(x[16:],y[16:]))
    restored=np.sqrt(np.mean((m.predict(x[16:])-y[16:])**2))
    assert restored==pytest.approx(best['validation_joint_rmse'])
    assert 1<=best['epoch']<=len(history)<=4


def test_confirmation_download_requires_seal(tmp_path,monkeypatch):
    from alphacontextlab.v2 import study
    monkeypatch.setattr('sys.argv',['study','fetch-confirmation','--outputs',str(tmp_path)])
    def forbidden(*args,**kwargs):raise AssertionError('Network reached without a seal')
    monkeypatch.setattr(study,'fetch',forbidden)
    with pytest.raises(ValueError,match='Seal champion'):study.main()
