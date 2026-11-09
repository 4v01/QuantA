import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphacontextlab.v3.data import causal_prices,Market,load,economic_holding_return
from alphacontextlab.data import sha256
from alphacontextlab.v3.ledger import plan_orders,trade,simulate
from alphacontextlab.v3.portfolio import solve
from alphacontextlab.v3.statistics import centered_max_check,paired_excess

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def config():return json.loads((ROOT/'configs/v3.json').read_text())


def test_causal_dividend_split_indices_and_future_append():
    dates=pd.date_range('2022-01-03',periods=5,freq='B')
    raw=pd.DataFrame({k:[10.,9.,9.,4.5,4.5] for k in ['open','close','high','low']},index=dates)
    events=[{'kind':'dividend','effective_date':str(dates[1].date()),'cash_per_share':1.},
            {'kind':'split','effective_date':str(dates[3].date()),'ratio':2.}]
    tr,_,_=causal_prices(raw,events)
    np.testing.assert_allclose(tr.close,1.)
    first,_,_=causal_prices(raw,events[:1]);pd.testing.assert_frame_equal(tr.iloc[:3],first.iloc[:3])
    changed=raw.copy();changed.iloc[4]*=10;future,_,_=causal_prices(changed,events)
    pd.testing.assert_frame_equal(tr.iloc[:4],future.iloc[:4])


def test_ex_date_entry_does_not_inherit_previous_holders_dividend():
    dates=pd.date_range('2024-01-02',periods=3,freq='B')
    events=[{'kind':'dividend','record_date':str(dates[0].date()),'effective_date':str(dates[1].date()),'cash_per_share':1.}]
    values=economic_holding_return([10.,9.,11.],np.ones(3),events,dates,1)
    assert values[0]==pytest.approx(0.)
    assert values[1]==pytest.approx(11/9-1)


def test_convex_epigraph_solution_matches_analytic_case(config):
    w,status=solve(np.array([.01]),np.array([[.0002]]),np.array([0.]),10.,.0005,config)
    assert status['success']
    assert w[0]==pytest.approx((.01-.0005)/(10*.0002*10),abs=1e-5)
    assert 0<=w.sum()<=1


def test_lots_commission_and_no_negative_cash(config):
    c=dict(config);c['symbols']=['510300'];c['_execution_date']='2025-01-03'
    requested=plan_orders(np.array([0]),np.array([3.]),np.array([1.]),1000.,100)
    assert requested.tolist()==[300]
    q,cash,record=trade(np.array([0]),1000.,requested,np.array([3.]),np.array([3.]),np.array([True]),np.array([False]),np.array([1e9]),config['scenarios']['primary'],c)
    assert q[0]%100==0 and cash>=0 and record['commission']==5.
    assert cash+q[0]*3==pytest.approx(1000-record['commission']-record['slippage']+record['rounding_pnl'])


def test_split_cancels_stale_orders_and_halts_never_fill(config):
    c=dict(config);c.update(symbols=['510500'],_execution_date='2022-08-29')
    for tradable,split in [(False,False),(True,True)]:
        q,cash,r=trade(np.array([100]),1000.,np.array([-100]),np.array([6.3]),np.array([6.3]),np.array([tradable]),np.array([split]),np.array([1e9]),config['scenarios']['primary'],c)
        assert q[0]==100 and cash==1000 and r['blocked_orders']==1


def toy_market(config):
    dates=pd.date_range('2025-01-02',periods=10,freq='B');p=np.full((10,1),10.)
    cov=np.tile(np.array([[[.0001]]]),(10,1,1));s=np.ones((10,1));tr=np.ones((10,1))
    market=Market(dates,['510300'],p.copy(),p.copy(),np.ones((10,1),bool),np.full((10,1),1e9),s,{},[],tr.copy(),tr.copy(),cov,cov.copy(),np.ones(10),[])
    meta=pd.DataFrame({'entry_date':dates[1:8],'exit_date':dates[2:9],'date':dates[:7],
                       'symbol':['510300']*7,'session':np.arange(7)})
    c=copy.deepcopy(config);c.update(symbols=['510300'],liquidation_sessions=1)
    return market,meta,c


def test_share_orders_are_fixed_before_future_open(config):
    market,meta,c=toy_market(config);prediction=np.tile([.01,.01],(len(meta),1));scenario=dict(c['scenarios']['free']);scenario['capital']=10000.
    _,orders=simulate(meta,prediction,market,'csi300',None,scenario,c)
    changed=copy.deepcopy(market);changed.opening[1,0]=10.2
    _,later=simulate(meta,prediction,changed,'csi300',None,scenario,c)
    assert orders.iloc[0].requested==later.iloc[0].requested==1000
    assert later.iloc[0].filled<orders.iloc[0].filled


def test_dividend_receivable_is_not_credited_twice(config):
    market,meta,c=toy_market(config);market.dividends=[{'asset':0,'record':2,'ex':3,'pay':5,'cash':1.}]
    market.opening[3:,0]=9.;market.closing[3:,0]=9.
    pred=np.tile([.01,.01],(len(meta),1));scenario=dict(c['scenarios']['free']);scenario['capital']=10000.
    path,orders=simulate(meta,pred,market,'csi300',None,scenario,c)
    assert path.iloc[-1].nav_end==pytest.approx(10000.)
    assert path.receivable_after.max()==1000.
    assert path.iloc[-1].receivable_after==0.


def test_split_ceil_and_odd_lot_exit_preserve_economic_value(config):
    market,meta,c=toy_market(config);market.split_ratio[3,0]=1.14539;market.split_rounding[3,0]='ceil'
    market.opening[3:,0]=8.73;market.closing[3:,0]=8.73
    pred=np.tile([.01,.01],(len(meta),1));scenario=dict(c['scenarios']['free']);scenario['capital']=10000.
    path,orders=simulate(meta,pred,market,'csi300',None,scenario,c)
    assert path.iloc[-1].nav_end==pytest.approx(1146*8.73)
    assert orders.iloc[-1].filled==-1146
    assert (orders.shares_after>=0).all()


def test_adv_capacity_uses_only_lagged_liquidity(config):
    c=dict(config);c.update(symbols=['510300'],_execution_date='2025-01-03')
    q,cash,r=trade(np.array([0]),100000.,np.array([10000]),np.array([10.]),np.array([10.]),np.array([True]),np.array([False]),np.array([100000.]),config['scenarios']['free'],c)
    assert q[0]==100 and r['liquidity_capped_orders']==1


def test_centered_max_check_does_not_promote_null_or_negative_families():
    assert centered_max_check(np.zeros((100,5)),50)['conditional_pvalue']==1.
    assert centered_max_check(np.full((100,5),-.001),50)['conditional_pvalue']==1.


def test_paired_intervals_respect_equal_paths_and_multiplicity():
    r=np.linspace(-.01,.01,100)
    out=paired_excess(r,r,100,comparisons=3)
    assert out['bonferroni_interval']==[0.,0.]
    assert out['interval_coverage']==pytest.approx(1-.05/3)


def test_raw_pipeline_future_quotes_and_future_events_do_not_change_past(config,tmp_path):
    dates=pd.date_range('2019-01-02',periods=380,freq='B');rng=np.random.default_rng(4);rows=[]
    for symbol in config['symbols']:
        prices=3*np.exp(np.cumsum(rng.normal(0,.005,len(dates))))
        for j,date in enumerate(dates):rows.append({'date':date,'symbol':symbol,'adjustment':0,'open':prices[j]*.999,
            'close':prices[j],'high':prices[j]*1.01,'low':prices[j]*.99,'volume':10000+j,'amount':1e8+j})
    q=pd.DataFrame(rows);c=copy.deepcopy(config);c['known_full_session_halts']={};c['known_no_open_execution']={}
    def build(quotes,events):
        quotes.to_csv(tmp_path/'quotes.csv',index=False,float_format='%.12g')
        (tmp_path/'manifest.json').write_text(json.dumps({'quotes_sha256':sha256(tmp_path/'quotes.csv')}))
        (tmp_path/'events.json').write_text(json.dumps({'events':events,'sources':[],'limitations':[]}))
        return load(tmp_path,tmp_path,c)
    x,m,market,_=build(q,[]);cut=dates[300];changed=q.copy()
    for key in ['open','close','high','low']:changed.loc[changed.date>=cut,key]*=2
    event={'kind':'dividend','symbol':'510300','record_date':str(dates[319].date()),'effective_date':str(dates[320].date()),
           'pay_date':str(dates[323].date()),'cash_per_share':.02}
    xx,mm,other,_=build(changed,[event])
    np.testing.assert_array_equal(x[(m.date<cut).to_numpy()],xx[(mm.date<cut).to_numpy()])
    np.testing.assert_array_equal(market.lw_cov[:300],other.lw_cov[:300])
    np.testing.assert_array_equal(market.sample_cov[:300],other.sample_cov[:300])
