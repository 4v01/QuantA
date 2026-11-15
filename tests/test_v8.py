import json
from pathlib import Path
import numpy as np
import pandas as pd
from alphacontextlab.v8.models import calibrate_slope, nested_split
from alphacontextlab.v8.portfolio import net_edge_target, risk_state, simulate
from alphacontextlab.v8.study import select_policy, return_metrics
from alphacontextlab.v8.benchmark import simulate_benchmark


def config():
    return json.loads(Path('configs/v8.json').read_text())


def base():
    return json.loads(Path('configs/v7.json').read_text())


def test_negative_calibration_is_cash_not_sign_search():
    days=pd.bdate_range('2020-01-01',periods=4); x=np.tile([-1.,0,1],4)
    d=pd.DataFrame({'date':np.repeat(days,3),'label':-x*.1,'exit_date':np.repeat(days+pd.Timedelta(days=7),3)})
    c=calibrate_slope(d,x,config()); assert c['raw_slope']<0 and c['slope']==0 and not c['sign_flipped']


def test_market_returns_do_not_leak_into_excess_calibration():
    days=pd.bdate_range('2020-01-01',periods=4); x=np.tile([-1.,0,1],4)
    d=pd.DataFrame({'date':np.repeat(days,3),'label':x*.02,'exit_date':np.repeat(days+pd.Timedelta(days=7),3)})
    c=calibrate_slope(d,x,config()); d['label']+=np.repeat([1.,-2.,3.,-4.],3)
    assert abs(calibrate_slope(d,x,config())['slope']-c['slope'])<1e-12


def test_nested_boundaries_purge_label_exits():
    days=pd.bdate_range('2018-01-01','2021-12-31'); n=len(days)
    p=pd.DataFrame({'date':np.repeat(days,40),'symbol':list(map(str,range(40)))*n,'eligible':True,
                    'target':.1,'target20':.1,'exit20':np.repeat(pd.Series(days).shift(-21),40)})
    a,b,r0,cal,r,q,info=nested_split(p,2021,1,base(),config())
    assert a.exit20.max()<b.date.min() and b.exit20.max()<cal.date.min()
    assert r0.exit20.max()<cal.date.min() and cal.exit20.max()<q.date.min() and r.exit20.max()<q.date.min()


def test_net_edge_all_cash_when_round_trip_cost_exceeds_forecast():
    c=config(); n=10; old=np.zeros(n); cov=np.eye(n)*.0001
    w,_=net_edge_target(np.full(n,.003),old,cov,1.,20,c)
    assert w.sum()<1e-7


def test_optimizer_constraints_and_cost_sensitivity():
    c=config(); n=15; old=np.zeros(n); mu=np.linspace(.001,.02,n); cov=np.eye(n)*.0004
    cheap,_=net_edge_target(mu,old,cov,.7,10,c); costly,_=net_edge_target(mu,old,cov,.7,30,c)
    assert cheap.min()>=0 and cheap.max()<=.1+1e-8 and cheap.sum()<=.7+1e-7
    assert costly.sum()<=cheap.sum()+1e-7


def test_known_close_only_risk_state():
    c=config(); rng=np.random.default_rng(7); history=rng.normal(0,.01,(200,5))
    state=risk_state(history[:150],c); future=history.copy(); future[150:]*=1000
    again=risk_state(future[:150],c)
    np.testing.assert_array_equal(state['covariance'],again['covariance'])
    assert state['trend']==again['trend'] and state['market_expected']==again['market_expected']


def test_allocator_handles_correlated_covariance_and_drifted_weights():
    c=config(); n=58; rng=np.random.default_rng(17)
    for _ in range(10):
        cov=np.full((n,n),.0001)+np.eye(n)*1e-8
        old=rng.dirichlet(np.ones(n)); mu=rng.normal(.002,.004,n)
        w,info=net_edge_target(mu,old,cov,1.,20,c)
        assert info['success'] and w.sum()<=1+1e-7 and (w>=0).all() and w.max()<=.1+1e-7
        assert info['optimality_residual']<1e-10


def test_allocator_matches_analytic_separable_solution_with_existing_positions():
    c=config(); old=np.array([.02,.03,.04]); mu=np.array([.007,.001,.004]); variance=np.array([.004,.006,.008])
    q=variance*5*c['risk_aversion']; cost=.002; unconstrained=(mu-cost)/q
    analytic=np.clip(old+np.sign(unconstrained-old)*np.maximum(abs(unconstrained-old)-cost/q,0),0,.1)
    w,_=net_edge_target(mu,old,np.diag(variance),1.,20,c)
    np.testing.assert_allclose(w,analytic,atol=1e-8)


def test_future_open_return_cannot_change_initial_target():
    c=config(); b=base(); b['min_cross_section']=3; days=pd.bdate_range('2020-01-01',periods=150); symbols=list('abcd')
    rng=np.random.default_rng(7); rows=[]
    for s in symbols:
        rows.append(pd.DataFrame({'date':days,'symbol':s,'close_return':rng.normal(.001,.01,len(days)),
                                 'next_open_return':.001,'open_observed':True}))
    tape=pd.concat(rows); pred=pd.DataFrame({'date':np.repeat(days[125:-1],4),'symbol':symbols*24,
                                           'prediction':np.tile([1.,2.,3.,4.],24),'expected_excess':np.tile([.01,.02,.03,.04],24)})
    _,w=simulate(pred,tape,symbols,b,c,'net_edge',20)
    changed=tape.copy(); changed.loc[changed.date>=days[126],'next_open_return']=.8
    _,w2=simulate(pred,changed,symbols,b,c,'net_edge',20)
    np.testing.assert_allclose(w[w.date==days[126]].weight,w2[w2.date==days[126]].weight)


def test_all_cash_not_eligible_for_sharpe_selection():
    c=config(); rows=[{'policy':p,'cost_bps':20,'net_sharpe':100 if p=='net_edge' else .3,
                      'mean_exposure':0 if p=='net_edge' else .5,'noncash_days':0 if p=='net_edge' else 200} for p in c['policies']]
    assert select_policy(pd.DataFrame(rows),c)['policy']=='buffer'


def test_return_r2_is_not_squared_correlation():
    d=pd.DataFrame({'label':[1.,2.,3.],'expected_return':[101.,102.,103.]})
    assert return_metrics(d)['r2']<0


def test_equal_weight_control_does_not_depend_on_alpha_dispersion():
    c=config(); b=base(); b['min_cross_section']=3; days=pd.bdate_range('2020-01-01',periods=150); symbols=list('abcd')
    tape=pd.concat([pd.DataFrame({'date':days,'symbol':s,'close_return':.001,'next_open_return':.001,'open_observed':True}) for s in symbols])
    p=pd.DataFrame({'date':np.repeat(days[125:-1],4),'symbol':symbols*24,'prediction':0.,'expected_excess':0.})
    path,w=simulate_benchmark(p,tape,symbols,b,c,20)
    assert path.gross_exposure.max()>0
    p['prediction']=np.arange(len(p)); changed,w2=simulate_benchmark(p,tape,symbols,b,c,20)
    np.testing.assert_allclose(path.net_return,changed.net_return)
    assert w.groupby('date').weight.std().max()<1e-12
