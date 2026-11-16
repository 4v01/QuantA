import json
from pathlib import Path
import numpy as np
import pandas as pd
from alphacontextlab.v10_clock.ledger import simulate,apply_cost,pool_turnover
from alphacontextlab.v9.portfolio import simulate as v9_simulate


def setup():
    c=json.loads(Path('configs/v10_clock.json').read_text());b=json.loads(Path('configs/v7.json').read_text());b['min_cross_section']=3;b['top_fraction']=.25;b['buffer_exit_fraction']=.5
    days=pd.bdate_range('2019-01-01',periods=150);symbols=list('abcd');rng=np.random.default_rng(3)
    tape=pd.concat([pd.DataFrame({'date':days,'symbol':s,'close_return':.001,'next_open_return':rng.normal(.001,.01,len(days)),'open_observed':True}) for s in symbols])
    p=pd.DataFrame({'date':np.repeat(days[125:-1],4),'symbol':symbols*24,'prediction':np.tile([1.,2.,3.,4.],24),'expected_excess20':0.,'label':99.})
    return c,b,days,symbols,tape,p


def test_single_sleeve_phase_zero_reproduces_v9_accounting():
    c,b,days,symbols,tape,p=setup()
    gross,w,accounts,s=simulate(p,tape,symbols,b,c,'phase0')
    for cost in c['cost_bps']:
        old_config=json.loads(Path('configs/v9.json').read_text())
        old,_=v9_simulate(p,tape,symbols,b,old_config,'economic_h5','weekly_buffer',cost)
        new=apply_cost(gross,cost)
        np.testing.assert_allclose(new[['gross_return','net_return','nav','turnover']],old[['gross_return','net_return','nav','turnover']],atol=1e-12)


def test_sleeves_start_funded_in_cash_and_deploy_once_per_session():
    c,b,days,symbols,tape,p=setup();gross,w,a,s=simulate(p,tape,symbols,b,c,'stagger5')
    np.testing.assert_allclose(gross.gross_exposure.iloc[:5],[.2,.4,.6,.8,1.],atol=.02)
    assert (a.groupby('date').active.sum()==1).all()
    np.testing.assert_allclose(a.groupby('date').budget_before.sum(),1.,atol=1e-12)
    assert (a.cash_before>=-1e-12).all() and (a.cash_after_return>=-1e-12).all()
    assert (a.target_exposure<=a.budget_before+1e-12).all()


def test_sleeve_budgets_drift_with_wealth_and_are_not_reset_each_day():
    c,b,days,symbols,tape,p=setup();tape['next_open_return']=.1
    gross,w,a,s=simulate(p,tape,symbols,b,c,'stagger5')
    first=a[a.sleeve==0].reset_index(drop=True)
    assert first.budget_before.iloc[1]>.2 and first.budget_before.iloc[4]>.2
    np.testing.assert_allclose(a.groupby('date').budget_after_return.sum(),1.,atol=1e-12)


def test_opposed_internal_intentions_have_zero_external_turnover():
    before=np.array([[.2,0.],[0.,.2]]);after=before[::-1].copy()
    net,gross=pool_turnover(before,after)
    assert net==0 and gross==.8


def test_future_returns_and_test_labels_cannot_choose_initial_holdings():
    c,b,days,symbols,tape,p=setup();gross,w,a,s=simulate(p,tape,symbols,b,c,'stagger5')
    changed=tape.copy();changed['next_open_return']=np.clip(changed.next_open_return*10,-.5,1.);p['label']=-999
    _,w2,_,_=simulate(p,changed,symbols,b,c,'stagger5')
    day=w.date.min();np.testing.assert_allclose(w[w.date==day].weight,w2[w2.date==day].weight)


def test_equal_weight_is_independent_of_flat_forecasts():
    c,b,days,symbols,tape,p=setup();p['prediction']=0
    gross,w,a,s=simulate(p,tape,symbols,b,c,'stagger5',benchmark=True)
    assert gross.gross_exposure.max()>.99
    stock,_,_,_=simulate(p,tape,symbols,b,c,'stagger5',benchmark=False)
    assert stock.gross_exposure.sum()==0
