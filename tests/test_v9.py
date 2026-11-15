import json
from pathlib import Path
import numpy as np
import pandas as pd
from alphacontextlab.v9.data import economic_financial
from alphacontextlab.v9.models import calibrate,apply_calibration,predict
from alphacontextlab.v9.portfolio import target


def config():return json.loads(Path('configs/v9.json').read_text())
def base():return json.loads(Path('configs/v7.json').read_text())


def financial():
    return pd.DataFrame([
        {'REPORT_DATE':'2018-03-31','NOTICE_DATE':'2018-04-20','UPDATE_DATE':'2018-04-20','EPSJB':.1,'BPS':2.,'ROEJQ':2.,'MGJYXJJE':.05},
        {'REPORT_DATE':'2018-12-31','NOTICE_DATE':'2019-03-20','UPDATE_DATE':'2019-03-20','EPSJB':1.,'BPS':2.,'ROEJQ':8.,'MGJYXJJE':.5},
        {'REPORT_DATE':'2019-03-31','NOTICE_DATE':'2019-04-20','UPDATE_DATE':'2019-04-20','EPSJB':.2,'BPS':2.2,'ROEJQ':2.,'MGJYXJJE':.1}])


def test_ttm_valuation_uses_only_disclosed_components_and_same_day_price():
    days=pd.to_datetime(['2019-04-20','2019-04-21']); price=pd.Series([10.,10.],index=days)
    d=economic_financial(financial(),days,price)
    assert abs(d.loc[days[0],'earnings_yield']-.1)<1e-12
    assert abs(d.loc[days[1],'earnings_yield']-.11)<1e-12
    assert abs(d.loc[days[1],'book_to_price']-.22)<1e-12 and d.loc[days[1],'roe_annual']==8


def test_valuation_is_invariant_to_common_currency_units():
    day=pd.to_datetime(['2019-04-22']); f=financial(); p=pd.Series([10.],index=day)
    original=economic_financial(f,day,p); f[['EPSJB','BPS','MGJYXJJE']]*=100
    changed=economic_financial(f,day,p*100)
    np.testing.assert_allclose(original,changed,atol=1e-12)


def test_future_financial_revisions_cannot_rewrite_earlier_features():
    day=pd.to_datetime(['2019-04-22']); f=financial(); p=pd.Series([10.],index=day)
    original=economic_financial(f,day,p)
    future=f.iloc[-1:].copy();future['UPDATE_DATE']='2021-01-01';future['EPSJB']=999
    changed=economic_financial(pd.concat([f,future]),day,p)
    np.testing.assert_allclose(original,changed,atol=1e-12)


def test_missing_ttm_component_stays_missing():
    day=pd.to_datetime(['2019-04-22']); f=financial().iloc[1:]
    d=economic_financial(f,day,pd.Series([10.],index=day))
    assert pd.isna(d.earnings_yield.iloc[0])


def test_calibration_excludes_unmatured_and_current_forecasts():
    dates=pd.to_datetime(['2019-12-01']*3+['2019-12-25']*3+['2020-01-02']*3)
    x=np.tile([-1.,0.,1.],3)
    p=pd.DataFrame({'date':dates,'method':'full_window','prediction':x,'label20':x*.02,
                    'exit20':pd.to_datetime(['2019-12-30']*3+['2020-02-01']*6)})
    c=calibrate(p,pd.Timestamp('2020-01-01'),'full_window',config())
    assert c['rows']==3 and abs(c['slope']-.01)<1e-12
    p.loc[p.exit20>pd.Timestamp('2020-01-01'),'label20']=999
    assert calibrate(p,pd.Timestamp('2020-01-01'),'full_window',config())==c


def test_negative_calibration_not_inverted_and_no_arbitrary_positive_cap():
    p=pd.DataFrame({'date':pd.to_datetime(['2019-01-01']*3),'method':'full_window','prediction':[-1.,0.,1.],
                    'label20':[1.,0.,-1.],'exit20':pd.to_datetime(['2019-02-01']*3)})
    assert calibrate(p,pd.Timestamp('2020-01-01'),'full_window',config())['slope']==0
    p['label20']*=-1
    assert calibrate(p,pd.Timestamp('2020-01-01'),'full_window',config())['slope']==.5


def test_twenty_day_covariance_matches_analytic_solution():
    c=config();b=base();b['min_cross_section']=3
    covariance=np.eye(4)*.001;state={'covariance':covariance}; old=np.zeros(4)
    w=target(np.arange(4,dtype=float),np.full(4,.008),old,state,'economic_h20','slow_net_edge',20,b,c)
    np.testing.assert_allclose(w,np.full(4,.04),atol=1e-8)


def test_equal_weight_benchmark_is_independent_of_flat_alpha():
    c=config();b=base();b['min_cross_section']=3
    w=target(np.zeros(4),np.zeros(4),np.zeros(4),None,'equal_weight','equal_slow',20,b,c)
    np.testing.assert_array_equal(w,np.full(4,.25))


def test_classic_forecast_does_not_use_labels():
    columns=['earnings_yield','book_to_price','roe_annual','fin_leverage','momentum_12_1','momentum_6_1','low_volatility_60']
    rng=np.random.default_rng(7);p=pd.DataFrame(rng.normal(size=(40,len(columns))),columns=columns);p['label']=1.
    first=predict(p,{'kind':'classic'},base())['fusion'];p['label']=-999
    np.testing.assert_array_equal(first,predict(p,{'kind':'classic'},base())['fusion'])
