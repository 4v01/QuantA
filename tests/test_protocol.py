import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from alphacontextlab.backtest import execute, forecast_metrics, limit_fraction, simulate, statistics, weights_from_scores
from alphacontextlab.data import FEATURES, Transform, fold_indices, panel, sequences, sha256
from alphacontextlab.models import make_model, parameter_count
from alphacontextlab.study import ridge_fit, ridge_predict

ROOT=Path(__file__).resolve().parents[1]


def test_fold_purges_label_overlap():
    dates=pd.date_range('2019-12-20','2021-01-10',freq='B')
    meta=pd.DataFrame({'entry_date':dates[:-1],'exit_date':dates[1:]})
    train,val,test=fold_indices(meta,2021)
    assert meta.iloc[train].exit_date.max()<pd.Timestamp('2020-01-01')
    assert meta.iloc[val].exit_date.max()<pd.Timestamp('2021-01-01')
    assert meta.iloc[test].entry_date.min()>=pd.Timestamp('2021-01-01')
    assert set(train).isdisjoint(val) and set(val).isdisjoint(test)


def test_transform_is_fit_only_and_deterministic():
    rng=np.random.default_rng(2);x=rng.normal(size=(40,20,8)).astype('float32')
    transform=Transform().fit(x[:20]);saved=json.dumps(transform.state(),sort_keys=True)
    a,t=transform.apply(x[20:]);transform.apply(x[20:]*100000)
    assert json.dumps(transform.state(),sort_keys=True)==saved
    assert t.min()>=0 and t.max()<12
    assert np.array_equal(a,transform.apply(x[20:])[0])


def test_memory_is_causal_and_reloadable(tmp_path):
    config=json.loads((ROOT/'configs/study.json').read_text())
    model=make_model('lstm_memory',8,config,17).eval();x=torch.randn(2,20,8);tokens=torch.randint(0,12,(2,20))
    first=model(x,tokens).detach()
    path=tmp_path/'weights.pt';torch.save(model.state_dict(),path)
    copy=make_model('lstm_memory',8,config,29).eval();copy.load_state_dict(torch.load(path,weights_only=True))
    assert torch.equal(first,copy(x,tokens).detach())
    # Lookup uses only final two/three historical states, never an appended future.
    changed=tokens.clone();changed[:,:-3]=11-changed[:,:-3]
    assert all(torch.equal(a,b) for a,b in zip(model.addresses(tokens),model.addresses(changed)))
    # The recurrent history does respond to earlier observations.
    altered=x.clone();altered[:,:-1]+=10
    assert not torch.equal(first,model(altered,tokens).detach())


def test_capacity_control_parameter_budget():
    config=json.loads((ROOT/'configs/study.json').read_text())
    counts={arm:parameter_count(make_model(arm,8,config,17)) for arm in ['lstm','lstm_capacity','lstm_memory']}
    assert counts['lstm']<counts['lstm_memory']
    assert abs(counts['lstm_capacity']-counts['lstm_memory'])/counts['lstm_memory']<.05


def test_self_financing_and_blocked_orders():
    p,c,t=execute([0,0],[1,0],[True,True],[True,True],.001)
    assert p.sum()+c==pytest.approx(1-.001*t)
    assert c>=0 and p.sum()<1
    p,c,t=execute([1,0],[0,1],[True,True],[False,True],.001)
    assert p.tolist()==[1,0] and c==0 and t==0


def test_positive_rule_and_tie_break():
    weights=weights_from_scores([[.1,.1],[-.1,-.2],[0,0]])
    assert weights.tolist()==[[1,0],[0,0],[0,0]]
    with pytest.raises(ValueError):weights_from_scores([[np.nan,0]])


def metadata():
    return pd.DataFrame({'date':pd.to_datetime(['2023-01-03','2023-01-04']),
                         'entry_date':pd.to_datetime(['2023-01-04','2023-01-05']),
                         'exit_date':pd.to_datetime(['2023-01-05','2023-01-06']),
                         'symbol':['510300']*2,'target':[.1,-.05],'entry_gap':[0,0],'exit_gap':[0,0],
                         'entry_tradable':[True,True],'exit_tradable':[True,True]})


def test_next_open_timing_round_trip_cost_and_cash():
    meta=metadata()
    gross=simulate(meta,[1,1],['510300'],0,'model')
    net=simulate(meta,[1,1],['510300'],5,'model')
    assert np.prod(1+gross.net_return)==pytest.approx(1.1*.95)
    # Only buy at first open and liquidate at final mark; costs do not create cash.
    assert np.prod(1+net.net_return)==pytest.approx(1.1*.95*(1-.0005)/(1+.0005))
    cash=simulate(meta,[0,0],['510300'],5,'cash')
    assert np.all(cash.net_return==0)
    assert (pd.to_datetime(net.entry_date)>meta.date).all()


def test_suspended_open_never_fills():
    meta=metadata();meta['entry_tradable']=[False,True]
    result=simulate(meta,[1,0],['510300'],5,'model')
    assert result.turnover.sum()==0 and result.net_return.sum()==0
    assert result.blocked_buys.sum()==1


def test_negative_r2_and_alpha_are_retained():
    assert forecast_metrics([0,1],[10,10])['r2']<0
    market=np.linspace(-.02,.02,100);r=.7*market-.001
    metrics=statistics(r,market)
    assert metrics['capm_beta']==pytest.approx(.7)
    assert metrics['capm_alpha_annual_arithmetic']==pytest.approx(-.252)
    assert statistics(np.zeros(100),market)['sharpe_rf_zero'] is None


def test_price_limit_regime():
    assert limit_fraction('159915','2020-08-21')==.1
    assert limit_fraction('159915','2020-08-24')==.2
    assert limit_fraction('510300','2024-01-01')==.1


def test_ridge_train_only_closed_form():
    rng=np.random.default_rng(3);x=rng.normal(size=(100,20,8));y=2*x[:,-1,0]+.2
    state=ridge_fit(x,y,.000001)
    assert np.mean((ridge_predict(state,x)-y)**2)<1e-10


def test_future_quotes_cannot_change_past_features_or_execution_time(tmp_path):
    dates=pd.date_range('2018-01-01',periods=100,freq='B')
    rows=[]
    for adjustment in (0,1):
        for symbol in ['510300','510050']:
            for j,date in enumerate(dates):
                close=1+j*.01
                rows.append(dict(date=date,symbol=symbol,adjustment=adjustment,open=close-.002,
                                 close=close,high=close+.01,low=close-.01,volume=1000+j))
    quotes=pd.DataFrame(rows)
    def save_and_build(q):
        q.to_csv(tmp_path/'quotes.csv',index=False)
        (tmp_path/'manifest.json').write_text(json.dumps({'quotes_sha256':sha256(tmp_path/'quotes.csv')}))
        c={'symbols':['510300','510050'],'sequence_length':20}
        frame,_=panel(tmp_path,c)
        x,meta=sequences(frame,c)
        return frame,x,meta
    frame,x,meta=save_and_build(quotes)
    changed=quotes.copy();mask=changed.date>=dates[80]
    for column in ['open','close','high','low','volume']:changed.loc[mask,column]*=3
    future,x2,meta2=save_and_build(changed)
    a=frame[frame.date<dates[80]][FEATURES].to_numpy()
    b=future[future.date<dates[80]][FEATURES].to_numpy()
    np.testing.assert_equal(a,b)
    assert (meta.entry_date>meta.date).all() and (meta.exit_date>meta.entry_date).all()
    first=meta.iloc[0];series=quotes[(quotes.symbol==first.symbol)&(quotes.adjustment==1)].set_index('date')
    assert first.target==pytest.approx(series.loc[first.exit_date,'open']/series.loc[first.entry_date,'open']-1)
    # Heldout prices never alter already available features or prefix sequences.
    valid=(meta.date<dates[80]).to_numpy()
    np.testing.assert_equal(x[valid],x2[(meta2.date<dates[80]).to_numpy()])


def test_incomplete_report_preserves_published_evidence(tmp_path):
    from alphacontextlab.study import report
    outputs=tmp_path/'outputs';outputs.mkdir();reports=tmp_path/'reports';reports.mkdir()
    sentinel=reports/'REPORT.md';sentinel.write_text('published evidence')
    with pytest.raises(ValueError,match='Incomplete study'):
        report(ROOT/'configs/study.json',outputs,reports)
    assert sentinel.read_text()=='published evidence'


def test_new_source_cannot_overwrite_old_output_provenance(tmp_path,monkeypatch):
    from alphacontextlab import study
    outputs=tmp_path/'outputs';outputs.mkdir()
    sentinel=outputs/'provenance.json';sentinel.write_text('{"source_commit":"old"}')
    monkeypatch.setattr(study,'panel',lambda *args:(None,{}))
    monkeypatch.setattr(study,'sequences',lambda *args:(np.zeros((1,20,8)),pd.DataFrame()))
    monkeypatch.setattr(study,'provenance',lambda *args:{'source_commit':'new'})
    with pytest.raises(RuntimeError,match='Existing output provenance differs'):
        study.run(ROOT/'configs/study.json',tmp_path/'data',outputs)
    assert json.loads(sentinel.read_text())=={'source_commit':'old'}
