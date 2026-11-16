from copy import deepcopy
import pytest
from alphacontextlab.prep_v10.contracts import Forecast,validate_allocation,combine_market_and_alpha,fact_available,membership_available
from alphacontextlab.prep_v10.contracts import staggered_clock,net_external_share_intents


NOW='2026-11-16T12:00:00+08:00'


def test_excess_is_not_an_absolute_cash_allocation_forecast():
    f=Forecast(.01,'excess_return',20,NOW,'csi300')
    with pytest.raises(ValueError,match='absolute return'):validate_allocation(f,'absolute_long_cash',NOW,20)
    validate_allocation(f,'active_against_benchmark',NOW,20,'csi300',.8)
    with pytest.raises(ValueError,match='budget'):validate_allocation(f,'active_against_benchmark',NOW,20,'csi300')


def test_market_direction_is_an_explicit_component():
    a=Forecast(.01,'excess_return',20,NOW,'csi300');m=Forecast(-.03,'absolute_return',20,NOW,'csi300')
    total=combine_market_and_alpha(a,m,1.)
    assert total.value==pytest.approx(-.02)
    validate_allocation(total,'absolute_long_cash',NOW,20)


def test_future_forecasts_and_wrong_horizons_are_rejected():
    f=Forecast(.01,'absolute_return',20,'2026-11-17T12:00:00+08:00')
    with pytest.raises(ValueError,match='Future'):validate_allocation(f,'absolute_long_cash',NOW,20)
    f=Forecast(.01,'absolute_return',5,NOW)
    with pytest.raises(ValueError,match='horizon'):validate_allocation(f,'absolute_long_cash',NOW,20)
    with pytest.raises(ValueError,match='Timezone'):validate_allocation(Forecast(.1,'absolute_return',20,'2026-11-16'),'absolute_long_cash',NOW,20)


def fact():
    return {'approved_for_feature_store':True,'document_value_verified':True,'document_sha256':'a'*64,
        'first_seen_at':'2026-11-16T12:00:00+08:00','publication_verified':True,'original_document_verified':True,
        'candidate_available_at':'2021-02-03T00:00:00+08:00','publication_date_claimed':'2021-02-02T00:00:00+08:00','publication_precision':'date'}


def test_historical_reconstruction_cannot_rewrite_first_seen():
    f=fact();old=deepcopy(f)
    assert fact_available(f,'2021-02-03T12:00:00+08:00','historical_reconstruction')
    assert not fact_available(f,'2021-02-03T12:00:00+08:00','strict_first_seen')
    assert f==old and fact_available(f,NOW,'strict_first_seen')


def test_unverified_publication_or_unapproved_pilot_cannot_enter_history():
    f=fact();f['publication_verified']=False
    assert not fact_available(f,NOW,'historical_reconstruction')
    f=fact();f['approved_for_feature_store']=False
    assert not fact_available(f,NOW,'strict_first_seen')
    f=fact();f['candidate_available_at']='2021-02-02T10:00:00+08:00'
    assert not fact_available(f,NOW,'historical_reconstruction')


def test_membership_requires_known_announcement_and_effective_interval():
    r={'source_verified':True,'announcement_at':'2021-01-05T16:00:00+08:00','effective_start':'2021-01-01T00:00:00+08:00','effective_end':'2022-01-01T00:00:00+08:00'}
    assert not membership_available(r,'2021-01-04T16:00:00+08:00')
    assert membership_available(r,'2021-01-06T16:00:00+08:00')
    assert not membership_available(r,'2022-01-03T16:00:00+08:00')


def test_staggered_clock_uses_every_session_without_selecting_a_winning_offset():
    schedule=staggered_clock(20,5)
    assert len(schedule)==20 and set(x['sleeve'] for x in schedule)==set(range(5))
    for sleeve in range(5):
        sessions=[x['session'] for x in schedule if x['sleeve']==sleeve]
        assert all(b-a==5 for a,b in zip(sessions,sessions[1:]))


def test_internal_sleeve_transfers_do_not_create_external_turnover():
    trades=net_external_share_intents([{'000001.SZ':100,'600000.SH':50},{'000001.SZ':-100,'600000.SH':-20}])
    assert trades=={'600000.SH':30}
