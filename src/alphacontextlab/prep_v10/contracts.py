"""Explicit economic and historical-availability contracts for the next study."""
from dataclasses import dataclass
from datetime import datetime,timedelta
import math


def timestamp(value):
    d=datetime.fromisoformat(value.replace('Z','+00:00')) if isinstance(value,str) else value
    if d.tzinfo is None:raise ValueError('Timezone-aware timestamp required')
    return d


@dataclass(frozen=True)
class Forecast:
    value: float
    basis: str
    horizon_sessions: int
    as_of: str
    benchmark_id: str | None = None


def validate_allocation(forecast,mode,decision_at,horizon_sessions,benchmark_id=None,benchmark_budget=None):
    if horizon_sessions<=0:raise ValueError('Positive horizon required')
    if not math.isfinite(forecast.value):raise ValueError('Unknown return is not zero alpha')
    if timestamp(forecast.as_of)>timestamp(decision_at):raise ValueError('Future forecast')
    if forecast.horizon_sessions!=horizon_sessions:raise ValueError('Forecast/allocation horizon mismatch')
    if mode=='absolute_long_cash':
        if forecast.basis!='absolute_return':raise ValueError('Long/cash allocation requires absolute return, not rank or excess alone')
    elif mode=='active_against_benchmark':
        if forecast.basis!='excess_return' or not benchmark_id or forecast.benchmark_id!=benchmark_id:
            raise ValueError('Excess return must name the same benchmark')
        if benchmark_budget is None or not 0<=benchmark_budget<=1:
            raise ValueError('Benchmark exposure budget must be explicit')
    else:raise ValueError('Unknown allocation mandate')


def combine_market_and_alpha(alpha,market,beta):
    if alpha.basis!='excess_return' or market.basis!='absolute_return':raise ValueError('Incorrect component bases')
    if not alpha.benchmark_id or alpha.benchmark_id!=market.benchmark_id:raise ValueError('Different underlying benchmarks')
    if alpha.horizon_sessions!=market.horizon_sessions:raise ValueError('Different component horizons')
    if timestamp(alpha.as_of)!=timestamp(market.as_of):raise ValueError('Components must share a decision snapshot')
    if not all(math.isfinite(v) for v in [alpha.value,market.value,beta]):raise ValueError('Unknown component')
    return Forecast(alpha.value+beta*market.value,'absolute_return',alpha.horizon_sessions,alpha.as_of,alpha.benchmark_id)


def fact_available(fact,decision_at,mode):
    if not fact.get('approved_for_feature_store') or not fact.get('document_value_verified'):return False
    if len(fact.get('document_sha256',''))!=64:return False
    at=timestamp(decision_at);first_seen=timestamp(fact['first_seen_at'])
    if mode=='strict_first_seen':return at>=first_seen
    if mode!='historical_reconstruction':raise ValueError('Unknown availability mode')
    if not fact.get('publication_verified') or not fact.get('original_document_verified'):return False
    available=timestamp(fact['candidate_available_at']);published=timestamp(fact['publication_date_claimed'])
    # Date-only evidence is conservatively available on the next calendar day.
    if fact.get('publication_precision')=='date' and available<published+timedelta(days=1):return False
    return at>=available


def membership_available(record,decision_at):
    at=timestamp(decision_at)
    return bool(record.get('source_verified')) and timestamp(record['announcement_at'])<=at and timestamp(record['effective_start'])<=at<timestamp(record['effective_end'])


def staggered_clock(session_count,horizon=5):
    if horizon<=0 or session_count<0:raise ValueError('Invalid session count or horizon')
    return [{'session':j,'sleeve':j%horizon} for j in range(session_count)]


def net_external_share_intents(sleeve_changes):
    # Accounting preparation only: virtual sleeve transfers are not exchange
    # trades. Real cash/lot/T+1/fill constraints remain a separate readiness gate.
    totals={}
    for changes in sleeve_changes:
        for security,shares in changes.items():
            if not math.isfinite(shares):raise ValueError('Unknown share intent')
            totals[security]=totals.get(security,0.)+shares
    return {security:shares for security,shares in totals.items() if abs(shares)>1e-12}
