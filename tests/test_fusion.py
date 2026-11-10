import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from alphacontextlab.fusion.core import Observation,EntityLink,FeatureStore,GroupSignal,FixedWeightFusion,alternative_features
from alphacontextlab.fusion.training import FeatureVector,build_vector,LinearGroupModel
from alphacontextlab.fusion.adapters import import_satellite_features,import_x_export


def config():return json.loads((Path(__file__).parents[1]/'configs/fusion.json').read_text())

def observation(**kwargs):
    fields=dict(source='fundamental',entity='600000',feature='roe',event_at='2020-03-31T15:59:59Z',published_at='2020-04-30T15:59:59Z',
                first_seen_at='2020-05-01T00:00:00Z',value=10.,unit='percent',observation_key='2020Q1',revision='a',source_ref='sha256:source')
    fields.update(kwargs);return Observation(**fields)

def links(source='fundamental',entity='600000'):
    return [EntityLink(source,entity,'600000','2019-01-01T00:00:00Z','2019-01-01T00:00:00Z','verified:site-or-symbol')]

def calibrated():
    c=config();rng=np.random.default_rng(7);d=pd.DataFrame({g:rng.normal(size=40) for g in c['weights']})
    d['decision_at']='2019-01-01T00:00:00Z';d['label_available_at']='2019-02-01T00:00:00Z';d['is_oof']=True
    return FixedWeightFusion(c).fit_calibration(d,'2019-03-01T00:00:00Z')

def signals():
    return [GroupSignal(g,'600000','2020-06-01T07:00:00Z',.5,'2020-06-01T06:59:00Z','2020-05-01T00:00:00Z',
          '2020-06-01T06:58:00Z','sha256:model',('sha256:data',),tuple(config()['alternative_weights']) if g=='alternative' else (g,)) for g in config()['weights']]


def test_publication_first_seen_and_future_restatement():
    s=FeatureStore([observation(),observation(value=99.,revision='b',first_seen_at='2021-01-01T00:00:00Z')])
    assert s.asof('600000','2020-04-30T23:00:00Z',links(),{'fundamental':550})=={}
    a=s.asof('600000','2020-06-01T00:00:00Z',links(),{'fundamental':550});assert a['fundamental:roe']['value']==10.


def test_old_period_revision_does_not_replace_new_period():
    s=FeatureStore([observation(),observation(event_at='2020-06-30T15:59:59Z',observation_key='2020Q2',value=20.,first_seen_at='2020-08-01T00:00:00Z'),
                    observation(first_seen_at='2020-09-01T00:00:00Z',value=999.,revision='b')])
    a=s.asof('600000','2020-10-01T00:00:00Z',links(),{'fundamental':550});assert a['fundamental:roe']['value']==20.


def test_explicit_mapping_no_global_market_broadcast():
    s=FeatureStore([observation(source='polymarket',entity='market123')]);when='2020-06-01T00:00:00Z'
    assert s.asof('600000',when,[],{'polymarket':100})=={}
    assert s.asof('600000',when,links('polymarket','market123'),{'polymarket':100})
    late=[replace(links('polymarket','market123')[0],first_seen_at='2021-01-01T00:00:00Z')]
    assert s.asof('600000',when,late,{'polymarket':100})=={}


def test_news_same_publication_second_is_not_accidentally_deduplicated():
    s=FeatureStore([observation(source='news',feature='sentiment',observation_key='article1',value=1.),
                    observation(source='news',feature='sentiment',observation_key='article2',value=3.)])
    rows=s.asof('600000','2020-05-02T00:00:00Z',links('news'),{'news':100},aggregation='mean')
    assert rows['news:sentiment']['value']==2.


def test_missing_alternative_does_not_renormalize_70percent():
    m=calibrated();r=m.fuse('600000','2020-06-01T07:00:00Z',signals()[:2])
    assert r['status']=='blocked' and r['score'] is None and r['weights']==config()['weights']
    assert r['contributions']=={}


def test_exact_30_40_30_contributions_and_source_coverage():
    m=calibrated();r=m.fuse('600000','2020-06-01T07:00:00Z',signals());assert r['status']=='ready'
    assert r['score']==pytest.approx(sum(r['contributions'].values()))
    for g,w in config()['weights'].items():assert r['contributions'][g]==pytest.approx(w*r['normalized_scores'][g])
    broken=signals();broken[-1]=replace(broken[-1],sources=('news','polymarket'))
    assert m.fuse('600000','2020-06-01T07:00:00Z',broken)['status']=='blocked'


def test_no_future_information_or_producer_before_features():
    m=calibrated();bad=signals();bad[0]=replace(bad[0],feature_available_at='2020-06-02T00:00:00Z')
    assert m.fuse('600000','2020-06-01T07:00:00Z',bad)['status']=='blocked'
    bad=signals();bad[0]=replace(bad[0],train_label_end='2020-06-01T06:59:30Z')
    assert m.fuse('600000','2020-06-01T07:00:00Z',bad)['status']=='blocked'


def test_no_partial_alternative_score():
    assert alternative_features({'news':.2},config())['score'] is None
    r=alternative_features(dict.fromkeys(config()['alternative_weights'],.4),config());assert r['score']==pytest.approx(.4)


def test_store_immutable_and_bad_units_fail(tmp_path):
    p=tmp_path/'features.jsonl';s=FeatureStore([observation()]);s.save(p)
    assert FeatureStore.load(p).records.keys()==s.records.keys()
    with pytest.raises(ValueError):s.save(p)
    s.add(observation(observation_key='other',unit='CNY',value=4.))
    with pytest.raises(ValueError):s.asof('600000','2020-06-01T00:00:00Z',links(),{'fundamental':550},aggregation='mean')


def test_timezone_and_staleness_are_enforced():
    with pytest.raises(ValueError):observation(first_seen_at='2020-05-01').validate()
    s=FeatureStore([observation()]);assert s.asof('600000','2022-01-01T00:00:00Z',links(),{'fundamental':550})=={}


def test_pixels_required_catalogue_quality_not_activity(tmp_path):
    p=tmp_path/'sat.csv';p.write_text('entity,feature,value,unit,observed_at,processed_at,scene_id,site_mapping_ref\nsite,cloud_cover,1,pct,2020-01-01T00:00:00Z,2020-01-02T00:00:00Z,scene,map\n')
    with pytest.raises(ValueError):import_satellite_features(p,'2020-01-03T00:00:00Z')


def test_x_import_requires_verified_entity_link(tmp_path):
    p=tmp_path/'x.json';p.write_text(json.dumps([{'id':'1','created_at':'2020-01-01T00:00:00Z','sentiment':.2}]))
    with pytest.raises(ValueError):import_x_export(p,'2020-01-02T00:00:00Z')


def test_group_training_and_signal_bridge_no_future_labels():
    vectors=[FeatureVector('600000','2019-01-01T00:00:00Z',{'roe':float(i)},'2018-12-31T00:00:00Z',('data',),('fundamental',)) for i in range(30)]
    labels=np.arange(30)*.01;ends=['2019-01-10T00:00:00Z']*30
    m=LinearGroupModel('fundamental',['roe']).fit(vectors,labels,ends,'2019-02-01T00:00:00Z')
    v=FeatureVector('600000','2020-06-01T07:00:00Z',{'roe':5.},'2020-05-01T00:00:00Z',('newdata',),('fundamental',))
    signal=m.predict(v,'2020-06-01T06:59:00Z');assert signal.group=='fundamental' and signal.model_ref.startswith('sha256:')
    with pytest.raises(ValueError):LinearGroupModel('fundamental',['roe']).fit(vectors,labels,['2020-01-01T00:00:00Z']*30,'2019-02-01T00:00:00Z')
    with pytest.raises(ValueError):m.predict(v,'2020-06-02T00:00:00Z')


def test_vector_assembly_missing_and_available_fields():
    s=FeatureStore([observation()]);spec={'roe':('fundamental','roe','last',None)}
    v,status=build_vector(s,'600000','2020-06-01T00:00:00Z',links(),spec,{'fundamental':550});assert v.values['roe']==10 and status['status']=='ready'
    v,status=build_vector(s,'600000','2020-01-01T00:00:00Z',links(),spec,{'fundamental':550});assert v is None
