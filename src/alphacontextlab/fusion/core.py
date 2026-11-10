"""Bitemporal features, explicit entity links and fixed-weight calibrated signals."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd


def timestamp(value):
    t=pd.Timestamp(value)
    if t.tzinfo is None:raise ValueError('Timezone-aware timestamp required')
    return t.tz_convert('UTC')


@dataclass(frozen=True)
class Observation:
    source:str
    entity:str
    feature:str
    event_at:str
    published_at:str
    first_seen_at:str
    value:float
    unit:str
    observation_key:str
    revision:str
    source_ref:str
    quality:str='usable'

    def validate(self):
        for t in [self.event_at,self.published_at,self.first_seen_at]:timestamp(t)
        if not math.isfinite(self.value):raise ValueError('Nonfinite feature; missing is not zero')
        if not all([self.source,self.entity,self.feature,self.unit,self.observation_key,self.revision,self.source_ref]):raise ValueError('Incomplete feature provenance')
        return self
    @property
    def available_at(self):return max(timestamp(self.event_at),timestamp(self.published_at),timestamp(self.first_seen_at))
    @property
    def identifier(self):return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class EntityLink:
    source:str
    entity:str
    symbol:str
    valid_from:str
    first_seen_at:str
    evidence_ref:str

    def usable_at(self,when):
        return bool(self.evidence_ref) and max(timestamp(self.valid_from),timestamp(self.first_seen_at))<=timestamp(when)


class FeatureStore:
    def __init__(self,observations=()):
        self.records={}
        for row in observations:self.add(row)
    def add(self,row):
        row.validate();self.records.setdefault(row.identifier,row)
    def save(self,path):
        p=Path(path)
        if p.exists():raise ValueError('Immutable snapshot already exists')
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(json.dumps(asdict(v),ensure_ascii=False,allow_nan=False)+'\n' for v in self.records.values()))
    @classmethod
    def load(cls,path):return cls(Observation(**json.loads(x)) for x in Path(path).read_text().splitlines() if x)
    def asof(self,symbol,when,links,max_age_days,window_days=None,aggregation='last'):
        t=timestamp(when);valid={(l.source,l.entity) for l in links if l.symbol==symbol and l.usable_at(t)}
        selected={}
        for row in self.records.values():
            if (row.source,row.entity) not in valid or row.quality!='usable' or row.available_at>t:continue
            age=(t-timestamp(row.event_at)).total_seconds()/86400
            if age>max_age_days.get(row.source,0) or (window_days is not None and age>window_days):continue
            selected.setdefault((row.source,row.feature),[]).append(row)
        result={}
        for key,rows in selected.items():
            # A late revision to an older financial period cannot replace a newer period.
            # For a given observation period use only the latest available revision.
            revisions={}
            for row in rows:
                revkey=(row.entity,row.observation_key)
                previous=revisions.get(revkey)
                if previous is None or row.available_at>previous.available_at:revisions[revkey]=row
            values=list(revisions.values());values.sort(key=lambda x:(timestamp(x.event_at),x.available_at,x.identifier))
            chosen=[values[-1]] if aggregation=='last' else values
            units={x.unit for x in chosen}
            if len(units)!=1:raise ValueError('Mixed units for the same feature')
            if aggregation=='last':value=chosen[0].value
            elif aggregation=='mean':value=float(np.mean([x.value for x in chosen]))
            elif aggregation=='sum':value=float(np.sum([x.value for x in chosen]))
            else:raise ValueError('Unknown aggregation')
            result[':'.join(key)]={'value':value,'unit':chosen[0].unit,'available_at':max(x.available_at for x in chosen).isoformat(),
                                    'observation_ids':[x.identifier for x in chosen]}
        return result


@dataclass(frozen=True)
class GroupSignal:
    group:str
    symbol:str
    decision_at:str
    score:float
    produced_at:str
    train_label_end:str
    feature_available_at:str
    model_ref:str
    source_refs:tuple
    sources:tuple=()


class FixedWeightFusion:
    def __init__(self,config):
        self.config=config;self.weights=config['weights'];self.calibration=None
        if set(self.weights)!={'fundamental','price_volume','alternative'} or not np.isclose(sum(self.weights.values()),1):raise ValueError('Three weights must sum to one')
        if any(v<0 for v in self.weights.values()):raise ValueError('Negative group weight')
        if not np.isclose(sum(config['alternative_weights'].values()),1) or any(v<0 for v in config['alternative_weights'].values()):raise ValueError('Alternative weights must be nonnegative and sum to one')
    def fit_calibration(self,oof_scores,calibration_end):
        """Calibration uses earlier OOF scores, not full-history score distributions."""
        cutoff=timestamp(calibration_end)
        required={'decision_at','label_available_at','is_oof',*self.weights}
        if not required<=set(oof_scores):raise ValueError('Incomplete calibration evidence')
        if not oof_scores.is_oof.eq(True).all():raise ValueError('Use out-of-fold calibration scores')
        if not all(timestamp(t)<cutoff for t in oof_scores.decision_at) or not all(timestamp(t)<cutoff for t in oof_scores.label_available_at):raise ValueError('Calibration overlaps decision boundary')
        if len(oof_scores)<20:raise ValueError('Insufficient calibration rows')
        calibration={}
        for group in self.weights:
            x=oof_scores[group].to_numpy(float)
            if not np.isfinite(x).all() or np.std(x,ddof=1)<1e-12:raise ValueError('Invalid or constant group calibration')
            calibration[group]={'mean':float(x.mean()),'std':float(x.std(ddof=1))}
        self.calibration=calibration;self.calibration_end=cutoff;return self
    def fuse(self,symbol,when,signals):
        t=timestamp(when);problems=[];by_group={}
        for s in signals:
            if s.group not in self.weights:raise ValueError('Unknown group signal')
            if s.group in by_group:raise ValueError('Duplicate group signal')
            by_group[s.group]=s
        if self.calibration is None:problems.append('calibration_not_fitted')
        elif self.calibration_end>=t:problems.append('calibration_not_before_decision')
        for group in self.weights:
            s=by_group.get(group)
            if s is None:problems.append('missing_group:'+group);continue
            if s.symbol!=symbol or timestamp(s.decision_at)!=t:problems.append('signal_context_mismatch:'+group)
            if not np.isfinite(s.score):problems.append('nonfinite_signal:'+group)
            if timestamp(s.produced_at)>t or timestamp(s.feature_available_at)>t:problems.append('future_information:'+group)
            if timestamp(s.train_label_end)>=min(t,timestamp(s.produced_at)):problems.append('training_label_overlap:'+group)
            if timestamp(s.feature_available_at)>timestamp(s.produced_at):problems.append('signal_precedes_features:'+group)
            if not s.model_ref or not s.source_refs:problems.append('missing_provenance:'+group)
            if group=='alternative':
                missing=set(self.config['alternative_weights'])-set(s.sources)
                problems.extend('missing_alternative_source:'+x for x in sorted(missing))
        contributions={};normalized={}
        if not problems:
            for group,weight in self.weights.items():
                c=self.calibration[group];z=float(np.clip((by_group[group].score-c['mean'])/c['std'],-self.config['score_clip'],self.config['score_clip']))
                normalized[group]=z;contributions[group]=weight*z
        return {'symbol':symbol,'decision_at':t.isoformat(),'status':'ready' if not problems else 'blocked',
                'score':float(sum(contributions.values())) if not problems else None,'weights':dict(self.weights),
                'normalized_scores':normalized,'contributions':contributions,'reasons':problems,
                'no_missing_weight_redistribution':True,'paper_admitted':False}


def alternative_features(values,config):
    """Fixed subsource combination after caller's training-only normalization."""
    missing=set(config['alternative_weights'])-set(values)
    if missing:return {'status':'blocked','score':None,'missing':sorted(missing)}
    if not all(np.isfinite(values[s]) for s in config['alternative_weights']):raise ValueError('Nonfinite alternative score')
    parts={s:float(w*values[s]) for s,w in config['alternative_weights'].items()}
    return {'status':'ready','score':sum(parts.values()),'contributions':parts}
