"""Small linear group learner and availability-checked bridge to fusion signals."""
from dataclasses import dataclass
import hashlib,json
from importlib.metadata import version
import numpy as np
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.preprocessing import StandardScaler
from .core import GroupSignal,timestamp


@dataclass(frozen=True)
class FeatureVector:
    symbol:str
    decision_at:str
    values:dict
    available_at:str
    source_refs:tuple
    sources:tuple


def build_vector(store,symbol,when,links,specs,max_age_days):
    """specs maps model feature name -> (source, feature, aggregation, window_days)."""
    values={};refs=[];availability=[];sources=[];missing=[]
    for name,(source,feature,aggregation,window) in specs.items():
        rows=store.asof(symbol,when,links,max_age_days,window_days=window,aggregation=aggregation)
        item=rows.get(source+':'+feature)
        if item is None:missing.append(name);continue
        values[name]=item['value'];refs.extend(item['observation_ids']);availability.append(item['available_at']);sources.append(source)
    if missing:return None,{'status':'blocked','missing_features':missing}
    vector=FeatureVector(symbol,when,values,max(availability,key=timestamp),tuple(sorted(set(refs))),tuple(sorted(set(sources))))
    return vector,{'status':'ready'}


class LinearGroupModel:
    """Fundamental/alternative baseline. Existing V4 price models can also emit GroupSignal."""
    def __init__(self,group,features,kind='ridge',alpha=100.,seed=17):
        if group not in ['fundamental','price_volume','alternative']:raise ValueError('Invalid group')
        self.group=group;self.features=list(features);self.kind=kind;self.alpha=alpha;self.seed=seed
    def fit(self,vectors,labels,label_available_at,fit_at):
        if len(vectors)<20 or len(vectors)!=len(labels) or len(vectors)!=len(label_available_at):raise ValueError('Insufficient/misaligned group training evidence')
        cutoff=timestamp(fit_at)
        for vector,label_end in zip(vectors,label_available_at):
            if timestamp(vector.available_at)>timestamp(vector.decision_at):raise ValueError('Future feature in group training')
            if timestamp(label_end)>=cutoff:raise ValueError('Unobserved training label')
            if timestamp(vector.decision_at)>=cutoff:raise ValueError('Training crosses fit boundary')
            if not vector.source_refs:raise ValueError('Missing training feature provenance')
        x=np.array([[v.values[n] for n in self.features] for v in vectors],float);y=np.asarray(labels,float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('No silent missing-feature imputation')
        self.scaler=StandardScaler().fit(x)
        if self.kind=='ridge':self.model=Ridge(alpha=self.alpha)
        elif self.kind=='elasticnet':self.model=ElasticNet(alpha=self.alpha,l1_ratio=.5,max_iter=5000,random_state=self.seed)
        else:raise ValueError('Use declared ridge/elasticnet baseline or a separately frozen V4 producer')
        self.model.fit(self.scaler.transform(x),y);self.fit_at=fit_at;self.train_label_end=max(label_available_at,key=timestamp)
        state={'group':self.group,'features':self.features,'kind':self.kind,'alpha':self.alpha,'seed':self.seed,'fit_at':fit_at,
               'train_label_end':self.train_label_end,'coef':self.model.coef_.tolist(),'intercept':float(self.model.intercept_),
               'scale_mean':self.scaler.mean_.tolist(),'scale_std':self.scaler.scale_.tolist(),
               'training_features_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
               'training_labels_sha256':hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
               'sklearn_version':version('scikit-learn'),
               'training_observations':len(vectors),'training_source_refs':sorted(set(r for v in vectors for r in v.source_refs))}
        self.state=state;self.model_ref='sha256:'+hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest();return self
    def predict(self,vector,produced_at):
        if timestamp(self.fit_at)>timestamp(produced_at):raise ValueError('Model did not yet exist')
        if max(timestamp(vector.available_at),timestamp(produced_at))>timestamp(vector.decision_at):raise ValueError('Future feature or prediction')
        if timestamp(vector.available_at)>timestamp(produced_at):raise ValueError('Features not available at prediction production')
        x=np.array([[vector.values[n] for n in self.features]],float)
        score=float(self.model.predict(self.scaler.transform(x))[0])
        return GroupSignal(self.group,vector.symbol,vector.decision_at,score,produced_at,self.train_label_end,vector.available_at,
                           self.model_ref,vector.source_refs,vector.sources)
