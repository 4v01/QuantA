"""Common split, validation-only ridge search and calibrated fixed-weight fusion."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from ..v4.data import FEATURES
from ..v4.metrics import daily_ic
from .data import FINANCIAL,NEWS_FEATURES

GROUPS={'fundamental':list(FINANCIAL),'price_volume':FEATURES,'alternative':NEWS_FEATURES}


def percentile(frame,prediction):
    p=pd.Series(prediction,index=frame.index)
    return (p.groupby(frame.date).rank(pct=True,method='average')-.5).to_numpy()


def estimator(alpha):
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))


def fit(train,validation,refit,test,config):
    predictions={};validation_predictions={};models={};selections={};calibration={}
    for group,features in GROUPS.items():
        tx=train[features].to_numpy();vx=validation[features].to_numpy();rx=refit[features].to_numpy();qx=test[features].to_numpy()
        candidates=[]
        for alpha in config['ridge_alphas']:
            model=estimator(alpha).fit(tx,train.target.to_numpy());p=model.predict(vx)
            raw=float(daily_ic(validation,p,config['min_cross_section']).rank_ic.mean())
            candidates.append({'alpha':alpha,'raw_validation_ic':raw if np.isfinite(raw) else None,
                                'selection_score':abs(raw) if np.isfinite(raw) else -1.,'direction':1 if not np.isfinite(raw) or raw>=0 else -1})
        chosen=max(candidates,key=lambda r:r['selection_score'])
        if chosen['selection_score']<0:raise ValueError('Group has no valid calibration: '+group)
        vm=estimator(chosen['alpha']).fit(tx,train.target.to_numpy())
        vp=percentile(validation,vm.predict(vx)*chosen['direction']);mean=float(vp.mean());std=float(vp.std(ddof=1))
        if std<1e-10:raise ValueError('Constant group validation score')
        model=estimator(chosen['alpha']).fit(rx,refit.target.to_numpy())
        qp=percentile(test,model.predict(qx)*chosen['direction'])
        predictions[group]=np.clip((qp-mean)/std,-config['score_clip'],config['score_clip'])
        validation_predictions[group]=np.clip((vp-mean)/std,-config['score_clip'],config['score_clip'])
        models[group]={'model':model,'features':features,'direction':chosen['direction'],'calibration':{'mean':mean,'std':std}}
        selections[group]={'candidates':candidates,'selected':chosen,'feature_names':features};calibration[group]={'mean':mean,'std':std}
    predictions['fusion']=sum(config['weights'][g]*predictions[g] for g in config['weights'])
    validation_predictions['fusion']=sum(config['weights'][g]*validation_predictions[g] for g in config['weights'])
    return predictions,validation_predictions,models,{'groups':selections,'calibration':calibration,'weights':config['weights']}
