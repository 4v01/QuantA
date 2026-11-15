import numpy as np
import pandas as pd
from ..v4.data import FEATURES
from ..v4.metrics import daily_ic
from ..v6.data import FINANCIAL,NEWS_FEATURES
from ..v6.models import estimator,percentile
from ..v7.models import validation_diagnostics
from ..v7.study import split
from .data import FUND,PRICE


def predict(frame,models,base):
    if models.get('kind')=='classic':
        value=frame[['earnings_yield','book_to_price']].mean(axis=1).fillna(0.)
        quality=(frame.roe_annual.fillna(0.)-frame.fin_leverage.fillna(0.))/2
        price=frame[['momentum_12_1','momentum_6_1','low_volatility_60']].mean(axis=1).fillna(0.)
        return {'fundamental':(value+quality).to_numpy()/2,'price_volume':price.to_numpy(),
                'fusion':(.5*(value+quality)/2+.5*price).to_numpy()}
    scores={}
    for group,state in models.items():
        raw=state['model'].predict(frame[state['features']].to_numpy())
        rank=percentile(frame,raw); c=state['calibration']
        scores[group]=np.clip((rank-c['mean'])/c['std'],-base['score_clip'],base['score_clip'])*state['confidence']
    scores['fusion']=sum(base['weights'][g]*scores[g] for g in base['weights'])
    return scores


def fit_fold(panel,year,half,method,base,config):
    a,b,r,q,info=split(panel,year,half,config['train_months'],base)
    if method=='classic_value_quality':
        model={'kind':'classic'}; selection={'classic_weights':{'fundamental':.5,'price_volume':.5},'learned':False}
    else:
        model={}; selection={}; target='target20' if method=='economic_h20' else 'target'
        for group in base['weights']:
            if group=='fundamental':features=FUND if method.startswith('economic') else list(FINANCIAL)
            elif group=='price_volume':features=FEATURES+(PRICE if method.startswith('economic') else [])
            else:features=NEWS_FEATURES
            candidates=[]; predictions=[]
            for alpha in base['ridge_alphas']:
                vm=estimator(alpha).fit(a[features].to_numpy(),a[target].to_numpy()); vp=vm.predict(b[features].to_numpy())
                # Model selection always uses its declared training horizon.
                validation=b.copy()
                if target=='target20':validation['label']=validation.label20
                d=validation_diagnostics(validation,vp,base)
                candidates.append({'alpha':alpha,'diagnostics':d,'score':d['stable_score']}); predictions.append(vp)
            index=max(range(len(candidates)),key=lambda i:candidates[i]['score']); chosen=candidates[index]
            rank=percentile(b,predictions[index]); mean=float(rank.mean()); std=float(rank.std(ddof=1))
            constant=std<1e-12 or chosen['diagnostics']['mean'] is None
            if std<1e-12:std=1.
            confidence=chosen['diagnostics']['reliability'] if method=='matched_control' else 1.
            if constant:confidence=0.
            final=estimator(chosen['alpha']).fit(r[features].to_numpy(),r[target].to_numpy())
            model[group]={'model':final,'features':features,'target':target,'confidence':confidence,'calibration':{'mean':mean,'std':std}}
            selection[group]={'candidates':candidates,'selected':chosen,'confidence':confidence,'features':features,'target':target,'direction':1}
    p=q[['date','symbol','label','target','exit_date','label20','target20','exit20']].copy()
    p['prediction']=predict(q,model,base)['fusion']
    return p,model,{'split':info,'groups':selection,'method':method}


def calibrate(history,begin,method,config):
    start=begin-pd.DateOffset(months=config['calibration_months'])
    h=history[(history.method==method)&(history.date>=start)&(history.date<begin)&(history.exit20<begin)].dropna(subset=['label20','prediction']).copy()
    if h.empty:
        return {'slope':0.,'raw_slope':0.,'rows':0,'first_date':None,'last_date':None,'exit20_max':None,'score_scale':0.,'reason':'no_matured_prequential_history'}
    h['x']=h.prediction-h.groupby('date').prediction.transform('mean')
    h['y']=h.label20-h.groupby('date').label20.transform('mean')
    xx=float(h.x@h.x); xy=float(h.x@h.y); raw=xy/xx if xx>1e-12 else 0.
    return {'slope':max(0.,raw)*config['calibration_slope_shrinkage'],'raw_slope':raw,
            'rows':len(h),'first_date':str(h.date.min()),'last_date':str(h.date.max()),'exit20_max':str(h.exit20.max()),
            'score_scale':float(h.x.abs().mean()),'reason':'matured_earlier_forecasts_only'}


def apply_calibration(pred,calibration):
    return calibration['slope']*(pred.prediction-pred.groupby('date').prediction.transform('mean'))
