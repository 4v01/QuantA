"""Signed temporal model selection, explicit reliability shrinkage and ranking."""
import numpy as np
from lightgbm import LGBMRanker,early_stopping,log_evaluation
from ..v6.models import estimator,percentile
from ..v6.data import FINANCIAL,NEWS_FEATURES
from ..v4.data import FEATURES
from ..v4.metrics import daily_ic
from .data import ALIGNED,CONTEXT


def validation_diagnostics(frame,pred,config):
    d=daily_ic(frame,pred,config['min_cross_section']);x=d.rank_ic.dropna().to_numpy()
    if len(x)<20:return {'mean':None,'blocks':[],'stable_score':-1.,'reliability':0.}
    blocks=[float(z.mean()) for z in np.array_split(x,config['validation_blocks']) if len(z)]
    mean=float(x.mean());disp=float(np.std(blocks,ddof=1)) if len(blocks)>1 else 0.
    reliability=max(mean,0)/(max(mean,0)+disp+config['reliability_floor'])*float(np.mean(np.array(blocks)>0))
    return {'mean':mean,'blocks':blocks,'stable_score':mean-config['instability_penalty']*disp,'reliability':float(reliability)}


def group_spec(method,group):
    advanced=method in ['aligned_fund','kaggle_context','kaggle_rank']
    if group=='fundamental':return (ALIGNED if advanced else list(FINANCIAL)),('target20' if advanced else 'target')
    if group=='price_volume':return FEATURES+(CONTEXT if method in ['kaggle_context','kaggle_rank'] else []),'target'
    return NEWS_FEATURES,'target'


def ranker(config,rounds):
    c=config['ranker']
    return LGBMRanker(objective='lambdarank',n_estimators=int(rounds),num_leaves=c['num_leaves'],max_depth=c['max_depth'],
      min_child_samples=c['min_child_samples'],reg_lambda=c['reg_lambda'],learning_rate=c['learning_rate'],verbosity=-1,
      deterministic=True,force_col_wise=True,n_jobs=2,random_state=config['seed'])


def relevance(frame):return np.clip(np.floor((frame.target.to_numpy()+.5)*5),0,4).astype(int)


def fit_method(train,validation,refit,test,method,config):
    models={};selection={};scores={};v_scores={}
    for group in config['weights']:
        features,target=group_spec(method,group);x=train[features].to_numpy();v=validation[features].to_numpy();r=refit[features].to_numpy();q=test[features].to_numpy()
        candidates=[]
        if method=='kaggle_rank' and group=='price_volume':
            vm=ranker(config,config['ranker']['max_rounds'])
            vm.fit(x,relevance(train),group=train.groupby('date',sort=False).size().to_numpy(),
              eval_set=[(v,relevance(validation))],eval_group=[validation.groupby('date',sort=False).size().to_numpy()],eval_at=[10],
              callbacks=[early_stopping(config['ranker']['early_stopping'],verbose=False),log_evaluation(0)])
            raw=vm.predict(v);d=validation_diagnostics(validation,raw,config)
            chosen={'kind':'lambdarank','best_iteration':int(vm.best_iteration_),'direction':1,'validation':d,'selection_score':d['stable_score']}
            candidates=[chosen];m=ranker(config,vm.best_iteration_).fit(r,relevance(refit),group=refit.groupby('date',sort=False).size().to_numpy())
        else:
            for alpha in config['ridge_alphas']:
                model=estimator(alpha).fit(x,train[target].to_numpy());raw=model.predict(v);d=validation_diagnostics(validation,raw,config)
                if method=='legacy_matched':score=abs(d['mean']) if d['mean'] is not None else -1.;direction=1 if d['mean'] is None or d['mean']>=0 else -1
                else:score=d['stable_score'];direction=1
                candidates.append({'kind':'ridge','alpha':alpha,'direction':direction,'validation':d,'selection_score':score})
            chosen=max(candidates,key=lambda z:z['selection_score'])
            vm=estimator(chosen['alpha']).fit(x,train[target].to_numpy());raw=vm.predict(v)
            m=estimator(chosen['alpha']).fit(r,refit[target].to_numpy())
        direction=chosen['direction'];vp=percentile(validation,raw*direction);mean=float(vp.mean());std=float(vp.std(ddof=1))
        constant=std<1e-12 or chosen['validation']['mean'] is None
        if std<1e-12:std=1.
        confidence=1. if method in ['legacy_matched','signed_stable'] else chosen['validation']['reliability']
        if constant:confidence=0.
        qp=percentile(test,m.predict(q)*direction)
        scores[group]=np.clip((qp-mean)/std,-config['score_clip'],config['score_clip'])*confidence
        v_scores[group]=np.clip((vp-mean)/std,-config['score_clip'],config['score_clip'])*confidence
        models[group]={'model':m,'features':features,'target':target,'direction':direction,'calibration':{'mean':mean,'std':std},'confidence':confidence}
        selection[group]={'candidates':candidates,'selected':chosen,'confidence':confidence,'features':features,'target':target,'constant_validation':constant}
    scores['fusion']=sum(config['weights'][g]*scores[g] for g in config['weights']);v_scores['fusion']=sum(config['weights'][g]*v_scores[g] for g in config['weights'])
    return scores,v_scores,models,selection
