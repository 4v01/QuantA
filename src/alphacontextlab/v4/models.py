"""Fixed small search, validation-only orientation, early stopping and symbolic features."""
import numpy as np
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor,early_stopping,log_evaluation
from gplearn.genetic import SymbolicTransformer
from .data import FEATURES
from .metrics import daily_ic


def validation_score(frame,pred,config):
    d=daily_ic(frame,pred,config['min_cross_section']);mean=float(d.rank_ic.mean())
    if not np.isfinite(mean):raise ValueError('Insufficient or constant validation predictions')
    return abs(mean),1 if mean>=0 else -1,mean


def lgb(config,leaves,rounds):
    return LGBMRegressor(objective='regression',n_estimators=int(rounds),num_leaves=leaves,learning_rate=.03,
                        min_child_samples=150,colsample_bytree=.8,subsample=1.,reg_lambda=10.,verbosity=-1,
                        deterministic=True,force_col_wise=True,n_jobs=2,random_state=config['seed'])


def x_rerank(frame,pred):
    # Independent small adaptation of retrieval -> ranking -> heuristic filtering.
    # The coefficients are OUR frozen assumptions, not X's financial algorithm.
    import pandas as pd
    d=frame[['date','vol20','amount_ratio20']].copy();d['prediction']=pred
    r=d.groupby('date').prediction.rank(pct=True)-.5
    return np.round((r-.20*d.vol20+.10*d.amount_ratio20).to_numpy(),12)


def fit_fold(train,validation,refit,test,config):
    x=train[FEATURES].to_numpy();v=validation[FEATURES].to_numpy();r=refit[FEATURES].to_numpy();q=test[FEATURES].to_numpy()
    y=train.target.to_numpy();vy=validation.target.to_numpy();ry=refit.target.to_numpy()
    scaler=StandardScaler().fit(x);xs=scaler.transform(x);vs=scaler.transform(v)
    refit_scaler=StandardScaler().fit(r);rs=refit_scaler.transform(r);qs=refit_scaler.transform(q)
    records={};models={};predictions={};validation_predictions={}
    for arm in ['ridge','elasticnet']:
        candidates=[]
        for alpha in config['ridge_alphas' if arm=='ridge' else 'elasticnet_alphas']:
            m=Ridge(alpha=alpha) if arm=='ridge' else ElasticNet(alpha=alpha,l1_ratio=config['elasticnet_l1_ratio'],max_iter=5000,tol=1e-6,random_state=config['seed'])
            m.fit(xs,y);p=m.predict(vs);score,sign,raw=validation_score(validation,p,config)
            candidates.append({'alpha':alpha,'absolute_validation_rank_ic':score,'direction':sign,'raw_validation_rank_ic':raw})
        best=max(candidates,key=lambda z:z['absolute_validation_rank_ic']);alpha=best['alpha']
        vm=Ridge(alpha=alpha) if arm=='ridge' else ElasticNet(alpha=alpha,l1_ratio=.5,max_iter=5000,tol=1e-6,random_state=config['seed'])
        vm.fit(xs,y);validation_predictions[arm]=vm.predict(vs)*best['direction']
        m=Ridge(alpha=alpha) if arm=='ridge' else ElasticNet(alpha=alpha,l1_ratio=.5,max_iter=5000,tol=1e-6,random_state=config['seed'])
        m.fit(rs,ry);predictions[arm]=m.predict(qs)*best['direction'];models[arm]={'scaler':refit_scaler,'model':m,'direction':best['direction']}
        records[arm]={'candidates':candidates,'selected':best}
    # LightGBM uses validation for both stopping and the small, declared leaf/sign search.
    candidates=[]
    for leaves in config['lightgbm_leaves']:
        m=lgb(config,leaves,config['lightgbm_max_rounds'])
        m.fit(x,y,eval_set=[(v,vy)],callbacks=[early_stopping(config['lightgbm_early_stopping'],verbose=False),log_evaluation(0)])
        p=m.predict(v);score,sign,raw=validation_score(validation,p,config)
        candidates.append({'leaves':leaves,'best_iteration':m.best_iteration_,'absolute_validation_rank_ic':score,
                           'direction':sign,'raw_validation_rank_ic':raw,'validation_prediction':p})
    best=max(candidates,key=lambda z:z['absolute_validation_rank_ic']);m=lgb(config,best['leaves'],best['best_iteration']);m.fit(r,ry)
    predictions['lightgbm']=m.predict(q)*best['direction'];validation_predictions['lightgbm']=best['validation_prediction']*best['direction']
    models['lightgbm']={'model':m,'direction':best['direction']}
    records['lightgbm']={'candidates':[{k:v for k,v in z.items() if k!='validation_prediction'} for z in candidates],
                         'selected':{k:v for k,v in best.items() if k!='validation_prediction'}}
    # GP learns on training only. The same frozen formulas are applied to validation,
    # refit and test; transformations use only each fit's feature-side statistics.
    rng=np.random.default_rng(config['seed']);ids=np.sort(rng.choice(len(xs),min(len(xs),config['gp_training_rows']),replace=False))
    gp=SymbolicTransformer(population_size=config['gp_population'],hall_of_fame=20,n_components=config['gp_components'],
        generations=config['gp_generations'],tournament_size=10,init_depth=(1,3),function_set=('add','sub','mul','div','abs','sqrt'),
        metric='spearman',parsimony_coefficient=.01,max_samples=.8,n_jobs=1,random_state=config['seed'],feature_names=FEATURES)
    gp.fit(xs[ids],y[ids])
    # Clip only GP outputs; bounded numeric guard is fixed, never selected on test.
    expand=lambda z:np.column_stack([z,np.clip(gp.transform(z),-100,100)])
    gx=expand(xs);gv=expand(vs);gscale=StandardScaler().fit(gx);candidates=[]
    for alpha in config['ridge_alphas']:
        m=Ridge(alpha=alpha).fit(gscale.transform(gx),y);p=m.predict(gscale.transform(gv));score,sign,raw=validation_score(validation,p,config)
        candidates.append({'alpha':alpha,'absolute_validation_rank_ic':score,'direction':sign,'raw_validation_rank_ic':raw})
    best=max(candidates,key=lambda z:z['absolute_validation_rank_ic']);vm=Ridge(alpha=best['alpha']).fit(gscale.transform(gx),y)
    validation_predictions['gp_ridge']=vm.predict(gscale.transform(gv))*best['direction']
    # Keep training input scaler, so symbolic functions retain their exact meaning.
    gr=expand(scaler.transform(r));gq=expand(scaler.transform(q));final_scale=StandardScaler().fit(gr)
    m=Ridge(alpha=best['alpha']).fit(final_scale.transform(gr),ry)
    predictions['gp_ridge']=m.predict(final_scale.transform(gq))*best['direction']
    models['gp_ridge']={'input_scaler':scaler,'gp':gp,'output_scaler':final_scale,'model':m,'direction':best['direction']}
    records['gp_ridge']={'candidates':candidates,'selected':best,'formulas':[str(p) for p in gp._best_programs],
                         'sample_rows':len(ids),'candidate_program_budget':config['gp_population']*config['gp_generations']}
    # X-like auxiliary ranker has no additional search. Lock orientation on validation.
    vp=x_rerank(validation,validation_predictions['lightgbm']);score,sign,raw=validation_score(validation,vp,config)
    predictions['x_rerank']=x_rerank(test,predictions['lightgbm'])*sign
    validation_predictions['x_rerank']=vp*sign
    records['x_rerank']={'selected':{'direction':sign,'absolute_validation_rank_ic':score,'raw_validation_rank_ic':raw},
                          'weights':{'pred_rank':1.,'vol20':-.2,'amount_ratio20':.1},'base':'lightgbm'}
    models['x_rerank']={'base':'lightgbm','direction':sign}
    return predictions,validation_predictions,models,records
