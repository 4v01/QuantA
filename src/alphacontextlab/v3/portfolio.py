"""Small convex SPO problems; bounded adaptations of the cited research."""
import numpy as np
from scipy.optimize import minimize

from ..v2.execution import choose


def solve(mu,cov,pre,gamma,cost,config):
    n=len(mu);hcov=(cov+cov.T)/2*config['horizon']
    # Scale the objective to keep optimizer tolerances meaningful for return units.
    def value(v):
        w,z=v[:n],v[n:];return float(10000*(.5*gamma*w@hcov@w-mu@w+cost*z.sum()))
    def gradient(v):return 10000*np.r_[gamma*hcov@v[:n]-mu,np.full(n,cost)]
    a=np.block([[-np.eye(n),np.eye(n)],[np.eye(n),np.eye(n)]])
    b=np.r_[pre,-pre]
    constraint={'type':'ineq','fun':lambda v:a@v+b,'jac':lambda v:a}
    budget={'type':'ineq','fun':lambda v:1-v[:n].sum(),'jac':lambda v:np.r_[-np.ones(n),np.zeros(n)]}
    initial=np.r_[pre,np.zeros(n)]
    result=minimize(value,initial,jac=gradient,bounds=[(0.,1.)]*(2*n),constraints=[constraint,budget],
                    method='SLSQP',options={'ftol':config['solver_ftol'],'maxiter':config['solver_maxiter']})
    feasible=np.isfinite(result.x).all() and result.x[:n].min()>=-1e-7 and result.x[:n].sum()<=1+1e-7 and (a@result.x+b).min()>=-1e-7
    if not result.success or not feasible or value(result.x)>value(initial)+1e-5:
        return pre.copy(),{'success':False,'message':str(result.message),'iterations':int(result.nit)}
    w=np.maximum(result.x[:n],0.)
    if w.sum()>1:w/=w.sum()
    return w,{'success':True,'message':'converged','iterations':int(result.nit)}


def weights(mu,downside,market,signal,pre,policy,gamma,config):
    if policy=='cash':return np.zeros_like(pre),{'success':True}
    if policy=='csi300':
        w=np.zeros_like(pre);w[market.symbols.index('510300')]=1.;return w,{'success':True}
    if policy=='equal_weekly':return np.ones_like(pre)/len(pre),{'success':True}
    if policy=='inverse_volatility':
        cov=market.lw_cov[signal];vol=np.sqrt(np.maximum(np.diag(cov),1e-10));w=1/vol;w/=w.sum()
        risk=np.sqrt(max(w@cov@w,1e-12)*252);w*=min(1.,.1/risk);return w,{'success':True}
    spec=config['policies'][policy];cov=market.lw_cov[signal] if spec['covariance']=='lw' else market.sample_cov[signal]
    if spec['kind']=='legacy':
        scale=np.sqrt(np.maximum(np.diag(cov),1e-12));corr=.9*cov/scale[:,None]/scale[None,:]+.1*np.eye(len(pre))
        w,_=choose(mu,np.maximum(downside,0),corr,pre,{'risk_penalty':.1,'cost_aware':True},config['assumed_cost_bps']/10000)
        return w,{'success':True}
    cost=config['assumed_cost_bps']/10000 if spec['cost_aware'] else 0.
    w,status=solve(mu*spec['mu_shrink'],cov,pre,gamma,cost,config)
    w=pre+spec['step']*(w-pre)
    if spec['variance_managed']:
        variance=float(w@cov@w);reference=config['variance_scale_annual']**2/252
        w*=min(1.,reference/max(variance,1e-12))
    return w,status
