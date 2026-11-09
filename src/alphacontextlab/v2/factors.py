"""Small hypothesis registry and bounded train-only template evolution; no eval/LLM."""
import numpy as np
from scipy.stats import rankdata

from .data import FACTOR_NAMES, THEMES


def correlation(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12:return 0.
    return float(np.corrcoef(a,b)[0,1])


class FactorSearch:
    def __init__(self,config):self.config=config
    def fit(self,x,y,years):
        values=x[:,-1,:];self.specs=[];self.ledger=[]
        for j,(name,theme) in enumerate(zip(FACTOR_NAMES,THEMES)):
            self.specs.append({'id':name,'op':'base','column':8+j,'theme':theme})
        def assess(spec):
            v=self.evaluate(spec,x)[:,-1];overall=correlation(rankdata(v),rankdata(y))
            sign=1. if overall>=0 else -1.
            scores=[sign*correlation(rankdata(v[years==year]),rankdata(y[years==year])) for year in sorted(set(years))]
            score=float(np.mean(scores)-.5*np.std(scores))
            record={'spec':spec,'train_pooled_rank_correlation':overall,'orientation':sign,'year_correlations':scores,
                    'stability_score':score,'finite':bool(np.isfinite(v).all()),'standard_deviation':float(v.std())}
            self.ledger.append(record);return record
        base=[assess(s) for s in self.specs]
        parents=sorted(base,key=lambda r:(-r['stability_score'],r['spec']['id']))[:self.config['parents']]
        for p in parents:
            spec=p['spec'];scale=max(p['standard_deviation'],1e-6)
            for op in ['tanh','vol_gate']:
                child={'id':spec['id']+'__'+op,'op':op,'parent':spec,'scale':scale,'theme':spec['theme']}
                self.specs.append(child);assess(child)
        for j in range(self.config['crossovers']):
            a=parents[j%len(parents)];b=parents[(j+1)%len(parents)]
            spec={'id':a['spec']['id']+'__mix__'+b['spec']['id'],'op':'mix','a':a['spec'],'b':b['spec'],
                  'scale_a':max(a['standard_deviation'],1e-6),'scale_b':max(b['standard_deviation'],1e-6),
                  'sign_a':a['orientation'],'sign_b':b['orientation'],'theme':'fusion'}
            self.specs.append(spec);assess(spec)
        self.selected=[];chosen=[]
        for record in sorted(self.ledger,key=lambda r:(-r['stability_score'],r['spec']['id'])):
            if not record['finite'] or record['standard_deviation']<1e-8:
                record['disposition']='reject_numerical';continue
            v=self.evaluate(record['spec'],x)[:,-1]
            if any(abs(correlation(v,other))>=self.config['max_absolute_correlation'] for other in chosen):
                record['disposition']='reject_redundant';continue
            if len(self.selected)>=self.config['select']:
                record['disposition']='reject_budget';continue
            record['disposition']='selected';self.selected.append(record);chosen.append(v)
        return self
    def evaluate(self,spec,x):
        op=spec['op']
        if op=='base':return x[:,:,spec['column']]
        if op=='tanh':return np.tanh(self.evaluate(spec['parent'],x)/spec['scale'])
        if op=='vol_gate':return self.evaluate(spec['parent'],x)/(1+x[:,:,3]/.02)
        if op=='mix':return .5*(spec['sign_a']*self.evaluate(spec['a'],x)/spec['scale_a']+spec['sign_b']*self.evaluate(spec['b'],x)/spec['scale_b'])
        raise ValueError('Unregistered factor operation')
    def transform(self,x,arm):
        if arm=='basic_ridge':return x[:,:,:8].copy()
        if arm=='static_ridge':return x.copy()
        extra=[r['orientation']*self.evaluate(r['spec'],x) for r in self.selected]
        return np.concatenate([x[:,:,:8],np.stack(extra,axis=-1)],axis=-1).astype(np.float32)
    def state(self):return {'selected':self.selected,'ledger':self.ledger,'budget':self.config,'selection_data':'fit_partition_only_no_validation_or_test'}


class Scale:
    def fit(self,x):
        self.mean=x.reshape(-1,x.shape[-1]).mean(0,dtype=np.float64)
        self.std=np.maximum(x.reshape(-1,x.shape[-1]).std(0,dtype=np.float64),1e-6);return self
    def apply(self,x):return ((x-self.mean)/self.std).astype(np.float32)
    def state(self):return {'mean':self.mean.tolist(),'std':self.std.tolist()}
