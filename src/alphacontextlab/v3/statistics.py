"""Selection-aware descriptive checks; confirmatory gate remains separately frozen."""
import numpy as np


def centered_max_check(differences,repetitions=1000,block=20,seed=2026):
    """White-inspired unstudentized circular-block max-mean null bootstrap.

    Conditional on the supplied finite candidate family and observed validation
    data. Not an implementation of every assumption of White's asymptotic test,
    not Hansen SPA, and not a correction for unrecorded historical research.
    """
    d=np.asarray(differences,dtype=float)
    if d.ndim!=2 or len(d)<2 or not np.isfinite(d).all():raise ValueError('Invalid candidate return matrix')
    n=len(d);means=d.mean(0);observed=float(np.sqrt(n)*max(0.,means.max()));centered=d-means
    rng=np.random.default_rng(seed);greater=0
    for _ in range(repetitions):
        start=rng.integers(0,n,size=int(np.ceil(n/block)))
        ids=((start[:,None]+np.arange(block))%n).reshape(-1)[:n]
        value=np.sqrt(n)*max(0.,centered[ids].mean(0).max())
        greater+=bool(value>=observed-1e-15)
    return {'candidates':d.shape[1],'days':n,'statistic':observed,'conditional_pvalue':float((1+greater)/(repetitions+1)),
            'repetitions':repetitions,'block':block,'scope':'Descriptive validation-family centered max bootstrap; not global research significance.'}


def paired_excess(a,b,repetitions=1000,block=20,comparisons=3,seed=2026):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:raise ValueError('Unpaired confirmation paths')
    d=a-b;rng=np.random.default_rng(seed);values=[]
    for _ in range(repetitions):
        start=rng.integers(0,len(d),size=int(np.ceil(len(d)/block)))
        ids=((start[:,None]+np.arange(block))%len(d)).reshape(-1)[:len(d)]
        values.append(float(d[ids].mean()*252))
    alpha=.05/comparisons
    return {'annualized_mean_excess':float(d.mean()*252),'bonferroni_interval':np.quantile(values,[alpha/2,1-alpha/2]).tolist(),
            'interval_coverage':1-alpha,'family_comparisons':comparisons,'block':block,'repetitions':repetitions}
