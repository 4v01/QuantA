"""CPU return/downside heads with common training inputs and validation boundaries."""
import copy
import math

import numpy as np
from scipy.linalg import solve
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
from torch import nn


def tabular(x):return np.concatenate([x[:,-1],x.mean(1)],axis=1).astype(np.float64)


class Ridge:
    def __init__(self,alpha):self.alpha=alpha
    def fit(self,x,y):
        z=tabular(x);self.mean=z.mean(0);self.target_mean=y.mean(0);zc=z-self.mean
        self.coef=solve(zc.T@zc+self.alpha*np.eye(z.shape[1]),zc.T@(y-self.target_mean),assume_a='pos');return self
    def predict(self,x):
        p=(tabular(x)-self.mean)@self.coef+self.target_mean;p[:,1]=np.maximum(p[:,1],0);return p


class Trees:
    def __init__(self,config,candidate,seed):
        self.models=[HistGradientBoostingRegressor(max_iter=config['hgb_max_iter'],learning_rate=.05,
                     min_samples_leaf=50,early_stopping=False,random_state=seed,**candidate) for _ in range(2)]
    def fit(self,x,y):
        z=tabular(x)
        for j,m in enumerate(self.models):m.fit(z,y[:,j]*100)
        return self
    def predict(self,x):
        p=np.column_stack([m.predict(tabular(x))/100 for m in self.models]);p[:,1]=np.maximum(p[:,1],0);return p


class LSTM(nn.Module):
    def __init__(self,features,hidden,seed):
        super().__init__();torch.manual_seed(seed)
        self.encoder=nn.LSTM(features,hidden,batch_first=True);self.head=nn.Linear(hidden,2)
    def forward(self,x):return self.head(self.encoder(x)[0][:,-1])
    @torch.no_grad()
    def predict(self,x):
        self.eval();values=np.concatenate([self(torch.from_numpy(part)).numpy() for part in np.array_split(x,max(1,math.ceil(len(x)/512)))])/100
        values[:,1]=np.maximum(values[:,1],0);return values


def fit_lstm(model,x,y,config,candidate,seed,validation=None,epochs=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);opt=torch.optim.AdamW(model.parameters(),**candidate)
    xx=torch.from_numpy(x);yy=torch.from_numpy((y*100).astype(np.float32));history=[];best=None;best_state=None;bad=0
    for epoch in range(1,(epochs or config['max_epochs'])+1):
        model.train();total=0.
        for order in np.array_split(rng.permutation(len(x)),max(1,math.ceil(len(x)/config['batch_size']))):
            ids=torch.from_numpy(order);p=model(xx[ids]);loss=(p-yy[ids]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite LSTM objective')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(ids)
        record={'epoch':epoch,'train_scaled_joint_mse':total/len(x)}
        if validation is not None:
            vx,vy=validation;prediction=model.predict(vx);value=float(np.sqrt(np.mean((prediction-vy)**2)))
            record['validation_joint_rmse']=value
            if best is None or value<best['validation_joint_rmse']-config['min_delta']:
                best=record.copy();best_state=copy.deepcopy(model.state_dict());bad=0
            else:bad+=1
        history.append(record)
        if validation is not None and bad>=config['patience']:break
    if best_state is not None:model.load_state_dict(best_state)
    return history,best
