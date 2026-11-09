"""Original small CPU models; Engram-inspired, not DeepSeek model reproduction."""
import math
import numpy as np
import torch
from torch import nn


class SequenceModel(nn.Module):
    def __init__(self, features, hidden=16, memory=False, buckets=(127,131,137,139), width=4, seed=17):
        super().__init__();self.memory=memory
        self.lstm=nn.LSTM(features,hidden,batch_first=True)
        self.head=nn.Linear(hidden,1)
        if memory:
            self.tables=nn.ModuleList([nn.Embedding(b,width) for b in buckets])
            d=len(buckets)*width
            self.key=nn.Linear(d,hidden,bias=False);self.value=nn.Linear(d,hidden,bias=False)
            rng=np.random.default_rng(seed)
            self.register_buffer('hash_multipliers',torch.tensor(rng.integers(1,100000,size=(len(buckets),3))*2+1,dtype=torch.long))
            self.buckets=tuple(buckets)
    def addresses(self,tokens):
        indices=[]
        for j,bucket in enumerate(self.buckets):
            size=2 if j<2 else 3
            # Fixed deterministic multi-head hashing of causal final state n-grams.
            value=tokens[:,-1]*self.hash_multipliers[j,0]
            for k in range(1,size):value=torch.bitwise_xor(value,tokens[:,-1-k]*self.hash_multipliers[j,k])
            indices.append(value.remainder(bucket))
        return indices
    def forward(self,x,tokens):
        h=self.lstm(x)[0][:,-1]
        if self.memory:
            m=torch.cat([table(idx) for table,idx in zip(self.tables,self.addresses(tokens))],dim=-1)
            key=self.key(m)
            norm=lambda t:t*torch.rsqrt(t.square().mean(-1,keepdim=True)+1e-6)
            affinity=(norm(h)*norm(key)).sum(-1,keepdim=True)/math.sqrt(h.shape[-1])
            gate=torch.sigmoid(affinity.sign()*affinity.abs().clamp_min(1e-6).sqrt())
            h=h+gate*self.value(m)
        return self.head(h).flatten()


def parameter_count(model):return sum(p.numel() for p in model.parameters())


def make_model(arm,features,config,seed):
    torch.manual_seed(seed)
    kwargs=dict(buckets=config['memory_buckets'],width=config['memory_width'],seed=seed)
    hidden=config['hidden_size']
    if arm=='lstm_capacity':
        reference=parameter_count(SequenceModel(features,hidden,memory=True,**kwargs))
        hidden=min(range(4,128),key=lambda d:abs(parameter_count(SequenceModel(features,d))-reference))
        torch.manual_seed(seed)
    return SequenceModel(features,hidden,memory=arm=='lstm_memory',**kwargs)


@torch.no_grad()
def predict(model,x,tokens,batch=512):
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[j:j+batch]),torch.from_numpy(tokens[j:j+batch])).numpy()
                           for j in range(0,len(x),batch)])


def train(model,x,tokens,y,config,candidate,seed,validation=None,epochs=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    optimizer=torch.optim.AdamW(model.parameters(),**candidate)
    x=torch.from_numpy(x);tokens=torch.from_numpy(tokens);y=torch.from_numpy(y.astype(np.float32))
    history=[];best=None;bad=0;limit=epochs or config['max_epochs']
    # Targets and outputs in percentage points improve numerical conditioning.
    for epoch in range(1,limit+1):
        model.train();total=0.
        for order in np.array_split(rng.permutation(len(x)),max(1,math.ceil(len(x)/config['batch_size']))):
            idx=torch.from_numpy(order);p=model(x[idx],tokens[idx]);loss=(p-y[idx]*100).square().mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=loss.item()*len(idx)
        record={'epoch':epoch,'train_scaled_mse':total/len(x)}
        if validation is not None:
            vx,vt,vy=validation;pred=predict(model,vx,vt)/100
            rmse=float(np.sqrt(np.mean((pred-vy)**2)));record['validation_rmse']=rmse
            if best is None or rmse<best['validation_rmse']-config['min_delta']:
                best=record.copy();bad=0
            else:bad+=1
        history.append(record)
        if validation is not None and bad>=config['patience']:break
    return history,best
