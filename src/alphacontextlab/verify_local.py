"""Replay every final checkpoint against hashed third-party quote-cache sequences."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import FEATURES, Transform, panel, sequences, fold_indices, sha256
from .models import make_model, predict, train
from .study import ridge_predict, write_json


def verify(config_path='configs/study.json',data_root='data',output_root='outputs',retrain_one=False):
    torch.set_num_threads(2)
    config=json.loads(Path(config_path).read_text());frame,manifest=panel(data_root,config);x,meta=sequences(frame,config)
    root=Path(output_root);verified=[]
    for path in sorted(root.glob('*/result.json')):
        record=json.loads(path.read_text());directory=path.parent
        if record['provenance']['data_sha256']!=manifest['quotes_sha256']:raise ValueError('Data mismatch')
        _,_,ids=fold_indices(meta,record['test_year']);raw=x[ids]
        checkpoint=directory/('model.npz' if record['arm']=='ridge' else 'model.pt')
        if sha256(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Checkpoint corruption')
        if record['arm']=='ridge':
            checkpoint=directory/'model.npz';state=np.load(checkpoint)
            transformation=json.loads((directory/'selection.json').read_text())['split']['refit_transform']
        else:
            checkpoint=directory/'model.pt';state=torch.load(checkpoint,weights_only=False,map_location='cpu')
            transformation=state['transform']
        # Only load this project's locally generated, hashed checkpoints.
        transform=Transform()
        for key,value in transformation.items():setattr(transform,key,np.array(value))
        xx,tt=transform.apply(raw)
        if record['arm']=='ridge':
            actual=ridge_predict((state['coef'],state['mean'],float(state['target_mean'])),xx)
        else:
            model=make_model(record['arm'],len(FEATURES),config,record['seed']);model.load_state_dict(state['state_dict'])
            actual=predict(model,xx,tt)/100
        stored=pd.read_csv(directory/'predictions.csv').prediction.to_numpy()
        error=float(np.max(np.abs(actual-stored)))
        if not np.allclose(actual,stored,rtol=1e-8,atol=1e-10):raise ValueError(f'Prediction replay mismatch {directory.name}: {error}')
        verified.append({'run':directory.name,'checkpoint_sha256':sha256(checkpoint),'max_absolute_prediction_error':error})
    expected=len(config['test_years'])*(1+3*len(config['seeds']))
    if len(verified)!=expected:raise ValueError('Incomplete checkpoint set')
    result={'status':'passed','verified_checkpoints':len(verified),'runs':verified,
            'scope':'Exact saved-parameter replay, not independent retraining.'}
    write_json(root/'local_replay_audit.json',result)
    if retrain_one:
        year=config['test_years'][0];seed=config['seeds'][0];directory=root/f'{year}-lstm-{seed}'
        record=json.loads((directory/'result.json').read_text())
        if sha256(directory/'model.pt')!=record['checkpoint_sha256']:raise ValueError('Checkpoint corruption')
        saved=torch.load(directory/'model.pt',weights_only=False,map_location='cpu')
        selection=json.loads((directory/'selection.json').read_text())['selected']
        ids=np.flatnonzero(((meta.exit_date<pd.Timestamp(year,1,1))&meta.forecast_valid).to_numpy())
        xx,tt=Transform().fit(x[ids]).apply(x[ids]);torch.use_deterministic_algorithms(True)
        model=make_model('lstm',len(FEATURES),config,seed)
        train(model,xx,tt,meta.iloc[ids].target.to_numpy(),config,selection['config'],seed,epochs=selection['best']['epoch'])
        errors={name:float((value-saved['state_dict'][name]).abs().max()) for name,value in model.state_dict().items()}
        retraining={'status':'passed' if max(errors.values())==0 else 'failed','run':directory.name,
                    'scope':'One same-environment deterministic retraining; all final runs also checked by saved-parameter replay.',
                    'max_parameter_absolute_error':max(errors.values()),'parameters':errors}
        write_json(root/'retraining_audit.json',retraining)
        if retraining['status']!='passed':raise ValueError('Retraining is not bitwise identical')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/study.json');p.add_argument('--data',default='data');p.add_argument('--outputs',default='outputs');p.add_argument('--retrain-one',action='store_true');a=p.parse_args()
    verify(a.config,a.data,a.outputs,a.retrain_one)
