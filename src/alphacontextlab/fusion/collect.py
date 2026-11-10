"""Read-only smoke acquisition and machine-readable coverage; no historic fusion claims."""
import argparse,json,subprocess
from pathlib import Path
from ..study import write_json
from ..data import sha256
from ..v4.data import utcnow
from .adapters import fundamentals,news_cache,polymarket,satellite_catalogue
from .core import FeatureStore


def collect(config_path='configs/fusion.json',root=None,symbols=None):
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Commit collector/config before snapshot acquisition')
    config=json.loads(Path(config_path).read_text());root=Path(root or ('data/fusion/'+utcnow().replace(':','').replace('+','_')))
    if root.exists() and any(root.iterdir()):raise ValueError('Use new immutable snapshot directory')
    root.mkdir(parents=True,exist_ok=True);store=FeatureStore();status={'fundamental':[]};failures=[]
    def attempt(name,fn):
        try:
            rows,metadata=fn()
            for row in rows:store.add(row)
            print(name,json.dumps(metadata,ensure_ascii=False),flush=True);return metadata
        except Exception as e:
            error={'source':name,'error':repr(e)};failures.append(error);print('FAILED',error,flush=True);return {'status':'failed','error':repr(e)}
    for symbol in (symbols or ['600000','002230','600519']):
        status['fundamental'].append(attempt('fundamental:'+symbol,lambda s=symbol:fundamentals(s,root/'fundamental',config['fundamental_fields'])))
    caches=sorted(Path('data/v4_news').glob('*.json'))
    status['news']=attempt('news',lambda:news_cache(caches[-1])) if caches else {'status':'no_news_cache'}
    status['polymarket']=attempt('polymarket',lambda:polymarket(root/'alternative'))
    status['satellite']=attempt('satellite',lambda:satellite_catalogue(root/'alternative'))
    status['x']={'status':'not_configured','records':0,'reason':'No authorized X export or configured API access; no paid requests attempted'}
    store.save(root/'features.jsonl')
    report={'created_at':utcnow(),'source_commit':source,
            'config_sha256':sha256(config_path),'features_sha256':sha256(root/'features.jsonl'),'feature_observations':len(store.records),
            'weights':config['weights'],'sources':status,'failures':failures,'fusion_status':'blocked',
            'blocking_reasons':['No trained/calibrated three-group OOF signal set','X input unavailable','Satellite economic features/site mappings unavailable',
                                'Polymarket to security/industry mappings absent','No historic first-seen archives'],
            'financial_performance_measured':False,'paper_admitted':False,'orders_sent':0}
    write_json(root/'acquisition.json',report);print('SNAPSHOT',root,flush=True);return root,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--config',default='configs/fusion.json');p.add_argument('--all-symbols',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_text());symbols=json.loads(Path(config['universe_config']).read_text())['symbols'] if a.all_symbols else None
    collect(a.config,a.root,symbols)
if __name__=='__main__':main()
