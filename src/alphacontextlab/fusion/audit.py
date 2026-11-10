"""Audit declared fusion weights and collector evidence without inventing performance."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from ..data import sha256
from .core import FeatureStore,timestamp


def run(reports='reports/fusion',local=None,config_path='configs/fusion.json'):
    r=Path(reports);c=json.loads(Path(config_path).read_text());m=json.loads((r/'acquisition.json').read_text())
    assert c['weights']=={'fundamental':.3,'price_volume':.4,'alternative':.3}
    assert m['weights']==c['weights'] and sha256(config_path)==m['config_sha256']
    frozen=subprocess.check_output(['git','show',m['source_commit']+':'+config_path])
    assert hashlib.sha256(frozen).hexdigest()==m['config_sha256']
    assert m['fusion_status']=='blocked' and not m['financial_performance_measured'] and not m['paper_admitted'] and m['orders_sent']==0
    sources=m['sources'];assert sources['x']['status']=='not_configured' and sources['x']['records']==0
    assert sources['satellite']['imagery_pixels_processed']==0 and sources['satellite']['economic_feature_observations']==0
    assert sources['polymarket']['stock_entity_links']==0
    fundamental=[x for x in sources['fundamental'] if 'symbol' in x]
    assert len({x['symbol'] for x in fundamental})==len(fundamental)
    total=sum(x['feature_observations'] for x in fundamental)+sources['news']['rows']+sources['polymarket']['observations']
    assert total==m['feature_observations']
    if local:
        root=Path(local);assert sha256(root/'features.jsonl')==m['features_sha256']
        store=FeatureStore.load(root/'features.jsonl');assert len(store.records)==total
        for meta in fundamental:
            import pandas as pd
            path=root/'fundamental'/(meta['symbol']+'-financial.csv');assert sha256(path)==meta['sha256']
            frame=pd.read_csv(path,dtype={'SECURITY_CODE':str});assert frame.SECURITY_CODE.str.zfill(6).eq(meta['symbol']).all()
        assert sha256(root/'alternative'/'polymarket.json')==sources['polymarket']['sha256']
        assert sha256(root/'alternative'/'satellite-catalogue.json')==sources['satellite']['sha256']
        # Newly observed snapshots cannot be used at the previous research cutoff.
        cutoff=timestamp('2026-10-30T15:59:59Z');assert all(x.available_at>cutoff for x in store.records.values())
    result={'status':'passed','group_weights':c['weights'],'fundamental_symbols':len(fundamental),
            'feature_observations':total,'fusion_status':m['fusion_status'],'historical_fusion_performance_measured':False,
            'local_snapshot_verified':bool(local)}
    print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',default='reports/fusion');p.add_argument('--local');a=p.parse_args();run(a.reports,a.local)
if __name__=='__main__':main()
