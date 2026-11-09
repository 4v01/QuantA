"""Versioned third-party quotes and causal, close-available features."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(config, root):
    root = Path(root); raw = root / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    rows = []; sources = []
    for symbol in config['symbols']:
        for adjustment in (0, 1):
            path = raw / f'{symbol}_fqt{adjustment}.json'
            url = ('https://push2his.eastmoney.com/api/qt/stock/kline/get?'
                   f'secid={0 if symbol.startswith("1") else 1}.{symbol}&klt=101&fqt={adjustment}'
                   f'&beg={config["start"].replace("-", "")}&end={config["end"].replace("-", "")}'
                   '&lmt=10000&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61')
            if not path.exists():
                with urlopen(url, timeout=45) as response:
                    payload = response.read()
                obj = json.loads(payload)
                if obj.get('rc') != 0 or not obj.get('data', {}).get('klines'):
                    raise ValueError(f'No valid quotes: {symbol}, fqt={adjustment}')
                path.write_bytes(payload)
            obj = json.loads(path.read_bytes()); data = obj['data']
            frame = pd.DataFrame([s.split(',') for s in data['klines']],
                                 columns=['date','open','close','high','low','volume','amount','amplitude','pct','change','turnover'])
            frame['date'] = pd.to_datetime(frame.date)
            for column in frame.columns[1:]: frame[column] = pd.to_numeric(frame[column], errors='raise')
            if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
                raise ValueError('Unsorted or duplicate dates')
            if not ((frame.open > 0) & (frame.low > 0) & (frame.high >= frame.low)).all():
                raise ValueError('Invalid OHLC')
            frame['symbol'] = symbol; frame['adjustment'] = adjustment; rows.append(frame)
            sources.append({'symbol':symbol,'adjustment':adjustment,'url':url,'sha256':sha256(path),
                            'rows':len(frame),'first':str(frame.date.min().date()),'last':str(frame.date.max().date()),
                            'name':data['name']})
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(root / 'quotes.csv', index=False, float_format='%.12g')
    fingerprint = sha256(root / 'quotes.csv')
    manifest = {'provider':'Eastmoney public historical daily kline, unadjusted and forward-adjusted',
                'retrieved_utc':datetime.now(timezone.utc).isoformat(),'quotes_sha256':fingerprint,
                'sources':sources,'limitations':['Third-party data, not an exchange-certified archive.',
                'Current forward-adjustment factors are retrospectively revised and rounded; not point-in-time.',
                'No permission to redistribute raw provider data is assumed; raw files stay local.']}
    (root/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


FEATURES = ['ret1','ret5','ret20','vol20','range','overnight','volume_relative','market_ret1']


def panel(root, config):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if sha256(root/'quotes.csv') != manifest['quotes_sha256']: raise ValueError('Quote fingerprint mismatch')
    quotes = pd.read_csv(root/'quotes.csv', dtype={'symbol':str}, parse_dates=['date'])
    adj = quotes[quotes.adjustment == 1].copy(); raw = quotes[quotes.adjustment == 0].copy()
    expected = pd.DatetimeIndex(sorted(adj.date.unique()))
    for kind, data in [('adjusted', adj), ('raw',raw)]:
        for symbol in config['symbols']:
            if not pd.DatetimeIndex(data[data.symbol == symbol].date).equals(expected):
                raise ValueError(f'Missing sessions: {kind} {symbol}; do not silently intersect calendars')
    market = adj[adj.symbol == '510300'].set_index('date').close.pct_change()
    frames = []
    for symbol in config['symbols']:
        f = adj[adj.symbol == symbol].set_index('date').copy()
        unadjusted = raw[raw.symbol == symbol].set_index('date')
        f['raw_open'] = unadjusted.open; f['raw_close'] = unadjusted.close
        f['ret1'] = f.close.pct_change(); f['ret5'] = f.close.pct_change(5); f['ret20'] = f.close.pct_change(20)
        f['vol20'] = f.ret1.rolling(20, min_periods=20).std(ddof=0)
        f['range'] = (f.high-f.low)/f.close
        f['overnight'] = f.open/f.close.shift(1)-1
        f['volume_relative'] = np.log((f.volume+1)/(f.volume.rolling(20, min_periods=20).mean()+1))
        f['market_ret1'] = market
        f['entry_date'] = f.index.to_series().shift(-1)
        f['exit_date'] = f.index.to_series().shift(-2)
        f['target'] = f.open.shift(-2)/f.open.shift(-1)-1
        # At entry, compare adjusted opening with last close (same adjustment basis).
        f['entry_gap'] = f.open.shift(-1)/f.close-1
        f['exit_gap'] = f.open.shift(-2)/f.close.shift(-1)-1
        frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True), manifest


def sequences(frame, config):
    xs=[]; metadata=[]; length=config['sequence_length']
    for symbol in config['symbols']:
        f=frame[frame.symbol == symbol].sort_values('date').reset_index(drop=True)
        values=f[FEATURES].to_numpy(dtype=np.float32)
        for end in range(length-1,len(f)):
            block=values[end-length+1:end+1]
            row=f.iloc[end]
            if not np.isfinite(block).all() or not np.isfinite(row.target): continue
            xs.append(block);metadata.append(row[['date','symbol','entry_date','exit_date','target','entry_gap','exit_gap','ret20']].to_dict())
    meta=pd.DataFrame(metadata)
    order=meta.sort_values(['entry_date','symbol'],kind='stable').index.to_numpy()
    return np.stack(xs)[order], meta.iloc[order].reset_index(drop=True)


def fold_indices(meta, test_year):
    # Every scored holding interval is wholly inside its named partition.
    val_start=pd.Timestamp(test_year-1,1,1); test_start=pd.Timestamp(test_year,1,1); end=pd.Timestamp(test_year+1,1,1)
    train=np.flatnonzero((meta.exit_date<val_start).to_numpy())
    val=np.flatnonzero(((meta.entry_date>=val_start)&(meta.exit_date<test_start)).to_numpy())
    test=np.flatnonzero(((meta.entry_date>=test_start)&(meta.exit_date<end)).to_numpy())
    if min(len(train),len(val),len(test))==0: raise ValueError('Empty fold')
    assert meta.iloc[train].exit_date.max()<meta.iloc[val].entry_date.min()
    assert meta.iloc[val].exit_date.max()<meta.iloc[test].entry_date.min()
    return train,val,test


class Transform:
    def fit(self, x):
        # Only the supplied fit partition; no validation/test quantiles or moments.
        flat=x.reshape(-1,x.shape[-1]).astype(np.float64)
        self.mean=flat.mean(0);self.std=np.maximum(flat.std(0),1e-6)
        self.return_edges=np.quantile(flat[:,0],[.25,.5,.75])
        self.volatility_edges=np.quantile(flat[:,3],[1/3,2/3])
        return self
    def apply(self,x):
        tokens=np.searchsorted(self.return_edges,x[:,:,0],side='right')*3+np.searchsorted(self.volatility_edges,x[:,:,3],side='right')
        return ((x-self.mean)/self.std).astype(np.float32),tokens.astype(np.int64)
    def state(self):
        return {k:v.tolist() for k,v in vars(self).items()}
