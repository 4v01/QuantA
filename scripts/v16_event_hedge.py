"""One exploratory event stress rule applied at the next mainland market open."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v12_market_trend import overlay
from alphacontextlab.v10_clock.ledger import apply_cost
from alphacontextlab.v4.metrics import path_statistics

root = Path('reports/v16_sources')
metadata = json.loads((root/'taiwan_history.json').read_text())
quotes = pd.DataFrame(metadata['response']['history']).sort_values('t')
quotes['timestamp'] = pd.to_datetime(quotes.t,unit='s',utc=True)
series = quotes.set_index('timestamp').p
assert series.index.is_unique and series.between(0,1).all()
rows = []
daily = []
for name, source in [('downside','reports/v15_downside/evaluation/daily.csv.gz'),
                     ('equal_weight','reports/v11_retention/daily.csv.gz')]:
    shadow = pd.read_csv(source).query('cost_bps == 0').copy()
    if name == 'equal_weight':
        shadow = shadow.query("phase == 'evaluation' and policy == 'equal_weight'").copy()
    shadow = shadow.set_index('date').sort_index()
    cutoff = pd.DatetimeIndex(shadow.signal_date).tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
    def lookup(index):
        return series.reindex(index,method='ffill',tolerance=pd.Timedelta(hours=36)).to_numpy()
    delta = lookup(cutoff) - lookup(cutoff-pd.Timedelta(days=7))
    covered = np.isfinite(delta) & (cutoff <= pd.Timestamp(metadata['end']))
    trigger = covered & (delta >= .01)
    for method,off in [('baseline',trigger & False),('event',trigger)]:
        p = apply_cost(overlay(shadow,pd.Series(~off,index=shadow.index)),20)
        p['covered'] = covered
        p['cash_signal'] = off
        p['delta_7days'] = delta
        daily.append(p[['signal_date','covered','cash_signal','delta_7days','entry_fee','net_return','nav']].assign(portfolio=name,rule=method))
        for period,part in [('full',p),('covered_sessions',p[covered])]:
            rows.append(dict(portfolio=name,rule=method,period=period,cash_days=int(part.cash_signal.sum()),**path_statistics(part)))
pd.concat(daily).to_csv(root/'event_daily.csv.gz',compression={'method':'gzip','mtime':0})
pd.DataFrame(rows).to_csv(root/'event_metrics.csv',index=False)
