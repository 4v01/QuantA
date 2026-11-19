"""Replay one fixed retention change on frozen V9 signals and V10 accounts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alphacontextlab.v10_clock.ledger import apply_cost, simulate
from alphacontextlab.v10_clock.study import inputs
from alphacontextlab.v8.study import path_metrics


def main():
    config = json.loads(Path('configs/v10_clock.json').read_text())
    base = json.loads(Path(config['base_config']).read_text())
    out = Path('reports/v11_retention')
    out.mkdir(exist_ok=True)
    rows = []
    paths = []
    for phase in ('development', 'evaluation'):
        predictions, tape, symbols = inputs(phase, config)
        predictions = predictions[predictions.method == 'economic_h5']
        old = pd.read_csv(Path('reports/v10_clock') / phase / 'summary.csv')
        for policy, limit in [('original', 0.4), ('wider_exit', 0.6), ('equal_weight', 0.4)]:
            benchmark = policy == 'equal_weight'
            gross, *_ = simulate(predictions, tape, symbols, dict(base, buffer_exit_fraction=limit),
                                  config, 'stagger5', benchmark)
            for cost in [0, *config['cost_bps']]:
                path = apply_cost(gross, cost)
                metrics = path_metrics(path)
                if policy != 'wider_exit' and cost:
                    method = 'equal_weight' if benchmark else 'economic_h5'
                    reference = old.query("method == @method and clock == 'stagger5' and cost_bps == @cost").iloc[0]
                    np.testing.assert_allclose(metrics['net_sharpe'], reference.net_sharpe, atol=1e-10)
                rows.append(dict(phase=phase, policy=policy, cost=cost, **metrics))
                paths.append(path.assign(phase=phase, policy=policy))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    pd.concat(paths).to_csv(out / 'daily.csv.gz', index=False, compression={'method':'gzip','mtime':0})


if __name__ == '__main__':
    main()
