"""Disjoint selection/calibration windows and explicit return magnitudes."""
import numpy as np
import pandas as pd
from ..v7.models import fit_method
from ..v7.study import split


def nested_split(panel, year, half, base, config):
    a, b, r, q, info = split(panel, year, half, base['long_train_months'], base)
    begin = pd.Timestamp(info['test_start'])
    boundary = begin - pd.DateOffset(months=config['calibration_months'])
    select = b[(b.date < boundary) & (b.exit20 < boundary)].copy()
    calibrate = b[b.date >= boundary].copy()
    earlier = r[(r.date < boundary) & (r.exit20 < boundary)].copy()
    if min(len(select), len(calibrate), len(earlier)) < 500:
        raise ValueError('Insufficient nested selection/calibration data')
    assert a.exit20.max() < select.date.min()
    assert earlier.exit20.max() < calibrate.date.min()
    assert select.exit20.max() < calibrate.date.min()
    assert calibrate.exit20.max() < begin and r.exit20.max() < begin
    info.update({'calibration_start': str(boundary), 'selection_rows': len(select),
                 'calibration_rows': len(calibrate), 'earlier_refit_rows': len(earlier),
                 'selection_exit20_max': str(select.exit20.max()),
                 'earlier_refit_exit20_max': str(earlier.exit20.max()),
                 'calibration_exit20_max': str(calibrate.exit20.max())})
    return a, select, earlier, calibrate, r, q, info


def calibrate_slope(frame, scores, config):
    d = frame[['date', 'label']].copy(); d['score'] = scores
    d['x'] = d.score - d.groupby('date').score.transform('mean')
    d['y'] = d.label - d.groupby('date').label.transform('mean')
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    denom = float(d.x @ d.x)
    raw = float(d.x @ d.y / denom) if denom > 1e-12 else 0.
    beta = float(np.clip(raw, 0., config['maximum_calibration_slope']) * config['calibration_slope_shrinkage'])
    return {'raw_slope': raw, 'slope': beta, 'rows': len(d), 'score_sum_squares': denom,
            'first_date': str(d.date.min()), 'last_date': str(d.date.max()),
            'label_exit_max': str(frame.exit_date.max()), 'sign_flipped': False}


def fit_fold(panel, year, half, base, config):
    a, b, earlier, cal, r, q, info = nested_split(panel, year, half, base, config)
    pcal, _, mcal, chosen = fit_method(a, b, earlier, cal, config['base_method'], base)
    p, _, final, chosen2 = fit_method(a, b, r, q, config['base_method'], base)
    assert chosen == chosen2
    calibration = calibrate_slope(cal, pcal['fusion'], config)
    result = q[['date', 'symbol', 'label', 'target', 'exit_date']].copy()
    result['prediction'] = p['fusion']
    result['expected_excess'] = calibration['slope'] * (result.prediction - result.groupby('date').prediction.transform('mean'))
    crows = cal[['date', 'symbol', 'label', 'exit_date']].copy(); crows['prediction'] = pcal['fusion']
    return result, crows, {'calibration': mcal, 'final': final}, {'split': info, 'groups': chosen, 'return_calibration': calibration}
