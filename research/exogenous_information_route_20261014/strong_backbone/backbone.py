"""Exact historical MLE480 execution; fixed coal contract, no new estimator."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE_ROOT=Path('E:/碳价格/独立数据归档/carbon-forecast-model-research')
SOURCE=SOURCE_ROOT/'research/deep_direction_review_20261013/calibration/calibrate.py'
spec=importlib.util.spec_from_file_location('historical_calibrate_strong',SOURCE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
# Original reader writes an audit; redirect ONLY its output, never historical files.
base.OUT=HERE
H=base.H
COAL_FIELDS=('coal_price','coal_change_5','coal_age_sessions','available_at')

def read_prefix():
    return base.read_prefix()

def coal_features(panel,origins):
    """Already issuance-aligned real panel only; availability is session index.

    Fixed columns: coal level, past-five-release change, age in carbon sessions.
    Caller must compute fields from timestamped actual releases, not future values.
    No panel is invented; incomplete/all-new-input-missing rows remain unavailable.
    """
    origins=np.asarray(origins,int)
    if panel is None or any(k not in panel for k in COAL_FIELDS):
        return None,np.zeros(len(origins),bool)
    arrays=[np.asarray(panel[k],float) for k in COAL_FIELDS]
    if any(a.shape!=(len(origins),) for a in arrays):
        raise ValueError('coal panel must be issuance-aligned, one scalar per origin')
    available=np.column_stack(arrays)
    valid=np.isfinite(available).all(1)
    if np.any(valid & ((arrays[3]>origins)|(arrays[2]<0))):
        raise ValueError('future release or negative age')
    return available[:,:3],valid

def forecast(p,fit_end,origins,panel=None):
    origins=np.asarray(origins,int)
    if not len(origins) or fit_end<2 or np.any(origins<fit_end) or np.any(origins>=len(p)):
        raise ValueError('issuance/fit boundary')
    pred,log=base.predict(p,fit_end,origins,dict(family='MLE',window=480))
    features,valid=coal_features(panel,origins)
    log.update(route='price_only_fallback' if panel is not None else 'price_only',
               coal_valid_rows=int(valid.sum()),coal_correction_enabled=False,
               status='interface_only_pending_real_panel_and_separate_frozen_estimator',
               coal_fields=list(COAL_FIELDS))
    # Even complete inputs do not silently fit an unregistered exogenous estimator.
    return pred,log

def replay(dates,p):
    initial=int(np.sum(dates<'2020-01-01'))
    origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
    if len(p)!=950 or len(origins)!=451:raise ValueError('950/451 protocol')
    prediction=np.full((len(origins),3),np.nan);logs=[]
    for fit in range(initial,len(p),40):
        pos=np.where((origins>=fit)&(origins<fit+40))[0]
        if len(pos):prediction[pos],log=forecast(p,fit,origins[pos]);logs.append(log)
    return origins,prediction,logs
