"""Freeze the R20 selection before opening its 2026 evaluation; never score data."""
import json
from pathlib import Path
import round20_holdout_runner as H

HERE = Path(__file__).resolve().parent
OUT = HERE / 'round20_frozen_selection.json'
if OUT.exists():
    raise FileExistsError(OUT)
paths = H.required_source_paths(include_development_report=True)
paths.update(protocol=HERE/'ROUND20_HOLDOUT_PROTOCOL_ZH.md', selection_freezer=Path(__file__).resolve())
primary = 'R20_joint_ensemble'
payload = dict(
    allow_test=True,
    selected_candidate=primary,
    comparisons=[name for name in H.ENSEMBLE_NAMES if name != primary],
    preprocessing_sha256=H.PREPROCESSING_SHA256,
    checkpoint_selected_epochs=H.selected_epochs(H.read_json(H.RESULTS)),
    source_hashes={key:dict(path=str(path.resolve()),sha256=H.digest(path)) for key,path in paths.items()},
    criteria=dict(
        primary_metric='equal-asset/equal-horizon QLIKE; primary minus R20_continued_ensemble',
        directional_confirmation='ensemble difference < 0 and at least two of three paired seed differences < 0',
        secondary='report all asset/horizon directions, normalized QLIKE, MSE, every fixed comparator and ablation',
        uncertainty='2000 synchronized noncircular moving-block draws; blocks22/44; report95percent intervals',
        claims='CI crossing zero allows only a point-improvement claim; no significance or true A/M/IV identification',
        mechanism='M has no established development contribution; joint-clock differences do not establish it',
        no_rescue='do not replace primary/metric/seed/epoch/asset/horizon using2026; MSE cannot rescue primary failure',
        value='judge magnitude and strongest nonoperator comparator separately from numerical direction; no publication guarantee'),
    development_note='Repeated2025 selection; joint epochs11/22/33=10/12/2. This file freezes one later evaluation.',
    training_scope='labels before2025 only;2025 selected epochs; no2025 refit;2026 updates causal observations only')
H.validate_selection_contract(payload)
with OUT.open('x', encoding='utf-8') as f:
    json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
H.authorize_test(OUT)
print('SELECTION_FROZEN',H.digest(OUT),'primary',primary,'test_not_executed')
