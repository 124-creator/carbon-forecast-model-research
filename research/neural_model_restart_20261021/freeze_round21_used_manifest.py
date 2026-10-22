"""Bind an explicitly reused-period R21 diagnostic; never a fresh test."""
import json
from pathlib import Path
import round21_frozen_diagnostic as Q
out = Path(__file__).resolve().parent/'round21_used_manifest.json'
if out.exists():
    raise FileExistsError(out)
paths = Q.source_paths('used2026')
paths['manifest_freezer'] = Path(__file__).resolve()
payload = dict(allow_test=True, used_period_diagnostic=True,
    not_independent_confirmation=True, proposed_series=Q.PROPOSED,
    preprocessing_sha256=Q.RR.PREPROCESSING_SHA256,
    checkpoint_epochs=Q.epoch_bindings(Q.read_json(Q.RESULTS)),
    source_hashes={key:dict(path=str(Path(path).resolve()),sha256=Q.digest(path)) for key,path in paths.items()},
    status='DESCRIPTIVE REUSE OF ALREADY USED2026; NO MODEL OR EPOCH RESELECTION')
Q.validate_used_flags(payload)
with out.open('x',encoding='utf-8') as handle:
    json.dump(payload,handle,ensure_ascii=False,indent=2,allow_nan=False)
Q.authorize_used_period(out)
print('DIAGNOSTIC_MANIFEST_BOUND', Q.digest(out))
