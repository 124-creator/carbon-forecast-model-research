"""Freeze this finite run before any market fit; refuses to overwrite."""
import json
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from run import STUDY, MODELS, NEURAL, SEEDS, DEFAULT, utc, sha, write_json
from data import SPECS


def main():
    output=STUDY/'provenance/PROTOCOL.json'
    if output.exists():
        raise FileExistsError('Preserve the frozen protocol')
    runtime=['source/data.py','source/learning.py','source/baselines.py','source/author_kernel.py','source/run.py']
    runtime += [str(p.relative_to(STUDY)) for p in sorted((STUDY/'vendor/r2d2').rglob('*')) if p.is_file()]
    evidence=['provenance/PRE_FIT_SCOPE.json','provenance/DATA_PREPARATION.json',
              'provenance/SYNTHETIC_SMOKE.json','provenance/SYNTHETIC_TIMING.json','provenance/WORKER_SYNTHETIC_SMOKE.json',
              'audits/INDEPENDENT_DATA_AUDIT.json','audits/TRANSPORT_SYNTHETIC_RECEIPT.json',
              'audits/LEARNING_SYNTHETIC_CHECK_V2.json','audits/BASELINES_SYNTHETIC_RECEIPT.json','methods/METHOD_ZH.md']
    tasks=[]
    for target,_,_ in SPECS:
        for budget in (100,400):
            for model in MODELS:
                for seed in (SEEDS if model in NEURAL else [17]):
                    tasks.append({'target':target,'budget':budget,'model':model,'seed':seed,
                                  'job':f'{target}_b{budget}_{model}_s{seed}'})
    assert len(tasks)==350
    protocol={'status':'FROZEN BEFORE FIRST FORMAL MARKET FIT','created_utc':utc(),
        'parent_commit':'1cba2ebaedd0fecd1891c90e322f0ccbcc5e2a99',
        'candidate':'transport of the target ridge prior with residual innovation penalty during profiled auxiliary learning',
        'novelty':'incremental learning-rule hypothesis; no global novelty or publication certification',
        'year':2022,'models':MODELS,'neural_models':NEURAL,'seeds':SEEDS,'static_seed':17,
        'seed_expansion_reason':'Pure synthetic timing220steps in1.37seconds supported three initialization seeds before any market outcome was computed; no market-result-driven extension',
        'config':DEFAULT,'tasks':tasks,'counts':{'neural_fits':330,'ridge_alpha_fits':50,'forecast_jobs':350,'market_retries':0},
        'data_exposure':'five approved pre2024 development files,2022/2023 previously exposed; EEX/EU excluded as mutual transfer sources',
        'supervision':'100/400 origin windows partition60/20/20 and240/80/80, nested same role endpoints, fullH20 purges; full causal past quotes allowed; not independent samples or real cold start',
        'source_cutoff':'complete labels strictly earlier than the common target meta first origin; pertarget same rich source histories and same source-episode schedule at both budgets',
        'training_loss':'all20 mean squared standardized cumulative price increments; causal sigma_t*sqrt(h); not aligned-WLS with evaluation metric',
        'primary_score':'equal market5 x budget2 x horizon3 average native-price MSE divided by same-cell RW MSE; average neural seed losses within each cell; not an ensemble',
        'validation':'same raw-price relative-MSE on horizons1,5,20; step0 eligible;restore selected state, then exactly one test forward; no final refit',
        'gate':{'candidate':'transport','strict_ratio':.995,'conditions':['all350jobscomplete','S_transport < .995*S_profile','S_transport < .995*S_covariance_profile','S_transport <=1'],
                'meaning':'finite resource gate, not significance or publishability; all conditions required'},
        'later_stages':'Only if gate passes may separately frozen2023 and newShenzhen confirmation be considered; none is used under this protocol',
        'failure_policy':'record failures with actual counters and private traces; no silent retry, jitter, extra hyperparameter grid or candidate rescue',
        'publication':'new study files only on new branch; preserve all parent files and main; no raw EEX prices, per-origin outputs or trained states in public tree',
        'compute':'CPU float64, deterministic torch, MKLDNN disabled, two processes, one thread each; max600updates per neural fit',
        'runtime':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'torch':torch.__version__},
        'source_sha256':{p:sha(STUDY/p) for p in runtime},'pre_run_evidence_sha256':{p:sha(STUDY/p) for p in evidence},
        'baseline_receipt_only_addition':{'previous_baseline_sha256':'198126e40e6fb01cb19ef6472e3343153899d76337340ac167c9d27791271e04',
            'current_sha256':sha(STUDY/'source/baselines.py'),'change':'added initial encoder hash to receipt only; numerical procedure unchanged; updated worker smoke executed afterwards'}}
    write_json(output,protocol)
    print(json.dumps({'protocol_sha256':sha(output),'created_utc':protocol['created_utc'],'tasks':len(tasks),'runtime_files':len(runtime)}))


if __name__=='__main__':
    main()
