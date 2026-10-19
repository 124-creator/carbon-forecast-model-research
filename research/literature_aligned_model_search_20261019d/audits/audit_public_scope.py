#!/usr/bin/env python3
"""Read-only review of proposed public study files and old-study changes.

The project lead explicitly permits numeric state_dict representations of our
own candidate/control models. Torch checkpoint files, new unlicensed author
tables, and point-level observations/forecast paths remain outside this scope.
The licensed ScienceDB source and its prepared data are intentionally public.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def sha_bytes(b):return hashlib.sha256(b).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--study',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Use a new public-scope receipt path')
    rel=args.study.resolve().relative_to(args.repo.resolve()).as_posix()
    names=git(args.repo,'ls-files','--cached','--others','--exclude-standard','--',rel).splitlines()
    names=sorted(set(names))
    history=git(args.repo,'log','--reverse','--format=%H','--',rel).splitlines()
    first=history[0];baseline=git(args.repo,'rev-parse',first+'^');head=git(args.repo,'rev-parse','HEAD')
    changed=git(args.repo,'diff','--name-only',baseline,'--','research').splitlines()
    old_changes=[p for p in changed if not p.startswith(rel+'/')]
    untracked=git(args.repo,'ls-files','--others','--exclude-standard','--','research').splitlines()
    old_untracked=[p for p in untracked if not p.startswith(rel+'/')]
    author_receipt=args.study/'provenance/PUBLIC_AUTHOR_DOWNLOAD_CLI_CHECK.json'
    authors=json.loads(author_receipt.read_text())
    forbidden_hashes={r['sha256']:r['local'] for r in authors['files']}
    forbidden_hashes.update({
        '1e36a5c1f19e271adef449b6232602be8b31b504016776e6436a83ea212feb9e':'NEQQLattice.ipynb',
        '77f93f3c0579d350196975719aeb2e2e12684bf2e946933db74a07854c2afa00':'GUANGZHOU_auxiliary.csv',
        '15d33910cc9c35d3f51a8043d6d909d5e045aa2c11ee0e55482c2cb5cdcf571d':'HUBEI_auxiliary.csv',
        'd21795622bbb037772ff39f63c79b689ada5ad21721740219ca38efa7c660715':'EWT point PREDICTIONS.csv',
    })
    forbidden_extensions={'.xlsx','.xls','.xlsb','.npz','.npy','.pt','.pth','.ckpt','.pkl','.pickle','.ipynb'}
    point_keys={'price','prices','close','closing_price','volume','volumes','volume_tons',
                'turnover','turnover_yuan','observed','observed_change','predicted_change','predictions'}
    issues=[];entries=[];csv_info=[];json_count=0;numeric_state_dict_locations=[]
    for name in names:
        p=args.repo/name
        if not p.is_file():continue
        local=p.relative_to(args.study).as_posix();body=p.read_bytes();digest=sha_bytes(body)
        licensed=local.startswith('data/sciencedb_819_v1/')
        entries.append({'path':local,'bytes':len(body),'sha256':digest})
        if digest in forbidden_hashes:
            issues.append({'path':local,'issue':'known_private_source_or_point_file','identity':forbidden_hashes[digest]})
        if p.suffix in forbidden_extensions and not licensed:
            issues.append({'path':local,'issue':'raw_or_checkpoint_file_outside_licensed_cohort'})
        if p.suffix=='.csv':
            reader=csv.DictReader(io.StringIO(body.decode('utf-8-sig')))
            cols=reader.fieldnames or [];rows=sum(1 for _ in reader)
            csv_info.append({'path':local,'rows':rows,'columns':cols,'licensed_daily_data':licensed})
            if not licensed and ('origin_date' in cols or 'origin_row' in cols or 'predicted_change' in cols or
                                 ('date' in cols and bool(set(cols)&point_keys))):
                issues.append({'path':local,'issue':'point_level_csv_outside_licensed_cohort'})
        if p.suffix=='.json' or p.name.endswith('.json.gz'):
            j=json.loads(gzip.decompress(body) if p.name.endswith('.json.gz') else body);json_count+=1
            def walk(x,pointer=''):
                if isinstance(x,dict):
                    for k,v in x.items():
                        sub=pointer+'/'+k
                        if k in {'state_dict','state_dict_including_reference_buffer'}:
                            numeric_state_dict_locations.append({'path':local,'json_pointer':sub,'authorized_own_model_parameters':True})
                            continue
                        if not licensed and k in point_keys and isinstance(v,list) and len(v)>20 and all(isinstance(a,(int,float)) for a in v):
                            issues.append({'path':local,'issue':'long_numeric_point_field','json_pointer':sub,'length':len(v)})
                        walk(v,sub)
                elif isinstance(x,list):
                    for i,v in enumerate(x):walk(v,pointer+'/'+str(i))
            walk(j)
    migration=args.study/'results/information_diagnostic/PRIVATE_PREDICTIONS_MANIFEST.json'
    m=json.loads(migration.read_text());private=Path(m['private_path'])
    assert private.exists() and sha_bytes(private.read_bytes())==m['sha256']
    assert not (args.study/m['original_public_relative_path']).exists()
    assert not old_changes and not old_untracked
    mutations=[]
    for e in entries:
        p=args.study/e['path']
        if not p.is_file() or sha_bytes(p.read_bytes())!=e['sha256']:mutations.append(e['path'])
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'PASS_CURRENT_PROPOSED_PUBLIC_SCOPE' if not issues and not mutations else 'REVIEW_REQUIRED',
        'scope_timing':'Snapshot after EWT migration, not an upload or authorization to publish unrelated files',
        'study':rel,'git_head':head,'first_study_commit':first,'comparison_baseline':baseline,
        'old_studies_changed_since_baseline':old_changes,'old_study_untracked_files':old_untracked,
        'publication_rule':{
            'permitted':'Code, literature/source documentation, aggregate metrics and fit histories, hashes/audits, licensed ScienceDB XLS and CSVs, numeric candidate/structural-control parameter representations.',
            'root_scope_clarification':'Project lead explicitly included our own trained candidate/control state_dict numeric JSONs for mechanism/forward verification; these are not upstream author price/volume tables.',
            'excluded':'New author datasets without declared redistribution licenses, their per-date quantity exports, point observations/forecast paths, and binary torch checkpoint files.',
            'licensed_exception':'ScienceDB DOI10.11922/sciencedb.819 under current official CC BY 4.0 metadata.'},
        'checked_git_visible_files':len(entries),'json_documents_including_gzip':json_count,
        'known_private_source_hashes_checked':len(forbidden_hashes),
        'csv_profiles':csv_info,'authorized_numeric_state_dict_occurrences':len(numeric_state_dict_locations),
        'authorized_numeric_state_dict_locations':numeric_state_dict_locations,
        'ewt_migration_receipt_sha256':sha_bytes(migration.read_bytes()),'ewt_private_copy_hash_verified':True,
        'ewt_public_point_file_absent':True,'issues':issues,'files_changed_during_check':mutations,
        'file_snapshot':entries,'audit_script_sha256':sha_bytes(Path(__file__).read_bytes()),
        'limitations':['Ignored runtime caches are outside Git-visible public candidates.',
                       'Known-hash, filename/extension, CSV-schema and JSON-field review is not a mathematical proof that arbitrary aggregates contain no information.',
                       'Later additions require a new snapshot or equivalent final package review.'],
        'market_fits':0,'source_or_result_edits_by_this_script':0}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'files':len(entries),'issues':len(issues),
                     'own_numeric_state_dicts_allowed':len(numeric_state_dict_locations),
                     'old_study_changes':len(old_changes),'out':str(args.out)}))
    if issues or mutations:raise SystemExit(1)


if __name__=='__main__':main()
