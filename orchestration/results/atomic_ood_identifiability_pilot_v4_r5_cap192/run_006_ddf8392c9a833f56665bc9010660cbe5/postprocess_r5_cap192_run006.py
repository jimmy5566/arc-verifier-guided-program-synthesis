#!/usr/bin/env python3
"""CPU-only post-freeze scorer for the frozen RUN_006 target-blind evidence."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()
def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x]
def unit(episode_id: str) -> str: return episode_id.split('-')[2]
def cell(episode_id: str) -> str: return episode_id.split('-')[1]
def aggregate(part, targets):
    def exact(x): return bool(x.get('parser_valid')) and x.get('parsed_grid')==targets[x['episode_id']]
    units={}; cells={}
    for x in part:
        units.setdefault(unit(x['episode_id']),[]).append(exact(x))
        cells.setdefault(cell(x['episode_id']),[]).append(exact(x))
    return {
      'rows':len(part),'parser_valid_rows':sum(bool(x.get('parser_valid')) for x in part),
      'exact_grid_rows':sum(exact(x) for x in part),'exact_grid_rate':sum(exact(x) for x in part)/len(part),
      'canonical_unit_macro_exact_grid_rate':sum(sum(v)/len(v) for v in units.values())/len(units),
      'canonical_units':{u:{'rows':len(v),'exact_grid_rows':sum(v),'rate':sum(v)/len(v)} for u,v in sorted(units.items())},
      'six_cells':{c:{'rows':len(v),'exact_grid_rows':sum(v),'rate':sum(v)/len(v)} for c,v in sorted(cells.items())},
    }
def main(a):
    raw=Path(a.raw_dir); package=json.loads(Path(a.package).read_text(encoding='utf-8'))
    freeze=json.loads(Path(a.raw_freeze_manifest).read_text(encoding='utf-8'))
    if freeze.get('freeze_condition')!='all raw evidence and terminal receipt byte-hash verified before CPU scorer opens sealed sidecar': raise SystemExit('RAW_FREEZE_CONDITION')
    receipt_path=raw/'TERMINAL_RECEIPT.json'
    if sha(receipt_path)!=freeze.get('terminal_receipt_sha256'): raise SystemExit('FROZEN_RECEIPT_SHA')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('status')!='COMPLETE_NO_UPDATE_TARGET_BLIND': raise SystemExit('TERMINAL_STATUS')
    if any(receipt.get(k) not in (0,False) for k in ('optimizer_steps','parameter_updates','backward_calls','target_sidecar_accessed','final_audit_opened')): raise SystemExit('NO_UPDATE_BOUNDARY')
    expected={x['arm_id']:(x['path'],x['sha256']) for x in receipt['raw_evidence']}
    if set(expected)!={'V7','E04_E_CONTROL','E04_E_TREATMENT'}: raise SystemExit('ARM_SET')
    for arm,(name,expected_sha) in expected.items():
        if freeze.get('raw_target_blind_sha256',{}).get(name)!=expected_sha or sha(raw/name)!=expected_sha: raise SystemExit('RAW_FREEZE_SHA:'+arm)
    # All target-independent frozen-byte checks have passed.  Only now may the
    # sealed sidecar be opened for the already-approved CPU exact-grid scorer.
    target_path=Path(a.target_sidecar)
    expected_target=package['inputs']['experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/SEALED_TARGET_SIDECAR.jsonl']
    if sha(target_path)!=expected_target: raise SystemExit('SEALED_TARGET_SIDECAR_IDENTITY')
    targets={x['episode_id']:x['target'] for x in rows(target_path)}
    fixed=set(package['batching']['fixed_b1_subset']['episode_ids']); arms={}; episode_maps=[]; b1_checks={}
    for arm,(name,expected_sha) in sorted(expected.items()):
        data=rows(raw/name)
        if len(data)!=204 or {x['arm_id'] for x in data}!={arm}: raise SystemExit('RAW_CARDINALITY:'+arm)
        primary=[x for x in data if x['phase']=='PRIMARY']; b1=[x for x in data if x['phase']=='B1_SENSITIVITY']
        if len(primary)!=192 or len(b1)!=12 or {x['episode_id'] for x in b1}!=fixed: raise SystemExit('PHASE_OR_B1:'+arm)
        if any(x['episode_id'] not in targets for x in data): raise SystemExit('TARGET_MAPPING:'+arm)
        if any(x['effective_batch_size']!=1 for x in b1): raise SystemExit('B1_PHYSICAL:'+arm)
        primary_by_id={x['episode_id']:x for x in primary}
        if len(primary_by_id)!=192: raise SystemExit('PRIMARY_EPISODE_DUPLICATE:'+arm)
        def exact(x): return bool(x.get('parser_valid')) and x.get('parsed_grid')==targets[x['episode_id']]
        checks=[]
        for b in b1:
            p=primary_by_id[b['episode_id']]
            checks.append({'episode_id':b['episode_id'],'parser_valid_match':bool(b.get('parser_valid'))==bool(p.get('parser_valid')),'parsed_grid_match':b.get('parsed_grid')==p.get('parsed_grid'),'exact_grid_match':exact(b)==exact(p)})
        b1_checks[arm]={'rows':len(checks),'parser_valid_matches':sum(x['parser_valid_match'] for x in checks),'parsed_grid_matches':sum(x['parsed_grid_match'] for x in checks),'exact_grid_matches':sum(x['exact_grid_match'] for x in checks),'all_rows_agree':all(all(v for k,v in x.items() if k!='episode_id') for x in checks)}
        arms[arm]={'primary':aggregate(primary,targets),'b1':aggregate(b1,targets)}
        episode_maps.append({x['episode_id'] for x in data})
    if not all(m==episode_maps[0] for m in episode_maps[1:]) or episode_maps[0]!=set(targets): raise SystemExit('CROSS_ARM_EPISODE_MAPPING')
    unit_names=sorted({unit(e) for e in targets}); effects={}
    for arm in ('E04_E_CONTROL','E04_E_TREATMENT'):
        values=[arms[arm]['primary']['canonical_units'][u]['rate']-arms['V7']['primary']['canonical_units'][u]['rate'] for u in unit_names]
        rng=random.Random(20261011); boot=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(10000))
        effects[arm+'_minus_V7']={'estimand':'equal_weight_canonical_unit_exact_grid_rate_delta','point_estimate':sum(values)/len(values),'bootstrap_95_ci':[boot[249],boot[9749]],'units':len(values)}
    b1_ok=all(x['all_rows_agree'] for x in b1_checks.values())
    result={'protocol_id':receipt['protocol_id'],'run_id':freeze['run_id'],'status':'COMPLETE_TARGET_BLIND_CPU_SCORED' if b1_ok else 'INVALID_B1_BATCH_SENSITIVITY_DISAGREEMENT','scientific_scope':'DEVELOPMENT_EVIDENCE_ONLY_NOT_INDEPENDENT_ARC_GENERALIZATION_CONFIRMATION','receipt_sha256':sha(receipt_path),'raw_evidence_sha256':{arm:expected[arm][1] for arm in sorted(expected)},'raw_freeze_manifest_sha256':sha(a.raw_freeze_manifest),'sealed_target_sidecar_sha256':expected_target,'target_content_released':False,'arms':arms,'paired_primary_effects':effects,'b1_conclusion_stability':{'exact_grid_rate_range':[min(v['b1']['exact_grid_rate'] for v in arms.values()),max(v['b1']['exact_grid_rate'] for v in arms.values())],'parser_valid_rate_range':[min(v['b1']['parser_valid_rows']/12 for v in arms.values()),max(v['b1']['parser_valid_rows']/12 for v in arms.values())],'row_level_agreement':b1_checks,'status':'PASS_ROW_LEVEL_PARSER_GRID_AND_EXACT_AGREEMENT' if b1_ok else 'FAIL_ROW_LEVEL_BATCH_SENSITIVITY_DISAGREEMENT'},'interpretation':'CPU-scored development evidence only. Exact-grid rates, six-cell summaries, paired canonical-unit contrasts, and B1 agreement are descriptive under the frozen protocol; they do not establish independent ARC generalization.'}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--raw-dir',required=True); p.add_argument('--raw-freeze-manifest',required=True); p.add_argument('--package',required=True); p.add_argument('--target-sidecar',required=True); p.add_argument('--output',required=True); main(p.parse_args())
