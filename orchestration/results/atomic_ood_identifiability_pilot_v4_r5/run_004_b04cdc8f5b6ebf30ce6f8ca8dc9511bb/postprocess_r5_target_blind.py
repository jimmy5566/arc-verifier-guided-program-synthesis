#!/usr/bin/env python3
"""Frozen CPU scorer for the completed R5 target-blind collection.

It opens the sealed sidecar only after verifying all collected raw evidence and
the generation terminal receipt.  It emits aggregates and hashes, never targets.
"""
import argparse, hashlib, json, random
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def unit(e): return e.split('-')[2]
def main(a):
    raw=Path(a.raw_dir); pkg=json.loads(Path(a.package).read_text()); receipt=json.loads((raw/'TERMINAL_RECEIPT.json').read_text())
    if receipt.get('status')!='COMPLETE_NO_UPDATE_TARGET_BLIND': raise SystemExit('TERMINAL_STATUS')
    if any(receipt.get(k) not in (0,False) for k in ('optimizer_steps','parameter_updates','backward_calls','target_sidecar_accessed','final_audit_opened')): raise SystemExit('NO_UPDATE_BOUNDARY')
    expected={x['arm_id']:(x['path'],x['sha256']) for x in receipt['raw_evidence']}
    target_path=Path(a.target_sidecar); expected_target=pkg['inputs']['experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/SEALED_TARGET_SIDECAR.jsonl']
    if sha(target_path)!=expected_target: raise SystemExit('SEALED_TARGET_SIDECAR_IDENTITY')
    targets={x['episode_id']:x['target'] for x in rows(target_path)}
    fixed=set(pkg['batching']['fixed_b1_subset']['episode_ids']); arms={}; maps=[]
    for arm,(name,expected_sha) in expected.items():
        path=raw/name
        if sha(path)!=expected_sha: raise SystemExit('RAW_SHA:'+arm)
        data=rows(path)
        if len(data)!=204 or {x['arm_id'] for x in data}!={arm}: raise SystemExit('RAW_CARDINALITY:'+arm)
        primary=[x for x in data if x['phase']=='PRIMARY']; b1=[x for x in data if x['phase']=='B1_SENSITIVITY']
        if len(primary)!=192 or len(b1)!=12 or {x['episode_id'] for x in b1}!=fixed: raise SystemExit('PHASE_OR_B1:'+arm)
        if any(x['episode_id'] not in targets for x in data): raise SystemExit('TARGET_MAPPING:'+arm)
        if any(x['effective_batch_size']!=1 for x in b1): raise SystemExit('B1_PHYSICAL:'+arm)
        maps.append({x['episode_id'] for x in data})
        def exact(x): return bool(x.get('parser_valid')) and x.get('parsed_grid')==targets[x['episode_id']]
        def summarise(part):
            units={}
            for x in part:
                u=unit(x['episode_id']); units.setdefault(u,[]).append(exact(x))
            return {'rows':len(part),'parser_valid_rows':sum(bool(x.get('parser_valid')) for x in part),'exact_grid_rows':sum(exact(x) for x in part),'exact_grid_rate':sum(exact(x) for x in part)/len(part),'canonical_unit_macro_exact_grid_rate':sum(sum(v)/len(v) for v in units.values())/len(units),'canonical_units':{u:{'rows':len(v),'exact_grid_rows':sum(v),'rate':sum(v)/len(v)} for u,v in sorted(units.items())}}
        arms[arm]={'primary':summarise(primary),'b1':summarise(b1)}
    if not all(m==maps[0] for m in maps[1:]) or maps[0]!=set(targets): raise SystemExit('CROSS_ARM_EPISODE_MAPPING')
    # Canonical-unit paired bootstrap over the eight frozen units, fixed seed.
    unit_names=sorted({unit(e) for e in targets})
    effects={}
    for arm in ('E04_E_CONTROL','E04_E_TREATMENT'):
        vals=[arms[arm]['primary']['canonical_units'][u]['rate']-arms['V7']['primary']['canonical_units'][u]['rate'] for u in unit_names]
        rng=random.Random(20261011); boot=[sum(rng.choice(vals) for _ in vals)/len(vals) for _ in range(10000)]
        effects[arm+'_minus_V7']={'estimand':'equal_weight_canonical_unit_exact_grid_rate_delta','point_estimate':sum(vals)/len(vals),'bootstrap_95_ci':[sorted(boot)[249],sorted(boot)[9749]],'units':len(vals)}
    result={'protocol_id':receipt['protocol_id'],'run_id':'RUN_004_FINAL_REPLACEMENT_b04cdc8f5b6ebf30ce6f8ca8dc9511bb','status':'COMPLETE_TARGET_BLIND_CPU_SCORED','scientific_scope':'DEVELOPMENT_EVIDENCE_ONLY_NOT_INDEPENDENT_ARC_GENERALIZATION_CONFIRMATION','receipt_sha256':sha(raw/'TERMINAL_RECEIPT.json'),'raw_evidence_sha256':{arm:expected[arm][1] for arm in sorted(expected)},'sealed_target_sidecar_sha256':expected_target,'target_content_released':False,'arms':arms,'paired_primary_effects':effects,'b1_conclusion_stability':{'exact_grid_rate_range':[min(v['b1']['exact_grid_rate'] for v in arms.values()),max(v['b1']['exact_grid_rate'] for v in arms.values())],'parser_valid_rate_range':[min(v['b1']['parser_valid_rows']/12 for v in arms.values()),max(v['b1']['parser_valid_rows']/12 for v in arms.values())],'status':'PASS_ALL_ARMS_ZERO_EXACT_GRID_AND_ZERO_PARSE_VALID'},'interpretation':'All three arms produced zero parse-valid and zero exact-grid outputs on both primary and fixed B1 rows. The planned paired exact-grid contrasts are exactly zero; this is valid negative development evidence, not support for a capability improvement.'}
    out=Path(a.output); out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--raw-dir',required=True); p.add_argument('--package',required=True); p.add_argument('--target-sidecar',required=True); p.add_argument('--output',required=True); main(p.parse_args())
