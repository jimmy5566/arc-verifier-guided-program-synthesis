"""CPU postprocessor for frozen raw no-update NLL rows."""
import argparse, json
from collections import defaultdict
from pathlib import Path

WEAK=("connected components","inside/contains","difference","width","orientation")

def sign(value): return -1 if value < 0 else (1 if value > 0 else 0)
def summarize(rows, mode):
    acc=defaultdict(lambda:[0.0,0])
    for row in rows:
        if row['mode_id']==mode and row['family'] in WEAK:
            key=(row['checkpoint_id'],row['family'],row['episode_id'])
            acc[key][0]+=row['row_nll_sum'];acc[key][1]+=row['supervised_token_count']
    by=defaultdict(lambda:[0.0,0])
    for (ckpt,family,_), (loss,count) in acc.items():
        by[(ckpt,family)][0]+=loss;by[(ckpt,family)][1]+=count
    return {f'{a}|{b}':loss/count for (a,b),(loss,count) in by.items()}
def classify(deltas, dev):
    if dev is None:
        return 'INCONCLUSIVE_MISSING_FAMILY_DEV_EVIDENCE'
    improved=[f for f in WEAK if deltas.get(f,0)<0]
    if len(improved)<3:return 'H1_SUPPORTED_CANDIDATE'
    if all(dev.get(f,0)<=0 for f in improved):return 'H3_SUPPORTED_CANDIDATE'
    return 'INCONCLUSIVE'
def resolve_decision(full_deltas, cross_b32_deltas, cross_b1_deltas, dev):
    primary=classify(full_deltas,dev); cross=classify(cross_b1_deltas,dev)
    stable=all(sign(cross_b32_deltas.get(f,0))==sign(cross_b1_deltas.get(f,0)) for f in WEAK)
    return 'NUMERICALLY_INCONCLUSIVE' if not stable or primary!=cross else primary
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--dev',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    rows=[json.loads(x) for x in a.raw.read_text().splitlines() if x.strip()];dev=json.loads(a.dev.read_text())
    b32=summarize(rows,'PRIMARY_B32');cb32=summarize(rows,'CROSSCHECK_B32');cb1=summarize(rows,'CROSSCHECK_B1');v7='CAPABILITY_REPAIR_BASELINE_V1_V7';out={}
    for candidate in sorted({r['checkpoint_id'] for r in rows}-{v7}):
        full={f:b32.get(candidate+'|'+f,0)-b32.get(v7+'|'+f,0) for f in WEAK};cross32={f:cb32.get(candidate+'|'+f,0)-cb32.get(v7+'|'+f,0) for f in WEAK};cross1={f:cb1.get(candidate+'|'+f,0)-cb1.get(v7+'|'+f,0) for f in WEAK}
        candidate_dev=dev.get('family_counts',{}).get(candidate,{}).get('family_counts')
        v7_dev=dev.get('family_counts',{}).get(v7,{}).get('family_counts')
        family_dev=None if candidate_dev is None or v7_dev is None else {f:candidate_dev[f]['exact']-v7_dev[f]['exact'] for f in WEAK}
        out[candidate]={'full_b32_delta':full,'cross_b32_delta':cross32,'cross_b1_delta':cross1,'family_dev_exact_delta':family_dev,'decision':resolve_decision(full,cross32,cross1,family_dev)}
    a.out.write_text(json.dumps({'schema_version':1,'status':'COMPLETE_CPU_POSTPROCESS','results':out,'raw_rows':len(rows)},sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
