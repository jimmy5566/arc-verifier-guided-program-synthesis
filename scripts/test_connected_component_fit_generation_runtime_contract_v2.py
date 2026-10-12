import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.connected_component_fit_generation_runtime_contract import aggregate_record_metrics, require_inventory_partition, token_metric, teacher_forced_layout, pack_teacher_forced_batch

def main():
    a=token_metric([0,4,1],1); b=token_metric([0,1,4],2)
    metrics=aggregate_record_metrics({"r1":[a,b],"r2":[a]})
    assert metrics["tokens"]==3 and metrics["records"]==2 and "sequence_normalized_nll" in metrics
    rows=[]
    for i in range(216):
        rows.append({"checkpoint":["V7","CONTROL","TREATMENT"][i%3],"cohort_record_id":f"r{i}","split":"TRAIN_FIT" if i>=168 else "OOD","disposition":"GENERATE_NEW_ONCE_AFTER_FRESH_AUTHORIZATION" if i>=168 else "REUSE_IMMUTABLE_RUN_002_RAW"})
    reused,fresh=require_inventory_partition({"inventory":rows})
    assert len(reused)==168 and len(fresh)==48
    try:
        require_inventory_partition({"inventory":rows[:-1]})
    except RuntimeError as e:
        assert str(e)=="INVENTORY_ROW_COUNT_INVALID"
    else: raise AssertionError("missing failure")
    packed=pack_teacher_forced_batch({'short':teacher_forced_layout([1],[2]),'long':teacher_forced_layout([1,2,3],[4,5])})
    assert len(packed['short']['input_ids'])==len(packed['long']['input_ids'])==5
    assert packed['short']['attention_mask']==[1,1,0,0,0]
    print("PASS_RUNTIME_CONTRACT_V2")
if __name__=="__main__": main()
