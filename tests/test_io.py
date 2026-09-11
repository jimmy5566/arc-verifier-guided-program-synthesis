import json
from arc.io import load_dataset

def test_loader_converts_to_numpy_and_aligns_solutions(tmp_path):
    challenges={"task":{"train":[{"input":[[0]],"output":[[1]]}],"test":[{"input":[[2]]}]}}
    solutions={"task":[[[3]]]}
    cp=tmp_path/"challenges.json"; sp=tmp_path/"solutions.json"; cp.write_text(json.dumps(challenges)); sp.write_text(json.dumps(solutions))
    task=load_dataset(cp,sp)["task"]
    assert task.train[0].input.values.shape==(1,1) and task.test[0].output.to_list()==[[3]]
