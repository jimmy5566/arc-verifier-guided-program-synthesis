import pandas as pd
from evaluation.splits import make_task_splits
def test_splits_are_reproducible_and_task_level(tmp_path):
    source=tmp_path/"metadata.csv"; output=tmp_path/"splits.csv"
    pd.DataFrame({"task_id":["a","b","c"],"shape_change":[True,False,True],"multi_object":[True,False,False],"symmetry":[False,True,False]}).to_csv(source,index=False)
    a=make_task_splits(source,output); b=make_task_splits(source,output)
    assert (a.split==b.split).all() and set(a.split)<= {"development","held_out","challenge_like"}
