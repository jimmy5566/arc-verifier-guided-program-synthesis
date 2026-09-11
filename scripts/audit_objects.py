"""Invariant audit of representation on a deterministic real-data sample (no solutions read)."""
from __future__ import annotations
import csv, random
from pathlib import Path
from arc.io import discover_dataset_files, load_dataset
from representations.objects import extract_objects
parser=__import__('argparse').ArgumentParser(); parser.add_argument('--data-dir',default='data/raw'); parser.add_argument('--output',default='reports/tables/object_representation_audit.csv'); args=parser.parse_args()
files=discover_dataset_files(args.data_dir); tasks=load_dataset(files['training_challenges']); ids=random.Random(20260910).sample(sorted(tasks),10); rows=[]
for task_id in ids:
    grid=tasks[task_id].train[0].input.values; bg=int(__import__('numpy').bincount(grid.ravel()).argmax()); four=extract_objects(grid,4,background=bg); eight=extract_objects(grid,8,background=bg)
    assert len(eight)<=len(four)
    for connectivity,objects in ((4,four),(8,eight)):
        for index,obj in enumerate(objects):
            assert obj.area==len(obj.coordinates) and obj.height==obj.shape_mask.shape[0] and obj.width==obj.shape_mask.shape[1]
            rows.append({'task_id':task_id,'connectivity':connectivity,'object_index':index,'color':obj.color,'area':obj.area,'bbox':obj.bbox,'centroid':obj.centroid,'border_touching':obj.touches_border,'holes':obj.holes})
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
with Path(args.output).open('w',newline='',encoding='utf-8') as handle: csv.DictWriter(handle,fieldnames=rows[0]).writeheader(); csv.DictWriter(handle,fieldnames=rows[0]).writerows(rows)
print(f'audited {len(ids)} tasks / {len(rows)} objects; all invariants passed')
