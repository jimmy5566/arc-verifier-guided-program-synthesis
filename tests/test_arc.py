import numpy as np
import pytest
from arc.task import ARCGrid, ARCExample, ARCTask
from representations.objects import extract_objects
from transforms import Rotate, TransformPipeline, Recolor

def test_rejects_invalid_grid():
    with pytest.raises(ValueError): ARCGrid([[10]])
    with pytest.raises(ValueError): ARCGrid([1,2])
def test_components_and_hole():
    grid=np.array([[2,2,2],[2,0,2],[2,2,2]])
    obj=extract_objects(grid,background=0)[0]; assert obj.area==8 and obj.holes==1 and obj.width==3
def test_transform_composition():
    out=TransformPipeline((Rotate(90),Recolor(1,3))).apply(np.array([[1,0]])); assert out.tolist()==[[0],[3]]
def test_task_requires_labeled_train():
    with pytest.raises(ValueError): ARCTask("x",(ARCExample(ARCGrid([[0]])),),(ARCExample(ARCGrid([[0]])),))
