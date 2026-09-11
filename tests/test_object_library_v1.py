import numpy as np
from arc.task import ARCGrid, ARCExample, ARCTask
from representations.objects import extract_objects
from representations.selectors import SelectLargest, SelectSmallest, SelectUniqueByArea, SelectUniqueByShape, SelectUniqueByColor
from representations.object_ops import crop_object, recolor_object, remove_object, render_object
from solvers.library_v1_object import ObjectSelectionSolver, ObjectTransformSolver

def objects(grid, connectivity=4): return extract_objects(np.array(grid),connectivity,background=0)
def make_task(train): return ARCTask('synthetic',tuple(ARCExample(ARCGrid(a),ARCGrid(b)) for a,b in train),(ARCExample(ARCGrid([[0]])),))

def test_component_geometry_connectivity_and_border():
    four=objects([[2,0],[0,2]],4); eight=objects([[2,0],[0,2]],8)
    assert len(four)==2 and len(eight)==1
    obj=objects([[0,2,2],[0,2,0]])[0]
    assert obj.area==3 and obj.bbox==(0,1,1,2) and obj.top==0 and obj.bottom==1 and obj.left==1 and obj.right==2 and obj.touches_border
    assert obj.shape_mask.tolist()==[[True,True],[True,False]] and obj.centroid==(1/3,4/3)
def test_selection_primitives():
    group=objects([[2,2,0,3],[0,0,0,0],[4,0,5,0]])
    assert SelectLargest.select(group)[0].color==2 and {o.color for o in SelectSmallest.select(group)}=={3,4,5}
    assert {o.color for o in SelectUniqueByArea.select(group)}=={2}
    assert SelectUniqueByColor.select(group)==group and SelectUniqueByShape.select(group)[0].color==2
def test_object_rendering_crop_recolor_remove_copy_and_boundary():
    grid=np.array([[0,2,2],[0,0,0],[0,0,0]]); obj=objects(grid)[0]
    assert crop_object(obj,grid).tolist()==[[2,2]]
    assert recolor_object(grid,obj,3).tolist()==[[0,3,3],[0,0,0],[0,0,0]]
    assert remove_object(grid,obj).tolist()==[[0,0,0],[0,0,0],[0,0,0]]
    assert render_object(grid,obj,1,0,copy=True).tolist()==[[0,2,2],[0,2,2],[0,0,0]]
    assert render_object(grid,obj,3,0) is None
def test_object_selection_solver_requires_all_train_pairs():
    solver=ObjectSelectionSolver().fit(make_task([([[0,2,2],[0,0,0]],[[2,2]]),([[0,3,3],[0,0,0]],[[3,3]])]))
    assert solver.candidates and solver.predict(np.array([[0,4,4],[0,0,0]]))[0].tolist()==[[4,4]]
    rejected=ObjectSelectionSolver().fit(make_task([([[0,2],[0,0]],[[2]]),([[0,3],[0,0]],[[0]])]))
    assert not rejected.candidates
def test_object_transform_recolor_train_consistency():
    solver=ObjectTransformSolver().fit(make_task([([[0,2,2],[0,0,0]],[[0,3,3],[0,0,0]]),([[0,2,2],[0,0,0]],[[0,3,3],[0,0,0]])]))
    assert solver.candidates and solver.predict(np.array([[0,2,2],[0,0,0]]))[0].tolist()==[[0,3,3],[0,0,0]]
