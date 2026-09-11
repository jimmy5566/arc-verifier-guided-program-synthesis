import numpy as np
from arc.task import ARCGrid, ARCExample, ARCTask
from transforms import AntiTranspose
from solvers.library_v0 import GlobalTransformSolver, RecolorSolver, FixedCropSolver, ForegroundBBoxCropSolver, TranslationSolver

def task(train, test_input=[[0]]):
    return ARCTask('synthetic',tuple(ARCExample(ARCGrid(i),ARCGrid(o)) for i,o in train),(ARCExample(ARCGrid(test_input)),))

def test_anti_transpose():
    assert AntiTranspose().apply(np.array([[1,2,3],[4,5,6]])).tolist()==[[6,3],[5,2],[4,1]]
def test_global_transform_requires_all_pairs():
    solver=GlobalTransformSolver().fit(task([([[1,2]],[[2],[1]]),([[3,4]],[[4],[3]])]))
    assert solver.candidates and solver.predict(np.array([[5,6]]))[0].tolist()==[[6],[5]]
def test_consistent_recolor_mapping():
    solver=RecolorSolver().fit(task([([[1,0]],[[2,0]]),([[0,1]],[[0,2]])]))
    assert solver.candidates and solver.predict(np.array([[1,0]]))[0].tolist()==[[2,0]]
def test_fixed_crop_rejects_nonunique_and_applies_shared_location():
    good=FixedCropSolver().fit(task([([[0,1,2],[3,4,5]],[[1,2]]),([[9,1,2],[8,7,6]],[[1,2]])]))
    assert good.candidates and good.predict(np.array([[4,1,2]]))[0].tolist()==[[1,2]]
    bad=FixedCropSolver().fit(task([([[1,1]],[[1]])])); assert not bad.candidates
def test_foreground_bbox_crop():
    solver=ForegroundBBoxCropSolver().fit(task([([[0,2,0],[0,2,0]],[[2],[2]])]))
    assert solver.candidates
def test_translation_requires_fixed_in_bounds_shift():
    solver=TranslationSolver().fit(task([([[0,2,0],[0,0,0]],[[0,0,0],[0,2,0]]),([[0,3,0],[0,0,0]],[[0,0,0],[0,3,0]])]))
    assert solver.candidates and solver.predict(np.array([[0,4,0],[0,0,0]]))[0].tolist()==[[0,0,0],[0,4,0]]
