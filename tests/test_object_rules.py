import numpy as np
from arc.task import ARCGrid, ARCExample, ARCTask
from representations.objects import extract_objects
from representations.correspondence import compare_objects
from solvers.object_rules import ObjectRule, select_object
from solvers.baselines import ObjectRuleSolver

def _obj(grid): return extract_objects(np.array(grid),background=0)[0]
def test_correspondence_detects_translation_and_recolor():
    source=_obj([[2,2],[0,0]])
    target=_obj([[0,0,0],[0,3,0],[0,3,0]])
    match=compare_objects(source,target)
    assert match.same_shape and not match.same_color and match.translation==(1,1)
def test_correspondence_detects_rotation_and_reflection():
    source=_obj([[2,0],[2,2]])
    target=_obj([[0,3],[3,3]])
    assert compare_objects(source,target).transform=='rotate90'
def test_selection_criteria_and_object_operations():
    grid=np.array([[2,2,0],[0,0,3],[0,0,0]])
    assert select_object(grid,'largest').color==2
    assert select_object(grid,'smallest').color==3
    assert ObjectRule('largest','crop').apply(grid).tolist()==[[2,2]]
    assert ObjectRule('smallest','remove').apply(grid).tolist()==[[2,2,0],[0,0,0],[0,0,0]]
    assert ObjectRule('largest','fill_bbox',color=4).apply(grid).tolist()==[[4,4,0],[0,0,3],[0,0,0]]
def test_object_candidate_ranking_prefers_exact_simple_rule():
    task=ARCTask('synthetic',(ARCExample(ARCGrid([[2,2,0],[0,0,3]]),ARCGrid([[2,2]])),),(ARCExample(ARCGrid([[4,4,0]])),))
    candidates=ObjectRuleSolver().fit(task).candidates
    assert candidates and candidates[0].exact_match and candidates[0].metadata['operation']=='crop'
