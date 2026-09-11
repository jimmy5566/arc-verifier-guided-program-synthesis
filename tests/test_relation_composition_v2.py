import numpy as np
from arc.task import ARCGrid, ARCExample, ARCTask
from primitives.registry import REGISTRY, validate_registry
from primitives.program import Program, ProgramExecutor, Step, align_object, compose_masks, copy_relative
from representations.objects import extract_objects
from representations import relations_v1 as rel

def objs(grid): return extract_objects(np.array(grid),background=0)
def test_registry_and_structured_program_validation():
    assert not validate_registry() and len(REGISTRY)==len(set(REGISTRY))
    p=Program((Step('SEL_LARGEST_V1',{'selector':'largest'}),Step('OBJ_CROP_V1',{})),'test',complexity_cost=2)
    assert Program.from_dict(p.to_dict())==p and ProgramExecutor().execute(p,np.array([[0,2,2],[0,0,0]])).tolist()==[[2,2]]
    assert ProgramExecutor().execute(Program((Step('UNKNOWN_V1',{}),),'test'),np.array([[0]])) is None
    assert ProgramExecutor().execute(Program((Step('SEL_LARGEST_V1',{'selector':'largest'}),Step('OBJ_RECOLOR_V1',{'color':12})),'test'),np.array([[0,2,2]])) is None
def test_relations_touching_alignment_distance_and_tie():
    a,b=objs([[2,0,3],[0,0,0]])
    assert rel.left_of(a,b) and rel.min_manhattan(a,b)==2 and rel.min_chebyshev(a,b)==2 and not rel.touching_4(a,b)
    diagonal=objs([[2,0],[0,3]]); assert not rel.touching_4(*diagonal)
    touch=objs([[2,3]]); assert rel.touching_4(*touch)
    center=objs([[2,0,3,0,4]]); assert rel.unique_nearest(center[1],center) is None
def test_alignment_composition_relative_copy_and_boundary():
    a,b=objs([[2,0,3],[0,0,0]])
    aligned=align_object(np.array([[2,0,3],[0,0,0]]),a,b,'top'); assert aligned is not None
    assert compose_masks(a,b,np.array([[2,0,3],[0,0,0]]),'union') is None  # explicit same-colour policy
    same=objs([[2,0,2],[0,0,0]]); assert compose_masks(same[0],same[1],np.array([[2,0,2],[0,0,0]]),'union').tolist()==[[2]]
    assert copy_relative(np.array([[2,0,3],[0,0,0]]),a,b,'column','right',0) is None
def test_relational_selection_train_consistency():
    from solvers.library_v2_relation import RelationalSelectionSolver
    task=ARCTask('r',(ARCExample(ARCGrid([[2,3,0],[0,0,4]]),ARCGrid([[4]])),),(ARCExample(ARCGrid([[5,6,0],[0,0,7]])),))
    solver=RelationalSelectionSolver().fit(task); assert solver.candidates
