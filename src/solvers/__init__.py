from .base import Solver, Candidate
from .baselines import IdentitySolver, SimpleTransformSolver, ObjectRuleSolver
from .object_rules import ObjectRule, select_object
from .library_v0 import GlobalTransformSolver, RecolorSolver, FixedCropSolver, ForegroundBBoxCropSolver, TranslationSolver
from .library_v1_object import ObjectSelectionSolver, ObjectTransformSolver, ObjectCopyMoveSolver
from .library_v2_relation import RelationalSelectionSolver, ObjectAlignmentSolver, TwoObjectCompositionSolver
from .capability_sanity import EnclosureFillSolver, PathSerializationSolver, CountGenerationSolver
