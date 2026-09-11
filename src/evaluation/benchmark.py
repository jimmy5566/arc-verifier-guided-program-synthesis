"""Task-level solver benchmark; output is designed for persisted experiment records."""
from __future__ import annotations
from time import perf_counter
from typing import Callable
from arc.task import ARCTask
from verification.verifier import verify_prediction

def benchmark_solver(tasks: dict[str, ARCTask], solver_factory: Callable[[], object], candidate_budget: int=2) -> dict:
    start=perf_counter(); solved=[]; second_additional=[]; unsolved=[]; candidate_counts=[]
    for task_id, task in tasks.items():
        solver=solver_factory(); solver.fit(task); candidate_counts.append(len(solver.candidates)); exact_by_rank=[]
        # If test labels are public, report independent test performance; otherwise report train-consistency coverage.
        labelled=[x for x in task.test if x.output is not None]
        if labelled:
            for rank in range(candidate_budget):
                exact_by_rank.append(all(len(solver.predict(ex.input.values, top_k=candidate_budget)) > rank and verify_prediction(ex.output.values, solver.predict(ex.input.values, top_k=candidate_budget)[rank]).exact_match for ex in labelled))
        else:
            exact_by_rank=[candidate.exact_match for candidate in solver.candidates[:candidate_budget]]
        if exact_by_rank and exact_by_rank[0]: solved.append(task_id)
        elif any(exact_by_rank): second_additional.append(task_id)
        else: unsolved.append(task_id)
    total=len(tasks); return {"total_tasks":total,"exact_solved":len(solved),"exact_accuracy":len(solved)/total if total else 0.0,"attempt_1_accuracy":len(solved)/total if total else 0.0,"attempt_2_additional_solves":len(second_additional),"runtime_seconds":perf_counter()-start,"average_candidates_per_task":sum(candidate_counts)/len(candidate_counts) if candidate_counts else 0.0,"solved_task_ids":solved,"second_attempt_task_ids":second_additional,"unsolved_task_ids":unsolved,"uniquely_solved_task_ids":solved}
