"""Shared target-blind logical search core for E1 search-order policies.

This module changes only the order in which already-retained non-EOS
continuations are expanded.  The ReadyCell request contract, cache owner,
prompt tokens, retention rule, EOS handling and physical scheduler all remain
the established Dynamic Ready implementation.
"""
from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
import math
import time
from typing import Any, Literal

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, _node, _prefix_hash, _record_frontier, _retained, _trace
from inference.nvarc_turbodfs_reference import ReferenceTurboDFSCandidate
from inference.search_order import RetainedWorkItem, SearchOrderName, make_search_order_policy
from inference.nvarc_turbodfs_dynamic_ready import (
    CacheOwner,
    ReadyCell,
    ReadyForwardRequest,
    _advance_cell,
    _cache_batch_size,
    _emit_eos_event,
    _eos_checkpoint_region,
    _is_transformers_cache,
    _legacy_cache,
    _new_state,
    _restore_cache_kind,
    cache_geometry,
    clone_legacy_cache,
    start_ready_cell,
)


@dataclass(frozen=True)
class _Frame:
    item: RetainedWorkItem
    token_id: int
    token_logprob: float
    parent_node_id: int
    prefix: tuple[int, ...]
    position: int
    restored: bool
    restore_rank: int | None


def _lcp(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    result = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        result += 1
    return result


def _elapsed_budget(state: dict[str, Any], started_unix: float) -> float:
    return float(state["active_elapsed_seconds"]) if state["active_time_accounting"] else time.time() - started_unix


def _update_owner_after_reply(owner: CacheOwner, outputs: Any) -> None:
    if _is_transformers_cache(owner.cache):
        if outputs.past_key_values is not owner.cache:
            raise RuntimeError("search-order ReadyCell reply replaced its DynamicCache owner")
    else:
        owner.cache = outputs.past_key_values


def _order_generator(
    *, cell_key: str, root_logits: Any, root_position: int, max_new_tokens: int,
    cache_owner: CacheOwner, root_cache_length: int, root_legacy_snapshot: Any | None,
    config: D1TurboDFSConfig, started_unix: float, state: dict[str, Any], root_node_id: int,
    policy_name: SearchOrderName,
) -> Generator[ReadyForwardRequest, Any, list[ReferenceTurboDFSCandidate]]:
    """Explicit retained-work event loop with a single cache owner per cell."""
    import torch

    policy = make_search_order_policy(policy_name, quantum=64)
    frames: dict[int, _Frame] = {}
    suffixes: list[ReferenceTurboDFSCandidate] = []
    current_prefix: tuple[int, ...] = ()
    ordinal = 0
    next_work_item_id = 0
    pending_by_parent: dict[int, int] = {}
    state["search_order"] = {
        "policy": policy_name, "quantum": 64, "retention_changed": False,
        "replayed_tokens": 0, "replay_model_forwards": 0, "useful_model_forwards": 0,
        "yielded_subtrees": 0, "pending_retained_work_at_r1024": 0,
    }
    if policy_name == "LDS_UNIT_DISCREPANCY_V1":
        state["search_order"]["discrepancy_definition"] = "unit_non_top1"
    state["search_order_work_items"] = []

    def record_item(frame: _Frame) -> dict[str, Any]:
        row = {
            "cell_key": cell_key, "work_item_id": frame.item.work_item_id,
            "parent_work_item_id": frame.item.parent_work_item_id,
            "root_branch_id": frame.item.root_branch_id, "prefix_length": len(frame.prefix),
            "cumulative_nll": frame.item.cumulative_nll, "cumulative_regret": frame.item.cumulative_regret,
            "regret_band": frame.item.regret_band, "insertion_order": frame.item.insertion_order,
            "first_scheduled_node": None, "last_scheduled_node": None, "number_of_expansions": 0,
            "number_of_yields": 0, "wait_nodes_before_first_expansion": None, "max_wait_nodes": 0,
            "status": "pending",
        }
        if policy_name == "LDS_UNIT_DISCREPANCY_V1":
            row["local_token_rank"] = frame.item.local_token_rank
            row["discrepancy_count"] = frame.item.discrepancy_count
        state["search_order_work_items"].append(row)
        return row

    work_rows: dict[int, dict[str, Any]] = {}

    def add_successors(children: list[_Frame], *, is_root: bool) -> None:
        if not children:
            return
        if is_root:
            if policy_name == "LDS_UNIT_DISCREPANCY_V1":
                policy.push_successors(tuple(child.item for child in children))
                return
            ordered = sorted(children, key=lambda child: (child.item.cumulative_nll, child.item.insertion_order, child.item.work_item_id))
            if policy_name == "CURRENT_DFS":
                ordered = list(reversed(ordered))
            for child in ordered:
                policy.push(child.item)
        else:
            policy.push_successors(tuple(child.item for child in children))

    def process_logits(*, logits: Any, remaining: int, score: float, regret: float, position: int,
                       parent_node: int, prefix: tuple[int, ...], parent_work_item_id: int | None,
                       root_branch_id: int | None, parent_discrepancy: int, is_root: bool) -> list[_Frame]:
        nonlocal next_work_item_id
        log_probs = logits.float().cpu().log_softmax(-1)[0]
        values = [(token, float(log_probs[token].item())) for token in config.arc_tokens]
        ranked = sorted(values, key=lambda pair: (-pair[1], pair[0]))
        probabilities = [math.exp(logprob) for _token, logprob in values]
        state["branch_probabilities"].append({
            "forward_index": state["model_forwards"], "lane": 0, "parent_node_id": parent_node,
            "token_position": position, "prefix_length": len(prefix), "cumulative_score_before": score,
            "cumulative_regret_before": regret,
            "full_arc_logprobs": [{"token_id": token, "logprob": logprob} for token, logprob in values],
            "top1_token_id": ranked[0][0], "top1_logprob": ranked[0][1],
            "top2_token_id": ranked[1][0], "top2_logprob": ranked[1][1],
            "margin": ranked[0][1] - ranked[1][1],
            "entropy": -sum(p * lp for p, (_token, lp) in zip(probabilities, values, strict=True)),
            "batch_size": 1, "active_lane_count": 1, "active_mask": (True,), "decoder_policy": config.policy_id,
        })
        kept, prune_reason = _retained(config, ranked, score_before=score, regret_before=regret,
                                       remaining=remaining, generated_length=len(prefix))
        state["retained_successors_by_parent"][parent_node] = sorted(int(value[2]) for value in kept)
        legal_tokens = {token for token, _lp in values if token == 15 or remaining > 1}
        kept_tokens = {value[2] for value in kept}
        for token, logprob in values:
            rank = next(index for index, (ranked_token, _x) in enumerate(ranked, start=1) if ranked_token == token)
            common = {
                "event_kind": "successor", "parent_node_id": parent_node, "lane": 0,
                "prefix_length": len(prefix) + 1, "prefix_hash": _prefix_hash(prefix + (token,)), "last_token": token,
                "local_token_rank": rank, "token_logprob": logprob, "path_cumulative_nll": score - logprob,
                "path_cumulative_regret": regret + ranked[0][1] - logprob, "frontier_insert_order": None,
                "frontier_pop_order": None, "frontier_size_at_insert": None, "frontier_size_at_pop": None,
                "successors_considered": len(legal_tokens), "successors_retained": len(kept), "prune_reason": None,
                "dedup_rejected": False, "dedup_reason": None, "candidate_completed": False,
                "candidate_completion_index": None, "nodes_expanded_so_far": state["expanded_nodes"],
                "candidates_completed_so_far": state["completed_candidates"],
            }
            if token not in legal_tokens:
                _node(state, parent_node_id=parent_node, lane=0, token_position=position, branch_depth=len(prefix) + 1,
                      selected_token=token, token_logprob=logprob, cumulative_score=score - logprob, state="pruned",
                      prune_reason="max_new_tokens", termination_reason="max_new_tokens", branch_rank=None)
                _trace(state, **{**common, "prune_reason": "max_new_tokens"})
            elif token not in kept_tokens:
                _node(state, parent_node_id=parent_node, lane=0, token_position=position, branch_depth=len(prefix) + 1,
                      selected_token=token, token_logprob=logprob, cumulative_score=score - logprob,
                      cumulative_regret=regret + ranked[0][1] - logprob, state="pruned", prune_reason=prune_reason,
                      termination_reason=None, branch_rank=None)
                _trace(state, **{**common, "prune_reason": prune_reason})
        restored = False
        if not kept and legal_tokens:
            all_legal = [(score - logprob, regret + ranked[0][1] - logprob, token, logprob)
                         for token, logprob in ranked if token in legal_tokens]
            kept = sorted(all_legal, key=lambda value: (value[0], value[2]))[:config.frontier_floor]
            restored = True
            state["frontier_floor_events"].append({"activation_index": len(state["frontier_floor_events"]), "lane": 0,
                "parent_node_id": parent_node, "token_position": position, "prefix_length": len(prefix),
                "restored_count": len(kept), "restored_tokens": [value[2] for value in kept],
                "restored_cumulative_nll": [value[0] for value in kept], "reason": "policy_pruning_empty_frontier",
                "decoder_policy": config.policy_id})
        eos_logprob = next(logprob for token, logprob in values if token == 15)
        eos_rank = next(index for index, (token, _logprob) in enumerate(ranked, start=1) if token == 15)
        eos_legal = 15 in legal_tokens
        eos_retained = any(token == 15 for _score, _regret, token, _logprob in kept)

        def emit_eos(primary: str, *, retained: bool, floor_rank: int | None, completed: bool,
                     candidate_id: int | None, prune: str | None, termination: str | None, terminal_node: int | None = None) -> None:
            _emit_eos_event(state, cell_key=cell_key, parent_node_id=parent_node, parent_prefix_hash=_prefix_hash(prefix),
                token_position=position, prefix_length=len(prefix), nodes_expanded_so_far=int(state["expanded_nodes"]),
                candidates_completed_so_far=int(state["completed_candidates"]), model_forwards_so_far=int(state["model_forwards"]),
                eos_token_id=15, eos_local_rank=eos_rank, eos_logprob=eos_logprob, best_token_logprob=ranked[0][1],
                eos_logprob_gap_from_best=ranked[0][1] - eos_logprob, eos_path_cumulative_nll=score - eos_logprob,
                eos_path_cumulative_regret=regret + ranked[0][1] - eos_logprob, eos_legal=eos_legal,
                eos_retained=retained, frontier_floor_activated=restored, frontier_floor_restore_rank=floor_rank,
                candidate_completed=completed, candidate_completion_index=candidate_id,
                candidate_terminal_node_id=terminal_node, prune_reason=prune, termination_reason=termination,
                eos_primary_class=primary, checkpoint_region=_eos_checkpoint_region(int(state["expanded_nodes"])))

        if not eos_legal:
            emit_eos("EOS_MAX_NEW_TOKENS_BLOCKED", retained=False, floor_rank=None, completed=False, candidate_id=None,
                     prune="max_new_tokens", termination="max_new_tokens")
        elif not eos_retained:
            emit_eos("EOS_POLICY_PRUNED", retained=False, floor_rank=None, completed=False, candidate_id=None,
                     prune=prune_reason, termination=None)
        children: list[_Frame] = []
        for restore_rank, (next_score, next_regret, token, logprob) in enumerate(kept, start=1):
            rank = next(index for index, (ranked_token, _x) in enumerate(ranked, start=1) if ranked_token == token)
            common = {
                "event_kind": "successor", "parent_node_id": parent_node, "lane": 0,
                "prefix_length": len(prefix) + 1, "prefix_hash": _prefix_hash(prefix + (token,)), "last_token": token,
                "local_token_rank": rank, "token_logprob": logprob, "path_cumulative_nll": next_score,
                "path_cumulative_regret": next_regret, "frontier_insert_order": None, "frontier_pop_order": None,
                "frontier_size_at_insert": None, "frontier_size_at_pop": None, "successors_considered": len(legal_tokens),
                "successors_retained": len(kept), "prune_reason": None, "dedup_rejected": False, "dedup_reason": None,
                "candidate_completed": False, "candidate_completion_index": None, "nodes_expanded_so_far": state["expanded_nodes"],
                "candidates_completed_so_far": state["completed_candidates"],
            }
            floor_rank = restore_rank if restored else None
            if token == 15:
                if state["completed_candidates"] >= config.max_completed_candidates:
                    state["budget_exhausted"] = True
                    _node(state, parent_node_id=parent_node, lane=0, token_position=position, branch_depth=len(prefix) + 1,
                          selected_token=token, token_logprob=logprob, cumulative_score=next_score, cumulative_regret=next_regret,
                          state="pruned", prune_reason="candidate_budget", termination_reason="candidate_budget", branch_rank=None)
                    _trace(state, **{**common, "prune_reason": "candidate_budget"})
                    emit_eos("EOS_CANDIDATE_BUDGET_PRUNED", retained=True, floor_rank=floor_rank, completed=False,
                             candidate_id=None, prune="candidate_budget", termination="candidate_budget")
                    continue
                node_id = _node(state, parent_node_id=parent_node, lane=0, token_position=position, branch_depth=len(prefix) + 1,
                                selected_token=token, token_logprob=logprob, cumulative_score=next_score, cumulative_regret=next_regret,
                                state="completed", prune_reason=None, termination_reason="eos", branch_rank=None,
                                frontier_floor_activated=restored, frontier_floor_restore_rank=floor_rank)
                candidate_id = state["next_candidate_id"]; state["next_candidate_id"] += 1
                suffixes.append(ReferenceTurboDFSCandidate(candidate_id, prefix + (15,), next_score, node_id,
                                                            state["model_forwards"], time.time()))
                _trace(state, **{**common, "candidate_completed": True, "candidate_completion_index": candidate_id})
                emit_eos("EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED" if restored else "EOS_COMPLETED", retained=True,
                         floor_rank=floor_rank, completed=True, candidate_id=candidate_id, prune=None, termination="eos",
                         terminal_node=node_id)
                state["completed_candidates"] += 1
                continue
            work_id = next_work_item_id; next_work_item_id += 1
            root_id = work_id if is_root else int(root_branch_id)
            discrepancy_increment = 0 if rank == 1 else 1
            item = RetainedWorkItem(
                work_id, root_id, parent_work_item_id, next_score, next_regret,
                state["next_frontier_insert_order"], local_token_rank=rank,
                discrepancy_count=parent_discrepancy + discrepancy_increment,
            )
            state["next_frontier_insert_order"] += 1
            frame = _Frame(item, token, logprob, parent_node, prefix + (token,), position, restored, floor_rank)
            event = _trace(state, **{**common, "frontier_insert_order": item.insertion_order,
                                     "frontier_size_at_insert": len(children) + 1})
            if event is not None:
                state["trace_pending"][work_id] = event
            frames[work_id] = frame
            work_rows[work_id] = record_item(frame)
            children.append(frame)
        pending_by_parent[parent_node] = len(children)
        _record_frontier(state, [children])
        return children

    def crop_to_prefix(target: tuple[int, ...]) -> Generator[ReadyForwardRequest, Any, None]:
        nonlocal current_prefix, ordinal
        if _is_transformers_cache(cache_owner.cache):
            common = _lcp(current_prefix, target)
            cache_owner.cache.crop(root_cache_length + common)
            current_prefix = current_prefix[:common]
        else:
            if root_legacy_snapshot is None:
                raise RuntimeError("non-Transformers search-order cache lacks root snapshot")
            cache_owner.cache = _restore_cache_kind(clone_legacy_cache(root_legacy_snapshot), root_legacy_snapshot)
            current_prefix = ()
        for token in target[len(current_prefix):]:
            request = ReadyForwardRequest(cell_key, ordinal, token, root_position + len(current_prefix), cache_owner,
                                          cache_geometry(cache_owner.cache), root_node_id)
            ordinal += 1
            outputs = yield request
            _update_owner_after_reply(cache_owner, outputs)
            state["model_forwards"] += 1
            state["search_order"]["replayed_tokens"] += 1
            state["search_order"]["replay_model_forwards"] += 1
            current_prefix = current_prefix + (token,)

    roots = process_logits(logits=root_logits, remaining=max_new_tokens, score=0.0, regret=0.0,
                           position=root_position, parent_node=root_node_id, prefix=tuple(), parent_work_item_id=None,
                           root_branch_id=None, parent_discrepancy=0, is_root=True)
    add_successors(roots, is_root=True)
    while (not state["budget_exhausted"] and _elapsed_budget(state, started_unix) < config.local_time_limit_seconds and
           (config.absolute_end_time_unix is None or time.time() < config.absolute_end_time_unix)):
        if state["expanded_nodes"] >= config.max_expanded_nodes:
            state["budget_exhausted"] = True
            break
        item = policy.pop()
        if item is None:
            break
        frame = frames[item.work_item_id]
        record = work_rows[item.work_item_id]
        wait = int(state["expanded_nodes"]) - int(item.insertion_order)
        record.update({"first_scheduled_node": int(state["expanded_nodes"]) + 1,
                       "wait_nodes_before_first_expansion": max(0, wait), "max_wait_nodes": max(0, wait)})
        # Crop to the longest shared prefix, then replay only the missing
        # parent prefix.  Replays never increment the scientific node budget.
        yield from crop_to_prefix(frame.prefix[:-1])
        node_id = _node(state, parent_node_id=frame.parent_node_id, lane=0, token_position=frame.position,
                        branch_depth=len(frame.prefix), selected_token=frame.token_id, token_logprob=frame.token_logprob,
                        cumulative_score=item.cumulative_nll, cumulative_regret=item.cumulative_regret, state="expanded",
                        prune_reason=None, termination_reason=None, branch_rank=1,
                        frontier_floor_activated=frame.restored, frontier_floor_restore_rank=frame.restore_rank)
        state["expanded_nodes"] += 1
        record.update({"last_scheduled_node": node_id, "number_of_expansions": 1, "status": "expanded"})
        event = state["trace_pending"].pop(item.work_item_id, None)
        if event is not None:
            remaining_siblings = max(0, pending_by_parent.get(frame.parent_node_id, 1) - 1)
            pending_by_parent[frame.parent_node_id] = remaining_siblings
            event.update({"frontier_pop_order": state["next_frontier_pop_order"], "frontier_size_at_pop": remaining_siblings,
                          "expanded_node_id": node_id, "nodes_expanded_so_far": state["expanded_nodes"],
                          "elapsed_seconds": time.perf_counter() - state["trace_started_perf"]})
            state["next_frontier_pop_order"] += 1
        request = ReadyForwardRequest(cell_key, ordinal, frame.token_id, frame.position, cache_owner,
                                      cache_geometry(cache_owner.cache), node_id)
        ordinal += 1
        outputs = yield request
        _update_owner_after_reply(cache_owner, outputs)
        state["model_forwards"] += 1; state["tokens_advanced"] += 1
        state["search_order"]["useful_model_forwards"] += 1
        current_prefix = frame.prefix
        children = process_logits(logits=outputs.logits[:, -1], remaining=max_new_tokens - len(frame.prefix),
                                  score=item.cumulative_nll, regret=item.cumulative_regret, position=frame.position + 1,
                                  parent_node=node_id, prefix=frame.prefix, parent_work_item_id=item.work_item_id,
                                  root_branch_id=item.root_branch_id, parent_discrepancy=item.discrepancy_count,
                                  is_root=False)
        add_successors(children, is_root=False)
        if policy.note_expansion(item):
            record["number_of_yields"] += 1
            state["search_order"]["yielded_subtrees"] += 1
    pending = [row for row in state["search_order_work_items"] if row["status"] == "pending"]
    for row in pending:
        row["status"] = "still_pending_at_r1024" if state["budget_exhausted"] else "unexpanded_search_exhausted"
    state["search_order"]["pending_retained_work_at_r1024"] = len(pending) if state["budget_exhausted"] else 0
    return suffixes


def start_search_order_cell(
    *, model: Any, input_ids: Any, config: D1TurboDFSConfig, cell_key: str, policy_name: SearchOrderName,
    normalize_root_cache: bool, active_time_accounting: bool = False,
    root_cache_transform: Callable[[Any], Any] | None = None,
    release_prefill_temporaries: bool = False, prefill_output_references: list[Any] | None = None,
    eos_event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> ReadyCell:
    """Create one E1 ReadyCell; only its logical continuation order is new.

    ``CURRENT_DFS`` deliberately uses the established recursive ReadyCell
    implementation.  That is the only way for P0 to retain its historical
    forward indices, cache transitions, frontier samples, and result receipt
    byte-for-byte.  The explicit retained-work scheduler is the shared core
    for the two experimental orders, whose only intended difference is their
    choice among the same retained continuations.
    """
    import torch

    if input_ids.ndim != 2 or tuple(input_ids.shape[:1]) != (1,):
        raise ValueError("search-order cells require one prompt at a time")
    if policy_name == "CURRENT_DFS":
        cell = start_ready_cell(
            model=model,
            input_ids=input_ids,
            config=config,
            cell_key=cell_key,
            normalize_root_cache=normalize_root_cache,
            active_time_accounting=active_time_accounting,
            root_cache_transform=root_cache_transform,
            release_prefill_temporaries=release_prefill_temporaries,
            prefill_output_references=prefill_output_references,
            eos_event_sink=eos_event_sink,
        )
        cell.state["search_order"] = {
            "policy": policy_name,
            "quantum": 64,
            "retention_changed": False,
            "implementation": "established_recursive_ready_dfs",
            "replayed_tokens": 0,
            "replay_model_forwards": 0,
            "useful_model_forwards": int(cell.state["tokens_advanced"]),
            "yielded_subtrees": 0,
            "pending_retained_work_at_r1024": 0,
        }
        cell.state["search_order_work_items"] = []
        return cell
    state = _new_state(config); state["eos_event_sink"] = eos_event_sink
    state["active_time_accounting"] = active_time_accounting; state["per_forward_trace"] = []
    started_unix = time.time(); created_perf = time.perf_counter()
    with torch.no_grad():
        started = time.perf_counter(); outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
        prefill_seconds = time.perf_counter() - started
    state["model_forward_seconds"] += prefill_seconds; state["active_elapsed_seconds"] += prefill_seconds; state["model_forwards"] = 1
    root_node = _node(state, parent_node_id=None, lane=0, token_position=int(input_ids.size(1)), branch_depth=0,
                      selected_token=None, token_logprob=None, cumulative_score=0.0, cumulative_regret=0.0,
                      state="root", prune_reason=None, termination_reason=None, branch_rank=None)
    root_cache = _legacy_cache(outputs.past_key_values) if normalize_root_cache else outputs.past_key_values
    if root_cache_transform is not None:
        root_cache = root_cache_transform(root_cache)
    root_logits = outputs.logits[:, -1]
    if release_prefill_temporaries:
        root_logits = root_logits.clone()
        if prefill_output_references is not None:
            prefill_output_references.append(outputs)
    owner = CacheOwner(root_cache)
    root_length = int(_legacy_cache(root_cache)[0][0].shape[-2])
    snapshot = None if _is_transformers_cache(root_cache) else clone_legacy_cache(root_cache)
    if config.calibration_assertions:
        batch = _cache_batch_size(root_cache)
        if batch is not None and batch != 1:
            raise RuntimeError(f"search-order scalar cache batch mismatch {batch}")
    generator = _order_generator(cell_key=cell_key, root_logits=root_logits, root_position=int(input_ids.size(1)),
                                 max_new_tokens=config.max_new_tokens, cache_owner=owner, root_cache_length=root_length,
                                 root_legacy_snapshot=snapshot, config=config, started_unix=started_unix,
                                 state=state, root_node_id=root_node, policy_name=policy_name)
    cell = ReadyCell(cell_key, config, state, started_unix, created_perf, generator, prefill_seconds=prefill_seconds,
                     active_elapsed_seconds=prefill_seconds, cache_owner=owner)
    _advance_cell(cell)
    if release_prefill_temporaries and prefill_output_references is None:
        del outputs
    return cell
