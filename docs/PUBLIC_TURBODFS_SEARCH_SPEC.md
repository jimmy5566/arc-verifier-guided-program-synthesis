# Public TurboDFS executable search specification

Reference: NVARC commit `846d0198efa752534594e321fc3289fc0a06c657`,
`ARC-AGI1/002_ivan_arc1.ipynb`, code cell 4, emitted file `arc_solver.py`,
functions `turbo_dfs` and `inference_turbo_dfs` (the exact source block begins
at notebook JSON line 624 and ends at line 706).  V4 implements this block in
`inference/nvarc_turbodfs_reference.py`; ARC2 only provides the adapted model,
native tokenizer, prompt, transform and post-decode canonicalisation.

| # | Public behaviour | V4 binding | Public source location |
|---|---|---|---|
| 1 | `inference_turbo_dfs` forwards the whole `prefix_tokens` tensor once with `use_cache=True`. | Root forward uses the supplied equal-width prompt batch. | cell 4, `inference_turbo_dfs`, `input_ids`/`outputs` |
| 2 | Each recursive call receives last-token logits, cache, cumulative scores and an absolute position. | Same argument state is passed recursively without conversion to a different scoring system. | cell 4, `turbo_dfs` signature |
| 3 | Every native ARC token is considered in the fixed `ARC_TOKENS` order. | `[0..10, 15]` is asserted. | cell 4, `ARC_VOCAB`, `ARC_TOKENS` |
| 4 | Token score is `torch.tensor(scores).view(n,1) - logits.float().cpu().log_softmax(-1)`. | Exact formula, float32 CPU log-softmax. | cell 4, first `nll =` line |
| 5 | Scores are cumulative negative log likelihood; there is no length normalisation. | `cumulative_nll` is retained unchanged. | cell 4, first `nll =` line |
| 6 | A token remains only when `score < max_score` (strict). | Strict `<`, never `<=`. | cell 4, `if score < max_score` |
| 7 | `max_score=-log(0.2)` is established by the public worker. | Frozen as `1.6094379124341003`. | cell 4, `max_score = -np.log(0.2)` |
| 8 | No additional cumulative-NLL, local frontier, or result-tuned pruning exists. | No non-reference pruning is implemented. | cell 4, token loop and recursive loop |
| 9 | EOS (`15`) is immediately appended as a completed suffix. | EOS creates a terminal candidate/node. | cell 4, `if t == EOS_ID` |
| 10 | Non-EOS tokens are eligible only while `max_new_tokens > 1`. | Identical length rule; rejected tokens are trace-only. | cell 4, `elif max_new_tokens > 1` |
| 11 | Candidate lists are sorted by score only; Python stable order resolves exact ties. | Sort key is score only. | cell 4, `sorted(... key=lambda x:x[0])` |
| 12 | Each lane pops one lowest-score candidate per forward. | One `pop(0)` per live lane. | cell 4, inner `for i in range(n)` |
| 13 | Empty lanes use `PAD_ID=13`, score `1000`, and stay in the batched cache forward. | Same pad/score behaviour. | cell 4, empty-candidate branch |
| 14 | Search stops when no lane has a candidate. | `alive == 0` ends that recursive frontier. | cell 4, `if num_alive_beams == 0` |
| 15 | The forward batch contains all lanes, with `position_ids` filled by current `pos`. | Identical shape and position construction. | cell 4, `outputs = model(...)` |
| 16 | The returned `past_key_values` is passed directly to the recursive call. | No cache copy, compacting, offload, or reindexing. | cell 4, `cache=outputs.past_key_values` |
| 17 | Cache branching is lane-aligned through the batched recursive model forward. | Calibration checks lane/logit/cache batch sizes when inspectable. | cell 4, batched model call + recursive call |
| 18 | Active lanes refill from their own candidate lists; a vacant lane does not clear other lanes. | Per-lane candidate lists remain independent. | cell 4, candidate lists and `num_alive_beams` |
| 19 | Recursive descendants are prefixed with their parent selected token and appended. | Same suffix construction, preserving discovery telemetry. | cell 4, `suffix_tokens.insert(0, batch_tokens[batch_id])` |
| 20 | No cross-path or grid dedup occurs inside the public search. | V4 retains every completed suffix; ARC2 canonical grid dedup is post-search only. | cell 4, `suffixes` accumulation |
| 21 | Completion is EOS, not grid parsing. | Parser validity is an ARC2 post-search label, never a branch prune. | cell 4, `if t == EOS_ID` |
| 22 | Invalid parsed grids are discarded only after decoding in the worker. | Every completed token candidate remains, with `valid_grid` recorded separately. | cell 4, worker `convert_tokens_to_array` |
| 23 | Native token filtering is exactly `ARC_TOKENS`; no general-token branching. | Runtime tokenizer preflight asserts the 16-token mapping. | cell 4, `ARC_VOCAB` and token loop |
| 24 | Decoder local time condition is `time.time()-start_time < 540`. | Exactly 540 seconds. | cell 4, recursive `while` |
| 25 | An external `end_time` is also honoured. | Optional absolute end time only; no unrelated cap. | cell 4, recursive `while` |
| 26 | There is no node, forward, candidate, or retained-suffix cap. | V4 has none. | cell 4, entire decoder block |
| 27 | Final output sorts each lane's completed beams by score only. | Same score-only ordering. | cell 4, `sorted_beams = sorted(beams, key=lambda x:x[0])` |
| 28 | Candidate post-search rescoring is outside `inference_turbo_dfs`. | V4 generation does not rescore or use Gold; existing downstream evidence remains separate. | cell 4, `calc_scores` / worker after decoder |

## Intentional integration boundary

The public notebook's task batching and ARC2's per-cell persistence differ in
orchestration only.  A V4 cell may use one lane; a grouped same-width prompt
batch may use multiple lanes.  In both cases, each lane executes the public
recursion above with no cross-lane candidate sharing.  V4 retains lane, cache
batch, model-forward and branch-probability telemetry solely for calibration
assertions and later reconstruction; those additions do not affect branching.

