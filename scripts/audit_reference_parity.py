"""CPU-only parity audit against the public NVARC ARChitects submission source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REPO_COMMIT = "846d0198efa752534594e321fc3289fc0a06c657"
REFERENCE_NOTEBOOK_SLUG = "sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _notebook_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(str(cell.get("source", "")) for cell in payload.get("cells", ()))


def _row(area: str, item: str, ours: str, reference: str, status: str, impact: str, evidence: str, rationale: str) -> dict[str, str]:
    if status not in {"CONFIRMED_MATCH", "CONFIRMED_MISMATCH", "UNVERIFIED"}:
        raise ValueError(f"invalid parity status: {status}")
    if impact not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError(f"invalid impact: {impact}")
    return {"area": area, "item": item, "ours": ours, "reference": reference, "status": status, "impact": impact, "reference_evidence": evidence, "impact_rationale": rationale}


def build_matrix(reference_repo: Path, reference_kernel: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    reference_notebook = next(reference_kernel.glob("*.ipynb"), None)
    if reference_notebook is None:
        raise FileNotFoundError("public NVARC submission notebook was not pulled")
    notebook = _notebook_source(reference_notebook)
    required = ("r=256", "turbo_dfs", "train_ds = puzzle_ds.augment(n=16", "eval_ds = puzzle_ds_multi.augment(n=2", "QwenDataCollatorForCompletionOnlyLM")
    if any(item not in notebook for item in required):
        raise ValueError("pulled reference notebook lacks required public NVARC implementation markers")
    current_native = (ROOT / "src/inference/nvarc_native.py").read_text(encoding="utf-8")
    current_adapter = (ROOT / "src/inference/arc_native_io.py").read_text(encoding="utf-8")
    current_aug = (ROOT / "src/inference/nvarc_native_augmentation.py").read_text(encoding="utf-8")
    current_ttt = (ROOT / "src/inference/nvarc_native_ttt.py").read_text(encoding="utf-8")
    current_candidates = (ROOT / "src/inference/nvarc_native_candidates.py").read_text(encoding="utf-8")
    config = json.loads((ROOT / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json").read_text(encoding="utf-8"))
    if not all(item in current_native + current_adapter + current_aug + current_ttt + current_candidates for item in ("ARCNativeInputAdapter", "bounded_native_augmentations", "NativeTaskLoRA", "deduplicate_candidates")):
        raise ValueError("current native implementation markers are missing")
    rows = [
        _row("MODEL / TOKENIZER", "checkpoint logical identifier", config["model"]["source"], "same Kaggle model source path in public notebook", "UNVERIFIED", "HIGH", "Both paths name sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1; no public model-file hash/revision is recorded.", "Weight revision drift could directly alter candidate recall."),
        _row("MODEL / TOKENIZER", "dtype", "Transformers BF16", "Unsloth BF16, load_in_4bit=False", "CONFIRMED_MATCH", "LOW", "current NVARCNativeProvider uses torch.bfloat16; public worker sets bf16=True and load_in_4bit=False.", "Numerical backend differs, but declared precision matches."),
        _row("MODEL / TOKENIZER", "16-token vocabulary", "digits 0-9, newline=10, user=11, assistant=12, pad=13, im_start=14, im_end=15", "same ARC_VOCAB and role IDs", "CONFIRMED_MATCH", "LOW", "Public ARC_VOCAB/USER_TOKEN_ID/ASSISTANT_TOKEN_ID/PAD_ID/EOS_ID match local tokenizer semantic mapping.", "Token identity mismatch is ruled out at the documented mapping level."),
        _row("MODEL / TOKENIZER", "tokenizer files", "locally repackaged checkpoint config files", "ARChitects/qwen3_configs files", "CONFIRMED_MATCH", "LOW", "Raw JSON hashes differ due formatting/metadata, while vocab, chat template and BOS/EOS semantic fields match.", "No semantic tokenizer mismatch was found; exact model-attached tokenizer revision remains unverified."),
        _row("MODEL / TOKENIZER", "BOS / EOS / padding", "no BOS; EOS=<|im_end|>; pad=<|endoftext|>", "same", "CONFIRMED_MATCH", "LOW", "Both tokenizer_config.json files specify add_bos_token=false, eos <|im_end|>, pad <|endoftext|>.", "Termination token handling is compatible at tokenizer level."),
        _row("MODEL / TOKENIZER", "chat template", "alternating <|im_start|>role\\ncontent<|im_end|>, assistant generation prefix", "QwenFormatter emits the same literal layout", "CONFIRMED_MATCH", "LOW", "Local chat_template.j2 and public QwenFormatter.fmt_query/fmt_train render the same role-token layout.", "Prompt wrapper mismatch is ruled out for normal native messages."),
        _row("TASK SERIALIZATION", "train-pair order", "canonical per base view; optional only canonical/reversed augmentation", "random full train-example permutation in shuffle_ex", "CONFIRMED_MISMATCH", "HIGH", "Local NativeAugmentation supports only canonical/reversed; public ArcDataset.augment calls shuffle_ex() with np.random.permutation(n).", "Order diversity can materially affect task-conditioned adaptation and generation."),
        _row("TASK SERIALIZATION", "test-input placement", "train user/assistant turns followed by final user grid and assistant prefix", "same", "CONFIRMED_MATCH", "LOW", "Local ARCNativeInputAdapter.messages and public fmt_train+fmt_query have the same turn placement.", "No material placement discrepancy found."),
        _row("TASK SERIALIZATION", "grid serialization", "digits concatenated per row; newline rows; no commas/brackets", "same", "CONFIRMED_MATCH", "LOW", "Local serialize_grid and public convert_grid_to_string both emit digit rows joined by newline.", "Core grid transport is faithful."),
        _row("TASK SERIALIZATION", "assistant target / termination", "grid text; generation stops at im_end; scorer appends EOS", "grid text plus <|im_end|>", "CONFIRMED_MATCH", "LOW", "Local generation eos_token_id and likelihood EOS append align with public fmt_reply.", "No documented target-format mismatch."),
        _row("TASK SERIALIZATION", "malformed-grid handling", "strictly rejects prose, special tokens, ragged rows and over-30 grids", "extracts digits by line and truncates rows above 30", "CONFIRMED_MISMATCH", "MEDIUM", "ARCNativeOutputParser is intentionally strict; public convert_tokens_to_array filters digits and truncates excess rows.", "Reference may retain candidates local parser discards."),
        _row("AUGMENTATION", "geometry family", "8 dihedral views including flips and anti-transpose", "transpose × four rotations (same 8 dihedral transforms)", "CONFIRMED_MATCH", "LOW", "Local bounded_native_augmentations and public ArcDataset.augment cover equivalent D4 geometry set.", "Geometry coverage itself is not the discrepancy."),
        _row("AUGMENTATION", "colour augmentation", "two cyclic offsets: 0,1", "random arbitrary 10-colour permutations", "CONFIRMED_MISMATCH", "HIGH", "Local config color_offsets=[0,1]; public permute_rnd_all_ uses np.random.permutation(10).", "Reference explores much broader colour symmetry than Aug8."),
        _row("AUGMENTATION", "generation augmentation count", "Aug8: one instance of each geometry, original colours/order", "16 views: 8 geometry × 2 random colour permutations", "CONFIRMED_MISMATCH", "HIGH", "Local Eval runs first 8 bounded views; public eval_ds.augment(n=2) creates 16 views.", "Candidate recall is directly bounded by generated views."),
        _row("AUGMENTATION", "TTT augmentation count", "8 canonical geometry views, one selected pair per step", "128 task sequences: 8 geometry × 16 colour permutations, each with shuffled pair order", "CONFIRMED_MISMATCH", "HIGH", "Local runner passes canonical Aug8 to 24-step NativeTaskLoRA; public train_ds.augment(n=16) is one-epoch training dataset.", "This changes adaptation data volume and representation coverage by orders of magnitude."),
        _row("AUGMENTATION", "inverse transform", "explicit inverse geometry and cyclic colour offset", "reverse descriptor order for rotation/transpose/permutation", "CONFIRMED_MATCH", "LOW", "Both implementations explicitly invert the applied view before candidate pooling.", "No documented coordinate-frame loss."),
        _row("AUGMENTATION", "rescoring views", "8 fixed geometric B-support views, no random colour permutation", "8 geometry × one seeded random colour permutation candidate score views", "CONFIRMED_MISMATCH", "MEDIUM", "Local b_support_views are geometry-only; public aug_dataset.augment(seed=hash(bk)) scores 8 permuted geometric views.", "Selection differs, though existing Eval60 evidence says selection is not primary."),
        _row("TTT", "LoRA rank", "8", "256", "CONFIRMED_MISMATCH", "HIGH", "Local NativeLoRAConfig/default frozen config rank=8; public peft_params r=256.", "Adapter capacity differs by 32×."),
        _row("TTT", "LoRA alpha", "16", "32", "CONFIRMED_MISMATCH", "HIGH", "Local alpha=16; public lora_alpha=32.", "Changes effective update scale with rank."),
        _row("TTT", "target modules", "q_proj, v_proj", "q/k/v/o projections, gate/up/down MLP, embed_tokens, lm_head", "CONFIRMED_MISMATCH", "HIGH", "Local target_suffixes are q_proj/v_proj; public peft_params lists nine module families.", "Reference adapts substantially more of the model."),
        _row("TTT", "optimizer", "AdamW", "adamw_torch", "CONFIRMED_MATCH", "LOW", "Local uses torch.optim.AdamW; public train_args optim=adamw_torch.", "Optimizer family matches, but schedule and LR do not."),
        _row("TTT", "learning rate", "5e-4", "5e-5", "CONFIRMED_MISMATCH", "HIGH", "Frozen light TTT config uses 0.0005; public train_args learning_rate=0.00005.", "A 10× LR change can dominate adaptation behavior."),
        _row("TTT", "scheduler", "none", "cosine with warmup_ratio=0.1", "CONFIRMED_MISMATCH", "MEDIUM", "Local per-task AdamW has no scheduler; public lr_scheduler_type=cosine.", "May matter during the larger reference epoch."),
        _row("TTT", "step / epoch schedule", "24 fixed optimization steps", "one epoch over augmented dataset", "CONFIRMED_MISMATCH", "HIGH", "Local fit_task loops effective_steps=24; public num_train_epochs=1 over train_ds.", "Reference update count is task/data dependent and far larger."),
        _row("TTT", "batch construction", "one transformed train pair per step", "full puzzle sequence per dataset example", "CONFIRMED_MISMATCH", "HIGH", "Local fit_task selects augmented.train[...] then one input/output; public Dataset.from_list(train_ds.as_list(formatter)) serializes all task train pairs in each example.", "Reference learns from pair relations in the full prompt context."),
        _row("TTT", "loss masking", "only one output continuation per update", "all assistant turns masked for completion-only loss", "CONFIRMED_MISMATCH", "HIGH", "Local cross entropy targets one output grid; public QwenDataCollatorForCompletionOnlyLM labels assistant spans only across full dialogue.", "This is a material objective mismatch."),
        _row("TTT", "gradient accumulation", "1", "1", "CONFIRMED_MATCH", "LOW", "Local one update per loop; public gradient_accumulation_steps=1.", "No accumulation discrepancy."),
        _row("TTT", "precision / checkpointing", "BF16 with gradient checkpointing enabled", "BF16 with gradient checkpointing disabled", "CONFIRMED_MISMATCH", "MEDIUM", "Local fit_task enables checkpointing; public use_gradient_checkpointing=False and gradient_checkpointing=False.", "Mostly throughput/memory, potentially numerics."),
        _row("TTT", "reset policy", "reset adapter before and after every task", "restore cloned default PEFT state before every task", "CONFIRMED_MATCH", "LOW", "Local reset(seed) and public set_peft_model_state_dict(default_weights) both prevent cross-task adapter leakage.", "Both have an explicit task reset."),
        _row("TTT", "adapter implementation", "custom replacement Linear adapters", "PEFT/Unsloth LoRA including embedding target", "CONFIRMED_MISMATCH", "HIGH", "Local NativeTaskLoRA only wraps compatible Linear modules; public FastLanguageModel.get_peft_model uses PEFT targets including embed_tokens.", "Exact reference adapter topology is not reproduced by light TTT."),
        _row("DECODING", "primary decoding", "greedy model.generate", "batched Turbo DFS", "CONFIRMED_MISMATCH", "HIGH", "Local do_sample=False/search_beams=1; public inference_turbo_dfs is the generation path.", "Reference candidate search is absent from baseline generator."),
        _row("DECODING", "DFS branch / pruning", "no DFS in Aug8 generator", "all ARC tokens below NLL max_score=-log(0.2), global per-call time cap", "CONFIRMED_MISMATCH", "HIGH", "Public turbo_dfs expands all tokens satisfying score < max_score and caps decoding at 540s.", "A different candidate set is expected even on identical logits."),
        _row("DECODING", "max completion length", "1024", "formatter-derived 30×30 grid plus EOS", "CONFIRMED_MISMATCH", "MEDIUM", "Local config max_new_tokens=1024; public max_new_tokens() derives native-token length for max grid.", "Bounds can alter termination and search space."),
        _row("DECODING", "sampling", "deterministic greedy", "deterministic score-bounded DFS", "CONFIRMED_MISMATCH", "HIGH", "Neither uses stochastic sampling in the public notebook's inference_turbo_dfs path, but only reference branches alternatives.", "Search diversity is materially different."),
        _row("DECODING", "KV/cache batching", "single-request greedy in frozen Aug8 path", "four-view batches with shared prefix KV cache through DFS", "CONFIRMED_MISMATCH", "MEDIUM", "Public batches subkeys in groups of four and carries past_key_values; current frozen condition uses generation batch size 1.", "Primarily throughput but batch behaviour can affect numerical tie ordering."),
        _row("CANDIDATE PROCESSING", "inverse-frame pooling", "inverse each generated view then pool full multi-test prediction", "invert descriptor before storing solution", "CONFIRMED_MATCH", "LOW", "Local augmentation.inverse_grid and public puzzle_ds_multi.invert_mod both restore original frame before grouping.", "No documented pooling-frame discrepancy."),
        _row("CANDIDATE PROCESSING", "dedup / multiplicity", "exact grid dedup preserving support_count", "groups identical grids and uses number of guesses", "CONFIRMED_MATCH", "LOW", "Local deduplicate_candidates retains all support augmentations; public score_sum groups hashable solutions and count is used by score_kgmon.", "Multiplicity preservation is conceptually aligned."),
        _row("CANDIDATE PROCESSING", "original likelihood", "mean conditional likelihood under original native prompt", "DFS beam NLL score", "CONFIRMED_MISMATCH", "MEDIUM", "Local rank_candidates does teacher-forced original-prompt likelihood; public keeps turbo DFS beam_score.", "Different candidate ordering, but cannot explain pool recall alone."),
        _row("CANDIDATE PROCESSING", "augmented likelihood", "eight geometry-only B-support likelihoods", "eight geometry+permutation score_aug values", "CONFIRMED_MISMATCH", "MEDIUM", "Local candidate_view_scores_many differs from public calc_scores over augmented ArcDataset.", "Selection difference; prior Eval60 has no observed selection misses."),
        _row("CANDIDATE PROCESSING", "final Top-2 selection", "frozen B-SUPPORT selector", "score_full_probmul_3 / score_kgmon grouping", "CONFIRMED_MISMATCH", "MEDIUM", "Current selector and public selection_algorithms are not identical.", "Relevant to rank, not absence of correct candidates."),
        _row("CANDIDATE PROCESSING", "task/test output mapping", "explicit task_id and test_index lists", "split_multi_replies key then fill_submission base_id/index", "CONFIRMED_MATCH", "LOW", "Both preserve per-test-output IDs before assembling submission.", "No source-level mapping mismatch found."),
    ]
    provenance = {
        "reference_repository": "https://github.com/1ytic/NVARC",
        "reference_repository_commit": REFERENCE_REPO_COMMIT,
        "reference_notebook": f"https://www.kaggle.com/code/{REFERENCE_NOTEBOOK_SLUG}",
        "reference_notebook_sha256": _sha256(reference_notebook),
        "reference_notebook_local": str(reference_notebook),
        "current_native_config_sha256": _sha256(ROOT / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"),
        "current_reference_tokenizer_dir": str(ROOT / "configs/nvarc_native_846d0198"),
    }
    return rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any((args.output_dir / name).exists() for name in ("REFERENCE_PARITY_MATRIX.csv", "REFERENCE_PARITY_REPORT.md", "REFERENCE_PARITY_REPORT.json")):
        raise FileExistsError("refusing to overwrite a completed reference parity audit")
    rows, provenance = build_matrix(args.reference_repo, args.reference_kernel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "REFERENCE_PARITY_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    high_mismatches = [row for row in rows if row["status"] == "CONFIRMED_MISMATCH" and row["impact"] == "HIGH"]
    unverified_critical = [row for row in rows if row["status"] == "UNVERIFIED" and row["impact"] == "HIGH"]
    top_five = high_mismatches[:5]
    report = {
        "experiment_id": "ARC2_CANDIDATE_GENERATION_REFERENCE_PARITY",
        "status": "COMPLETE_CPU_ONLY_REFERENCE_AUDIT",
        "scope": "Code/source audit only; no model loading, CUDA, Kaggle inference, target scoring, or production-code modification.",
        "provenance": provenance,
        "row_count": len(rows),
        "status_counts": {value: sum(row["status"] == value for row in rows) for value in ("CONFIRMED_MATCH", "CONFIRMED_MISMATCH", "UNVERIFIED")},
        "high_impact_mismatch_count": len(high_mismatches),
        "unverified_critical_count": len(unverified_critical),
        "top_5_most_important_differences": [{"area": row["area"], "item": row["item"], "ours": row["ours"], "reference": row["reference"]} for row in top_five],
        "stage_b_gate": {"reference_style_ttt_is_documented": True, "light_ttt_is_reference_faithful": False, "required_reference_elements": ["rank=256", "alpha=32", "all documented PEFT target modules", "one epoch over n=16 augmented full-puzzle sequences", "assistant-only dialogue masking", "BF16 / no gradient checkpointing", "AdamW cosine LR 5e-5"]},
        "causality_statement": "The audit identifies plausible interface/inference causes of low candidate recall; it does not establish causality without the staged reference-style pilot.",
    }
    (args.output_dir / "REFERENCE_PARITY_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    top = "\n".join(f"{index}. **{row['area']} / {row['item']}** — ours: {row['ours']}; reference: {row['reference']}" for index, row in enumerate(top_five, 1))
    (args.output_dir / "REFERENCE_PARITY_REPORT.md").write_text(
        "# Reference inference parity audit\n\n"
        "This is a source-only audit against the public NVARC repository and its linked Kaggle submission notebook. It does not claim that a mismatch is causal.\n\n"
        f"- Reference repository commit: `{REFERENCE_REPO_COMMIT}`\n"
        f"- Reference notebook SHA256: `{provenance['reference_notebook_sha256']}`\n"
        f"- HIGH_IMPACT_MISMATCH_COUNT = **{len(high_mismatches)}**\n"
        f"- UNVERIFIED_CRITICAL_COUNT = **{len(unverified_critical)}**\n\n"
        "## Top five differences\n\n" + top + "\n\n"
        "The full row-level evidence and classifications are in `REFERENCE_PARITY_MATRIX.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"HIGH_IMPACT_MISMATCH_COUNT": len(high_mismatches), "UNVERIFIED_CRITICAL_COUNT": len(unverified_critical), "TOP_5_MOST_IMPORTANT_DIFFERENCES": [row["item"] for row in top_five]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
