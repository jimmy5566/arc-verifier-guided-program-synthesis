# Post-V7 independent-generalization mechanism audit V1

## Verdict

The Count curriculum has a confirmed identifiability defect, but its causal role in the failed independent transfer remains unproven. Topology predicts the query count in 48/48 tasks, and the first demonstration count equals the query count in 48/48 tasks. Query color and rotation phase are also constant. The cleanest next causal test changes only this curriculum construction.

## Quantitative evidence

- Original Count TRAIN: topology-to-count accuracy 48/48; first-demo-count accuracy 48/48; normalized mutual information 1.0.
- Supervision: 45/60 assistant tokens (75%) are demonstrations and 15/60 (25%) are the final query. Existing code-pair weighting preserves this 75/25 split.
- Equal-Slot effective coefficient mass: Count 12.5%, Object Extraction 12.5%, Replay 75%; the final Count query is 3.125% of full-schedule nominal mass. Raw token exposure is 2,880/27,360 Count and 21,600/27,360 replay, but those raw shares are not the Equal-Slot weights.
- Implementation audit: all 48 Count tasks pass user-mask/assistant-label, causal-shift, and mass-preservation checks. Code-Pair fails its documented semantic locator on 48/48 tasks: it selects columns 0/1, not 1/2, so Count 2 and Count 3 receive the same selected pattern.
- Independent V2: V7 64/104, Control 53/104, Treatment 52/104. Treatment-Control Count +1/24, 95% CI [-2,4]; Treatment-V7 Count -4/24, CI [-10,2].
- E04-D: positive start-state local alignment (bootstrap CI 0.842828 to 0.859694) deprioritizes direct local gradient opposition but does not prove training compatibility.
- New CPU panel: 48 TRAIN-derived tasks, topology/count MI 0.0, first-demo match 16/48, and 48/48 oracle-invariant order pairs.
- Fresh target-blind OOD: 48 canonical units and 96 paired-order prompts; no V2 or Gold/dGold targets used.

## Ranked bottlenecks

1. **Count-task identifiability and shortcut-confounded curriculum** - Defect confirmed; causal contribution to model failure unproven until controlled training. Evidence: topology->count 48/48; first-demo->query count 48/48; normalized MI(topology,count)=1.0; query color and rotation phase each constant
2. **Code-Pair decision-token locator semantics** - Implementation defect confirmed; contribution to the failed independent result is unquantified. Evidence: 0/48 tasks match the documented row-1 columns 1/2; all 48 implement columns 0/1, making Count 2 and Count 3 codes identical
3. **Weak effective query supervision under all-assistant CE and replay dilution** - Allocation confirmed; objective mismatch and replay dilution remain unproven causes. Evidence: demo/query mass=45/15 (75%/25%) within a count slot; Equal-Slot nominal mass is Count 12.5%, Object 12.5%, Replay 75%; Count-query mass is 3.125% of the full schedule
4. **Free-running exact-output acquisition versus token-level fit** - Independent generalization failure confirmed; train-fit/Greedy mechanism unresolved because comparable raw per-example fit artifacts are unavailable and the V7 teacher-forced diagnostic was invalid. Evidence: V2 exact-grid: V7 64/104, Control 53/104, Treatment 52/104; count Treatment-Control +1/24 CI [-2,4], Treatment-V7 -4/24 CI [-10,2]
5. **Object-centric representation or structural transfer** - Plausible downstream mechanism, but current curriculum confounding prevents attribution. Evidence: Both trained arms lost 2/24 object-extraction correct relative to V7 with CI [-5,0].
6. **Direct local gradient conflict** - Deprioritized: positive local alignment does not establish compatibility, but direct start-state opposition was not observed. Evidence: E04-D rotation-marker aggregate CI [0.842828,0.859694] with decision LOCAL_CONFLICT_DEPRIORITIZED

## Highest-information experiment

**POST_V7_DECONFOUNDED_COUNT_CURRICULUM_CAUSAL_PILOT_V1**

Breaking topology/count and demo-order shortcuts, while holding model, optimizer, updates, token mass, object-extraction slots, and replay fixed, improves free-running structural-OOD count accuracy.

Control: V7 initialization plus the original 48 confounded count tasks, the same 48 object-extraction tasks, and the same 288 replay slots.

Treatment: Same V7 initialization and identical non-count slots; replace only the 48 count tasks with the CPU-validated counterbalanced TRAIN-derived panel.

This is a two-arm, one-variable causal pilot. Query-only supervision is explicitly deferred to a separate conditional experiment so curriculum identifiability and objective choice are not changed together.

GPU requirement: one BF16-capable >=24GB GPU, two sequential fresh-V7 arms, hard cap 2 GPU-hours. GPU execution is not authorized by this audit.

## Evidence limits

Raw V2 per-example predictions are absent from the committed result package, so per-example error transitions are NOT_RUN. Demo/query/decision/formatting gradient decomposition is also NOT_RUN because this CPU-only audit did not load a model. Neither omission is silently converted into a causal claim.
