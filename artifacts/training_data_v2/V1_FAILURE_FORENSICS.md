# V1 failure forensics

V1 correctly stopped before training but its binary project-wide blacklist was intentionally too broad for corpus recovery.
`TASK_ID_RE` extracted every eight-hex token from reachable repository text; `scan_project_blacklist()` treated any matching official task ID as an exclusion.
`_classify_evidence()` used path-name heuristics. In particular, a path containing `frozen` set `reserved=True`; it did not inspect the semantic meaning of the configuration.
`llm_full_1000_frozen_v1.json` explicitly declares all official training challenges. Its `frozen` term describes configuration immutability, not a future holdout. `llm_challenge_like_frozen_v1.json` and the stopped pilot are historical exposure evidence, not permanent exclusion instructions.
Manifest/path-based `gold` inference was retained as audit metadata but is not a role assignment mechanism in v2.
Result: all 1,000 official training tasks were excluded due repository-wide textual occurrence, yielding the v1 terminal no-eligible-tasks result. V2 retains that receipt and replaces the rule with explicit semantic roles.
