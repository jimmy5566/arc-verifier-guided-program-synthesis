# Controller next mission

After infrastructure PASS, and before any real RunPod GPU training:

1. Verify the frozen Foundation-V2 baseline.
2. Derive and freeze the Base-vs-V2 composition delta.
3. Create a deterministic capability-priority map.
4. Choose targeted atomic and composition capabilities.
5. Construct fresh, isolated TRAIN, TARGET_DEV, and FINAL_AUDIT datasets.
6. Create the targeted evaluation protocol and retention sentinel.
7. Preregister acceptance and rejection gates.
8. Produce `TARGETED_CAPABILITY_REPAIR_V1` and submit it to the Director.
9. Wait for Director `CONTINUE` or `CONTINUE_WITH_WARNING`.

Do not start the 8-hour training during orchestration setup.
