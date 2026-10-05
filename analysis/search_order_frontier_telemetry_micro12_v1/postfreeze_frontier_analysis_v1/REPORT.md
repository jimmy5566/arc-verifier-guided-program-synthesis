# Frontier telemetry Micro12 post-freeze Gold analysis

Integrity
- Frozen source commit: 9b5109e16d4aa30ebb1f44521df276b99e35029e
- Gold SHA256: 84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1
- Telemetry items: 99,624
- Gold-prefix retained items: 13,996
- Never-expanded Gold-prefix retained items: 58
- Cells with a never-expanded Gold-prefix item: 58
- P3 priority reconstruction mismatches: 0

True frontier decision set
- Scientific pop decisions reconstructed: 95,928
- Gold-opportunity frontier states: 64,005
- Maximum frontier size: 309

Same-frontier ranking
Strongest frozen baseline: CUMULATIVE_NLL
Top1 baseline=0.329052 learned=0.360706 delta=+0.031654
Top2 baseline=0.484259 learned=0.595953 delta=+0.111694
Top5 baseline=0.807234 learned=0.865760 delta=+0.058527
MRR baseline=0.520379 learned=0.563891 delta=+0.043512

Task-bootstrap 95% CI
Delta Top1=[-0.047404,+0.119117]
Delta MRR=[-0.008829,+0.114206]
Catastrophic folds=1

Secondary global diagnostics
ROC-AUC=0.936005
Average precision=0.664824
Positive prevalence=0.140488

Decision
FRONTIER_VALUE_SIGNAL_NOT_ESTABLISHED
P4 authorized=NO
Next=DO_NOT_LAUNCH_P4_VALUE_GUIDED_SEARCH
No P4 search policy was implemented or run.
