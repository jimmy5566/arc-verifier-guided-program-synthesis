"""Preregistered CPU-only Stage-1 simulator and analysis runner."""
from __future__ import annotations
import csv,json
from pathlib import Path
import sys
import numpy as np

THIS=Path(__file__).resolve(); ROOT=THIS.parents[1]
sys.path.insert(0,str(THIS.parent))
from attribution import CLASSES, fit_models, probabilities
from metrics import attribution_metrics
from reliability_update import run_updates
from plotting import confusion, lines, scatter

OUT=ROOT/'outputs'
REGIMES={
 'A_balanced':([.40,.15,.15,.15,.15],1101),
 'B_low_external':([.50,.15,.25,.05,.05],1201),
 'C_high_non_selector':([.35,.20,.10,.15,.20],1301),
 'D_selector_heavy':([.40,.10,.30,.10,.10],1401),
}
FEATURES=['steps','failure_step_ratio','candidate_count','score_margin','selector_entropy','disagreement','committed_action','executor_status','execution_latency','state_delta_ratio','expected_transition','environment_warning','prior_failure_rate']
FEATURE_GROUPS={'selector_scores':[3,4,5,6],'execution':[7,8,9,10],'environment':[10,11,12],'all':list(range(len(FEATURES))),'all_without_executor_status':[i for i in range(len(FEATURES)) if i!=7]}


def episodes(prevalence, n, seed, noise=1.0):
    """Generate labels internally plus deliberately overlapping observed telemetry."""
    rng=np.random.default_rng(seed); y=rng.choice(5,size=n,p=prevalence); steps=rng.integers(3,9,size=n); pos=rng.uniform(.15,.95,size=n)
    # Index source: 0 success, 1 G, 2 V, 3 X, 4 E.  Means overlap by design.
    means=np.array([[3.8,.48,.43,.42,.82,.78],[1.9,.35,.60,.38,.52,.35],[3.0,.15,.84,.73,.48,.30],[3.4,.43,.52,.45,.42,.35],[3.3,.40,.57,.50,.36,.38]])
    candidate=np.clip(rng.normal(means[y,0],.95*noise),1,6)
    margin=np.clip(rng.normal(means[y,1],.18*noise),.01,.99)
    entropy=np.clip(rng.normal(means[y,2],.18*noise),.01,1.2)
    disagree=np.clip(rng.normal(means[y,3],.20*noise),0,1)
    committed=np.mod(rng.integers(0,4,size=n)+(y==2),4)
    status_probs=np.array([[.85,.12,.03],[.52,.31,.17],[.60,.25,.15],[.18,.28,.54],[.34,.39,.27]])
    status=np.array([rng.choice(3,p=status_probs[source]) for source in y])
    latency=np.clip(rng.normal(means[y,4],.28*noise),.05,2.0)
    delta=np.clip(rng.normal(means[y,5],.24*noise),0,1)
    transition=np.array([rng.random()<p for p in [.94 if s==0 else .32 if s==1 else .27 if s==2 else .36 if s==3 else .33 for s in y]],dtype=float)
    envwarn=np.array([rng.random()<p for p in [.10 if s==0 else .18 if s==1 else .16 if s==2 else .28 if s==3 else .54 for s in y]],dtype=float)
    # Prior history is noisy and does not reveal current source.
    prior=np.clip(rng.normal(np.cumsum(y!=0)/(np.arange(n)+1),.10*noise),0,1)
    x=np.column_stack([steps,pos,candidate,margin,entropy,disagree,committed,status,latency,delta,transition,envwarn,prior]).astype(float)
    return x,y


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def evaluate(regime, train_x,train_y,test_x,test_y,seed,context, attr_rows, reliability_rows, delay_rows):
    models=fit_models(train_x,train_y,seed)
    model_prob={name:probabilities(model,test_x) for name,model in models.items()}
    for name,p in model_prob.items():
        row={'context':context,'regime':regime,'model':name,**attribution_metrics(test_y,p)}; attr_rows.append(row)
    rf=model_prob['random_forest']
    perfect=np.eye(5)[test_y]
    for delay in (0,1,3,5):
        method_data={'STATIC':rf,'NAIVE':rf,'ORACLE':perfect,'PREDICTED':rf,'GATED':rf}
        for method,p in method_data.items():
            result=run_updates(test_y,p,delay,method)
            row={'context':context,'regime':regime,'delay':delay,'method':method,**result};reliability_rows.append(row)
            if regime=='A_balanced' and context=='id': delay_rows.append(row)
    return rf, model_prob


def main():
    OUT.mkdir(parents=True,exist_ok=True); attr=[]; reliability=[]; delay=[]; ablation=[]
    saved={}
    for name,(prevalence,seed) in REGIMES.items():
        xtr,ytr=episodes(prevalence,6000,seed); xv,yv=episodes(prevalence,2000,seed+1); xt,yt=episodes(prevalence,5000,seed+2)
        # Validation exists solely as a fixed design check; no model choice or
        # thresholds are changed from it.
        rf,probs=evaluate(name,xtr,ytr,xt,yt,seed,'id',attr,reliability,delay); saved[name]=(xtr,ytr,xt,yt,rf)
    # Predeclared balanced-feature ablations, with RF fixed a priori.
    xtr,ytr,xt,yt,_=saved['A_balanced']
    for label,indices in FEATURE_GROUPS.items():
        model=fit_models(xtr[:,indices],ytr,2100+len(indices))['random_forest']; p=probabilities(model,xt[:,indices]); am=attribution_metrics(yt,p)
        naive=run_updates(yt,p,0,'NAIVE')['mean_online_abs_error']; oracle=run_updates(yt,np.eye(5)[yt],0,'ORACLE')['mean_online_abs_error']; predicted=run_updates(yt,p,0,'PREDICTED')['mean_online_abs_error']
        recovery=(naive-predicted)/(naive-oracle) if naive>oracle else 0.0
        ablation.append({'label':label,'macro_f1':am['macro_f1'],'accuracy':am['accuracy'],'selector_f1':am['selector_f1'],'naive_error':naive,'oracle_error':oracle,'predicted_error':predicted,'recovery_ratio':recovery})
    # OOD: A-trained attribution sees C prevalence plus degraded telemetry.
    xtr,ytr,_,_,_=saved['A_balanced']; xo,yo=episodes(REGIMES['C_high_non_selector'][0],5000,1501,noise=1.45)
    evaluate('C_high_non_selector',xtr,ytr,xo,yo,1501,'ood_A_to_C_noisy',attr,reliability,delay)
    # Random/uniform attribution is audit-only; it should not match learned telemetry.
    uniform=np.full((len(yt),5),.2); random_a=attribution_metrics(yt,uniform)
    audit={'single_feature_macro_f1':{},'uniform_random_attribution':random_a,'design':'No latent source/injection flag is included in FEATURES. Execution status deliberately overlaps executor and environment sources.'}
    for index,feature in enumerate(FEATURES):
        model=fit_models(xtr[:,[index]],ytr,3000+index)['logistic']; audit['single_feature_macro_f1'][feature]=attribution_metrics(yt,probabilities(model,xt[:,[index]]))['macro_f1']
    audit['max_single_feature_macro_f1']=max(audit['single_feature_macro_f1'].values())
    audit['obvious_execution_feature_ablation_macro_f1']=next(row['macro_f1'] for row in ablation if row['label']=='execution')
    audit['all_without_executor_status_macro_f1']=next(row['macro_f1'] for row in ablation if row['label']=='all_without_executor_status')
    write_csv(OUT/'attribution_metrics.csv',attr);write_csv(OUT/'reliability_results.csv',reliability);write_csv(OUT/'regime_results.csv',[r for r in reliability if r['delay']==0]);write_csv(OUT/'delay_results.csv',delay);write_csv(OUT/'ood_results.csv',[r for r in reliability if r['context'].startswith('ood')]);write_csv(OUT/'ablation_results.csv',ablation)
    confusion(next(r['confusion'] for r in attr if r['context']=='id' and r['regime']=='A_balanced' and r['model']=='random_forest'),OUT/'attribution_confusion_matrix.png')
    lines([r for r in reliability if r['context']=='id' and r['delay']==0],'regime','mean_online_abs_error','method','Reliability adaptation by regime','Failure-prevalence regime','Mean online absolute error',OUT/'reliability_comparison.png')
    scatter(ablation,OUT/'oracle_recovery_curve.png')
    lines(delay,'delay','mean_online_abs_error','method','Balanced-regime feedback-delay sensitivity','Feedback delay (decisions)','Mean online absolute error',OUT/'delay_sensitivity.png')
    lines([r for r in reliability if r['delay']==0 and r['method'] in ('NAIVE','ORACLE','PREDICTED')],'context','mean_online_abs_error','method','In-distribution vs OOD reliability adaptation','Evaluation context','Mean online absolute error',OUT/'distribution_shift.png')
    (OUT/'leakage_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    # Recovery over four ID regimes using preregistered RF, delay 0.
    recovery=[]
    for regime in REGIMES:
        items={r['method']:r for r in reliability if r['context']=='id' and r['regime']==regime and r['delay']==0}
        oracle_gain=items['NAIVE']['mean_online_abs_error']-items['ORACLE']['mean_online_abs_error']
        predicted_gain=items['NAIVE']['mean_online_abs_error']-items['PREDICTED']['mean_online_abs_error']
        # Do not force a ratio when the preregistered denominator is zero.
        ratio=None if abs(oracle_gain) < 1e-12 else predicted_gain/oracle_gain
        recovery.append({'regime':regime,'recovery_ratio':ratio,'oracle_gain':oracle_gain,'predicted_gain':predicted_gain,'macro_f1':next(r['macro_f1'] for r in attr if r['context']=='id' and r['regime']==regime and r['model']=='random_forest')})
    (OUT/'recovery_summary.json').write_text(json.dumps(recovery,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','regimes':list(REGIMES),'outputs':str(OUT),'balanced_rf_macro_f1':recovery[0]['macro_f1']},indent=2))

if __name__=='__main__': main()
