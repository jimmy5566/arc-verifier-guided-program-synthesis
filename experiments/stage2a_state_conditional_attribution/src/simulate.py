"""CPU-only Stage-2A state-conditional attribution and policy experiment."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
STATES=['A_high','B_medium','C_low','D_volatile']; STATE_P=np.array([.30,.25,.25,.20])
# source index success,G,V,X,E; frozen in preregistration
SOURCE_P=np.array([[.548,.10,.072,.14,.14],[.2125,.25,.1875,.20,.15],[.20,.20,.36,.08,.16],[.3575,.05,.1425,.30,.15]])
FEATURES=['state_A','state_B','state_C','state_D','candidate_count','margin','entropy','disagreement','executor_status','latency','state_delta','expected_transition','environment_warning','history']

def episodes(n,seed):
    rng=np.random.default_rng(seed); state=rng.choice(4,n,p=STATE_P); source=np.array([rng.choice(5,p=SOURCE_P[s]) for s in state])
    means=np.array([[3.8,.48,.43,.42,.78,.80],[1.9,.35,.60,.38,.52,.35],[3.0,.15,.84,.73,.48,.30],[3.4,.43,.52,.45,.42,.35],[3.3,.40,.57,.50,.36,.38]])
    cand=np.clip(rng.normal(means[source,0],.95),1,6); margin=np.clip(rng.normal(means[source,1],.18),.01,.99); ent=np.clip(rng.normal(means[source,2],.18),.01,1.2); dis=np.clip(rng.normal(means[source,3],.20),0,1)
    status_probs=np.array([[.85,.12,.03],[.52,.31,.17],[.60,.25,.15],[.18,.28,.54],[.34,.39,.27]])
    status=np.array([rng.choice(3,p=status_probs[x]) for x in source]);lat=np.clip(rng.normal(means[source,4],.28),.05,2);delta=np.clip(rng.normal(means[source,5],.24),0,1)
    trans=np.array([rng.random()<(.94 if x==0 else .32 if x==1 else .27 if x==2 else .36 if x==3 else .33) for x in source],float)
    warn=np.array([rng.random()<(.10 if x==0 else .18 if x==1 else .16 if x==2 else .28 if x==3 else .54) for x in source],float)
    history=np.clip(rng.normal(np.cumsum(source!=0)/(np.arange(n)+1),.10),0,1); onehot=np.eye(4)[state]
    x=np.column_stack([onehot,cand,margin,ent,dis,status,lat,delta,trans,warn,history]);return x,state,source

def proba(model,x):
    raw=model.predict_proba(x);out=np.zeros((len(x),5));out[:,model.classes_]=raw;return out

def ece(p,y):
    total=0
    for lo in np.linspace(0,.9,10):
        hi=lo+.1;m=(p>=lo)&(p<(hi if hi<1 else 1.00001))
        if m.any():total+=m.mean()*abs(p[m].mean()-y[m].mean())
    return float(total)

def update_policy(source,state,pred,method,threshold=.70,cost=.08):
    """Online state estimates; source is evaluation-only, never a model input."""
    alpha=np.ones(4);beta=np.ones(4);global_a=global_b=1.; records=[]
    clean=np.array([1-(SOURCE_P[s,2]/(1-SOURCE_P[s,1])) for s in range(4)])
    for i,(z,s,p) in enumerate(zip(source,state,pred)):
        if method=='GLOBAL_NAIVE': est=np.repeat(global_a/(global_a+global_b),4)
        else: est=alpha/(alpha+beta)
        reliability=est[s];call=reliability<threshold
        # Decision outcome: strong verifier can repair V only; other primary
        # sources still defeat final task outcome.
        if call and z==2: success=np.random.default_rng(900000+i).random()<.96
        else: success=(z==0)
        records.append({'state':STATES[s],'latent_source':int(z),'estimate_before':reliability,'clean_target':clean[s],'commit':not call,'strong_call':call,'success':success,'wrong_commit':bool((not call) and z==2),'unnecessary_call':bool(call and z in (0,3,4))})
        # Feedback is immediate. G is omitted from state-oracle selector evidence.
        if method in ('GLOBAL_NAIVE','STATE_NAIVE'):
            pos,neg=(1.,0.) if z==0 else (0.,1.)
        elif method=='ORACLE_STATE':
            if z==1: continue
            pos,neg=((1.,0.) if z!=2 else (0.,1.))
        else:
            q=p;pos,neg=float(q[0]+q[3]+q[4]),float(q[2])
        if method=='GLOBAL_NAIVE':global_a+=pos;global_b+=neg
        else:alpha[s]+=pos;beta[s]+=neg
    rows=[]
    for s,name in enumerate(STATES):
        group=[r for r in records if r['state']==name];final=(global_a/(global_a+global_b) if method=='GLOBAL_NAIVE' else alpha[s]/(alpha[s]+beta[s]))
        if not group:
            continue
        selector=np.array([r['latent_source']==2 for r in group],float);pselector=np.array([pred[i,2] for i,(st) in enumerate(state) if st==s])
        rows.append({'method':method,'state':name,'episode_count':len(group),'true_reliability':clean[s],'final_estimate':final,'abs_reliability_error':abs(final-clean[s]),'selector_brier':float(np.mean((pselector-selector)**2)),'selector_ece':ece(pselector,selector),'wrong_commit_rate':np.mean([r['wrong_commit'] for r in group]),'unnecessary_call_rate':np.mean([r['unnecessary_call'] for r in group]),'task_success_rate':np.mean([r['success'] for r in group]),'strong_call_rate':np.mean([r['strong_call'] for r in group]),'strong_verifier_cost':cost*np.mean([r['strong_call'] for r in group])})
    return rows

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True);xtr,strain,ytr=episodes(8000,2201);xv,sv,yv=episodes(2000,2202);xt,st,yt=episodes(12000,2203)
    model=RandomForestClassifier(n_estimators=160,min_samples_leaf=8,n_jobs=-1,random_state=2201,class_weight='balanced_subsample').fit(xtr,ytr);p=proba(model,xt)
    prior=np.bincount(strain,minlength=5)/len(strain); degraded=.5*p+.5*prior
    yhat=p.argmax(1);attr=[{'model':'random_forest','accuracy':accuracy_score(yt,yhat),'macro_f1':f1_score(yt,yhat,average='macro'),'selector_f1':f1_score(yt==2,yhat==2),'selector_auroc':roc_auc_score(yt==2,p[:,2]),'selector_brier':np.mean((p[:,2]-(yt==2))**2),'selector_ece':ece(p[:,2],yt==2)}]
    # State-only audit checks whether context prevalence itself decodes sources.
    stateonly=RandomForestClassifier(n_estimators=160,min_samples_leaf=8,n_jobs=-1,random_state=2204).fit(xtr[:,:4],ytr);ps=proba(stateonly,xt[:,:4]);attr.append({'model':'state_only_audit','accuracy':accuracy_score(yt,ps.argmax(1)),'macro_f1':f1_score(yt,ps.argmax(1),average='macro'),'selector_f1':f1_score(yt==2,ps.argmax(1)==2),'selector_auroc':roc_auc_score(yt==2,ps[:,2]),'selector_brier':np.mean((ps[:,2]-(yt==2))**2),'selector_ece':ece(ps[:,2],yt==2)})
    rows=[]
    for method,pred in [('GLOBAL_NAIVE',p),('STATE_NAIVE',p),('ORACLE_STATE',np.eye(5)[yt]),('PREDICTED_STATE',p),('PREDICTED_DEGRADED',degraded)]:rows.extend(update_policy(yt,st,pred,method))
    write(OUT/'attribution_metrics.csv',attr);write(OUT/'state_calibration_and_decisions.csv',rows)
    aggregate=[]
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method]
        weights=np.array([r['episode_count'] for r in group],float);weights/=weights.sum()
        aggregate.append({'method':method,'weighted_abs_reliability_error':float(np.dot(weights,[r['abs_reliability_error'] for r in group])),'weighted_wrong_commit_rate':float(np.dot(weights,[r['wrong_commit_rate'] for r in group])),'weighted_unnecessary_call_rate':float(np.dot(weights,[r['unnecessary_call_rate'] for r in group])),'weighted_task_success_rate':float(np.dot(weights,[r['task_success_rate'] for r in group])),'weighted_strong_verifier_cost':float(np.dot(weights,[r['strong_verifier_cost'] for r in group])),'cost_adjusted_success':float(np.dot(weights,[r['task_success_rate']-r['strong_verifier_cost'] for r in group]))})
    write(OUT/'aggregate_decision_results.csv',aggregate)
    # Summarize exact recovery on calibration error relative to State-Naive.
    summary=[]
    for name in STATES:
        d={r['method']:r for r in rows if r['state']==name};gain=d['STATE_NAIVE']['abs_reliability_error']-d['PREDICTED_STATE']['abs_reliability_error'];oracle=d['STATE_NAIVE']['abs_reliability_error']-d['ORACLE_STATE']['abs_reliability_error'];summary.append({'state':name,'predicted_recovery_ratio':gain/oracle if oracle else None,'predicted_vs_state_naive_success_delta':d['PREDICTED_STATE']['task_success_rate']-d['STATE_NAIVE']['task_success_rate'],'predicted_vs_state_naive_wrong_commit_delta':d['PREDICTED_STATE']['wrong_commit_rate']-d['STATE_NAIVE']['wrong_commit_rate']})
    (OUT/'recovery_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','rf_macro_f1':attr[0]['macro_f1'],'outputs':str(OUT)},indent=2))
if __name__=='__main__':main()
