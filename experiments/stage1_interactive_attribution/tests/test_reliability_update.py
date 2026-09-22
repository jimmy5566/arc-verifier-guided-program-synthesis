from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reliability_update import run_updates

def test_oracle_avoids_generator_negative_updates():
    # One selector error, one generator limitation, one selector-correct success.
    y=np.array([1,2,0]);p=np.eye(5)[y]
    naive=run_updates(y,p,0,'NAIVE');oracle=run_updates(y,p,0,'ORACLE')
    assert oracle['final_abs_error'] < naive['final_abs_error']
    assert oracle['updates']==2 and naive['updates']==3

def test_delay_changes_online_not_final_oracle_evidence():
    y=np.array([0,2,3,0]);p=np.eye(5)[y]
    immediate=run_updates(y,p,0,'ORACLE');delayed=run_updates(y,p,3,'ORACLE')
    assert immediate['final_estimate']==delayed['final_estimate']
    # Delay changes the online evidence path; a particular short sequence need
    # not be monotone because the shared prior may temporarily be closer to the
    # target than an early update.
    assert delayed['mean_online_abs_error'] != immediate['mean_online_abs_error']
