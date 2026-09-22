from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from simulate import FEATURES, episodes

def test_episode_telemetry_has_no_latent_source_column_and_is_reproducible():
    x1,y1=episodes([.4,.15,.15,.15,.15],100,11);x2,y2=episodes([.4,.15,.15,.15,.15],100,11)
    assert x1.shape==(100,len(FEATURES));assert (x1==x2).all();assert (y1==y2).all()
    # No raw class label is returned in the telemetry matrix.
    assert x1.shape[1]==len(FEATURES)
