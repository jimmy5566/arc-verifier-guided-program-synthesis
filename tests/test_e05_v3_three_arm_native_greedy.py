import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e05_v3_three_arm_native_greedy import validate_preimport
def test_preimport_contract():
    result=validate_preimport()
    assert result['rows']==768 and len(result['arms'])==3 and not result['model_imported']
if __name__=='__main__':
    test_preimport_contract()
    print('PASS_E05_V3_THREE_ARM_CONTRACT')
