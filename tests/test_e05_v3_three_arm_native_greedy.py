import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e05_v3_three_arm_native_greedy import validate_preimport
from scripts.run_e05_v3_three_arm_native_greedy import E04E_ADAPTER_CONFIG_SHA256, verify_native_config
def test_preimport_contract():
    result=validate_preimport()
    assert result['rows']==768 and len(result['arms'])==3 and not result['model_imported']
def test_frozen_native_config_identity():
    result=verify_native_config()
    assert result['config_dir']=='configs/nvarc_native_846d0198'
    assert result['identity_sha256']
def test_e04e_adapter_config_identities_are_pinned():
    assert E04E_ADAPTER_CONFIG_SHA256 == {
        'E04E_RUN003_CONTROL_EQUAL_SLOT':'aa2c74857b1ff6e1738d5723259ce9ffd698e783e40568faba6aaf42463cfe65',
        'E04E_RUN003_TREATMENT_EQUAL_SLOT':'9f9ec9013387e85985bdc33a1de15ca12b5bb4b9ffc22f9b876ce513bd006f19',
    }
if __name__=='__main__':
    test_preimport_contract()
    test_frozen_native_config_identity()
    test_e04e_adapter_config_identities_are_pinned()
    print('PASS_E05_V3_THREE_ARM_CONTRACT')
