from scripts.e05_v3_three_arm_native_greedy import validate_preimport
def test_preimport_contract():
 r=validate_preimport(); assert r['rows']==768 and len(r['arms'])==3 and not r['model_imported']
if __name__=='__main__':
 test_preimport_contract(); print('PASS_E05_V3_THREE_ARM_CONTRACT')
