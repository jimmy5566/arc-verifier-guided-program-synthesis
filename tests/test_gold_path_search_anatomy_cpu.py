"""Exercise the actual post-freeze diagnostic reader; never load a model."""
import gzip
import json

from scripts.run_v7_ttt24_eval60_gold_path_diagnostic import search_anatomy, prefix_hash


def test_actual_search_reader_preserves_unknown_wait_and_wall(tmp_path):
    (tmp_path/'tables/candidate_index').mkdir(parents=True)
    (tmp_path/'raw').mkdir()
    index = {'source_archive_file':'raw/fixture.json.gz'}
    (tmp_path/'tables/candidate_index/a_o0.jsonl.gz').write_bytes(gzip.compress((json.dumps(index)+'\n').encode()))
    cell = {'augmentation_id':'geom=identity__color=id__order=canonical',
            'nodes':[{'node_id':0,'parent_node_id':None,'selected_token':None,'state':'root'},
                     {'node_id':1,'parent_node_id':0,'selected_token':1,'state':'expanded'}],
            'diagnostic_trace':{'search_trace':[{'prefix_hash':prefix_hash((1,)),
                 'frontier_insert_order':99,'frontier_pop_order':3,
                 'elapsed_seconds':5,'nodes_expanded_so_far':3}]}}
    (tmp_path/'raw/fixture.json.gz').write_bytes(gzip.compress(json.dumps({'cells':{'identity':cell}}).encode()))
    rows = search_anatomy(tmp_path,['a:o0'],{'a:o0':[[1]]},'base')
    assert len(rows)==2
    assert rows[0]['search_state']=='GOLD_SUCCESSOR_EXPANDED'
    assert rows[0]['frontier_wait'] is None
    assert rows[0]['frontier_wait_seconds'] is None
    assert rows[0]['frontier_wait_status']=='UNAVAILABLE_INSERTION_CLOCK_NOT_PRESERVED'
    assert rows[0]['wall_censored'] is None
    assert rows[1]['search_state']=='UNOBSERVED_OR_BUDGET_CENSORED'
