#!/usr/bin/env python3
"""CPU tests for target-free tokenizer-only prompt-ID materialization."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import materialize_connected_component_prompt_token_ids_v1 as m

class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return '|'.join(f"{row['role']}:{row['content']}" for row in messages) + '|assistant:'
    def __call__(self, text, *, add_special_tokens, **kwargs):
        assert add_special_tokens is False
        return {'input_ids':[len(text), 7, 15]}

def task():
    return {'train':[{'input':[[1]],'output':[[2]]}], 'test':[{'input':[[3]]}]}

def main():
    assert 'transformers' not in sys.modules
    records=[{'cohort_record_id':'R:0','split':'OOD','prompt_task_sha256':m.canonical_sha256(task()),'task':task()}]
    out=m.materialize_records(records,FakeTokenizer())
    assert out[0]['prompt_token_ids']==[len('user:1|assistant:2|user:3|assistant:'),7,15]
    assert out[0]['prompt_task_sha256']==records[0]['prompt_task_sha256']
    frozen=m.load_target_free_records(m.DEFAULT_INPUT)
    assert len(frozen)==72 and len(m.materialize_records(frozen,FakeTokenizer()))==72
    with tempfile.TemporaryDirectory() as raw:
        root=Path(raw)
        identity={'tokenizer_files':{}}
        for name,payload in {'tokenizer.json':b'a','tokenizer_config.json':b'b'}.items():
            (root/name).write_bytes(payload); identity['tokenizer_files'][name]=hashlib.sha256(payload).hexdigest()
        assert m.verify_tokenizer_identity(root,identity)==identity['tokenizer_files']
        (root/'tokenizer.json').write_bytes(b'x')
        try: m.verify_tokenizer_identity(root,identity)
        except RuntimeError as exc: assert str(exc)=='TOKENIZER_FILE_SHA_MISMATCH:tokenizer.json'
        else: raise AssertionError('expected hash mismatch')
    bad={'target_content_present':False,'records':records*72}
    bad['records'][1]=dict(records[0])
    with tempfile.TemporaryDirectory() as raw:
        path=Path(raw)/'bad.json';path.write_text(json.dumps(bad))
        try: m.load_target_free_records(path)
        except RuntimeError as exc: assert str(exc)=='PROMPT_RECORD_DUPLICATE_OR_TASK_INVALID'
        else: raise AssertionError('expected duplicate rejection')
    assert 'transformers' not in sys.modules
    print('PASS_TOKENIZER_ONLY_PROMPT_ID_MATERIALIZATION_V1')
if __name__=='__main__': main()
