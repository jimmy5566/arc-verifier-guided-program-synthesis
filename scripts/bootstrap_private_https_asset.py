#!/usr/bin/env python3
"""Download a private HTTPS asset to a verified cache and atomically publish it."""
from __future__ import annotations
import argparse, hashlib, os, tempfile, urllib.parse, urllib.request
from pathlib import Path

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def fetch(*, url: str, revision: str, expected_sha256: str, cache_dir: Path, destination: Path, token_env: str) -> dict:
    parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or not parsed.netloc: raise RuntimeError('PRIVATE_ASSET_HTTPS_REQUIRED')
    if not revision: raise RuntimeError('PRIVATE_ASSET_PINNED_REVISION_REQUIRED')
    if len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256): raise RuntimeError('PRIVATE_ASSET_SHA256_INVALID')
    cache_dir.mkdir(parents=True,exist_ok=True); cache=cache_dir/expected_sha256
    if cache.is_file() and sha256_file(cache)==expected_sha256: source='CACHE'
    else:
        token=os.environ.get(token_env)
        if not token: raise RuntimeError('PRIVATE_ASSET_READONLY_CREDENTIAL_MISSING')
        fd,tmp=tempfile.mkstemp(prefix='.asset.',suffix='.tmp',dir=cache_dir)
        try:
            request=urllib.request.Request(url,headers={'Authorization':f'Bearer {token}','Accept':'application/octet-stream'})
            with os.fdopen(fd,'wb') as out, urllib.request.urlopen(request,timeout=60) as response:
                while True:
                    block=response.read(1024*1024)
                    if not block: break
                    out.write(block)
                out.flush(); os.fsync(out.fileno())
            if sha256_file(Path(tmp))!=expected_sha256: raise RuntimeError('PRIVATE_ASSET_SHA256_MISMATCH')
            os.replace(tmp,cache); source='HTTPS'
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    if destination.exists() and sha256_file(destination)!=expected_sha256: raise RuntimeError('PRIVATE_ASSET_DESTINATION_CONFLICT')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        fd,tmp=tempfile.mkstemp(prefix='.publish.',suffix='.tmp',dir=destination.parent)
        try:
            with os.fdopen(fd,'wb') as out, cache.open('rb') as inp:
                for block in iter(lambda:inp.read(1024*1024),b''): out.write(block)
                out.flush(); os.fsync(out.fileno())
            if sha256_file(Path(tmp))!=expected_sha256: raise RuntimeError('PRIVATE_ASSET_PUBLISH_HASH_MISMATCH')
            os.replace(tmp,destination)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    return {'status':'PRIVATE_ASSET_READY','source':source,'revision':revision,'sha256':expected_sha256,'bytes':destination.stat().st_size,'credential_present':bool(os.environ.get(token_env))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--url',required=True);p.add_argument('--revision',required=True);p.add_argument('--sha256',required=True);p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--token-env',default='GH_TOKEN');a=p.parse_args()
    print(fetch(url=a.url,revision=a.revision,expected_sha256=a.sha256,cache_dir=a.cache_dir,destination=a.destination,token_env=a.token_env))
if __name__=='__main__': main()
