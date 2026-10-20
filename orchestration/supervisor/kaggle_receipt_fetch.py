"""Fetch named small receipts, not every installed-package output page.

Kaggle's current output API is latest-session-only. The caller separately
verifies the unique one-push kernel/version/source and the downloaded nonce.
No URL or credential is printed. This helper never launches a kernel.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


def fetch_named(api, kernel_ref, nonce, output, names, request_type, get,
                max_pages=8, max_file_bytes=16 * 1024 * 1024):
    assert re.fullmatch(r'jimmy5566/arc2-paired48-[a-z0-9-]+', kernel_ref)
    assert re.fullmatch(r'[0-9a-f]{32}', nonce)
    owner, slug = kernel_ref.split('/')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = set(names)
    for name in names:
        p = PurePosixPath(name)
        assert p.name == name and not p.is_absolute() and name not in {'.', '..'}
        assert name.endswith(('.json', '.txt', '.jsonl'))
    wanted = {f'arc2_paired48_{nonce}/{name}' for name in names}
    downloaded, token, seen = [], '', set()
    pages = 0
    with api.build_kaggle_client() as client:
        for _ in range(max_pages):
            request = request_type()
            request.user_name, request.kernel_slug = owner, slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            pages += 1
            for item in response.files or []:
                name = item.file_name
                if name not in wanted:
                    continue
                rel = PurePosixPath(name)
                target = output.joinpath(*rel.parts).resolve()
                assert target.is_relative_to(output), 'UNSAFE_OUTPUT_PATH'
                if target.exists():
                    # An existing immutable receipt is never silently replaced.
                    wanted.remove(name)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with get(item.url, stream=True, timeout=(10, 30)) as result:
                        result.raise_for_status()
                        with tempfile.NamedTemporaryFile('wb', dir=target.parent, delete=False) as f:
                            temporary = Path(f.name)
                            size = 0
                            for block in result.iter_content(65536):
                                size += len(block)
                                if size > max_file_bytes:
                                    raise ValueError('RECEIPT_SIZE_LIMIT')
                                f.write(block)
                        if name.endswith('.json'):
                            json.loads(temporary.read_bytes())
                        os.replace(temporary, target)
                        temporary = None
                    downloaded.append(name)
                    wanted.remove(name)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            if not wanted:
                break  # Do not enumerate dependency trees after finding receipts.
            token = response.next_page_token
            if not token or token in seen:
                break
            seen.add(token)
    return {'downloaded': downloaded, 'missing': sorted(wanted), 'pages': pages}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kernel-ref', required=True)
    p.add_argument('--nonce', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', action='append', default=[])
    args = p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    import requests
    api = KaggleApi()
    api.authenticate()
    result = fetch_named(api, args.kernel_ref, args.nonce, args.output,
                         args.name or ['TERMINAL_RECEIPT.json'],
                         ApiListKernelSessionOutputRequest, requests.get)
    print(json.dumps(result))
    return 2 if result['missing'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
