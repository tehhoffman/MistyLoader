#!/usr/bin/env python3
"""Fetch only the original FP8 expert scales, with a pinned source revision."""
import concurrent.futures
import json
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads((ROOT / 'configs/model-provenance.json').read_text())
REPO = PROVENANCE['official_repository']
REV = PROVENANCE['official_revision']
(ROOT / 'artifacts').mkdir(exist_ok=True)
index_url = f'https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors.index.json'
with urllib.request.urlopen(index_url, timeout=45) as response:
    index_bytes = response.read(4 * 1024 * 1024 + 1)
if len(index_bytes) > 4 * 1024 * 1024:
    raise ValueError('unexpected model index size')
INDEX = json.loads(index_bytes)['weight_map']


def fetch(file, start, end):
    url = f'https://huggingface.co/{REPO}/resolve/{REV}/{file}?scale_range={start}-{end}'
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=45) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
            raise RuntimeError('range not honored')
        data = response.read(end - start + 2)
    if len(data) != end - start + 1:
        raise RuntimeError('incomplete range')
    return data


def header(file):
    n = struct.unpack('<Q', fetch(file, 0, 7))[0]
    if n > 4 * 1024 * 1024:
        raise ValueError('unexpected header size')
    h = json.loads(fetch(file, 8, n + 7))
    (ROOT / 'artifacts' / (file + '.header.json')).write_text(json.dumps(h, indent=2))
    return file, (n + 8, h)


with concurrent.futures.ThreadPoolExecutor(3) as pool:
    headers = dict(pool.map(header, sorted(set(INDEX.values()))))


def scales(layer):
    result = {'layer': layer}
    for kind in ['gate_up', 'down']:
        name = f'language_model.model.layers.{layer}.mlp.experts.{kind}_proj_scale_inv'
        file = INDEX[name]
        offset, h = headers[file]
        t = h[name]
        if t['shape'] != [128, 1, 1] or t['dtype'] != 'BF16':
            raise ValueError(f'unexpected scale layout: {name}: {t}')
        start, end = t['data_offsets']
        data = fetch(file, offset + start, offset + end - 1)
        vals = [struct.unpack('<f', struct.pack('<I', v << 16))[0] for v in struct.unpack('<128H', data)]
        if not all(0 < v < 1 for v in vals):
            raise ValueError(f'unexpected scale values: {name}')
        result[kind] = vals
    print(f'Fetched layer {layer} scales: gate_up[0]={result["gate_up"][0]:.9g}, down[0]={result["down"][0]:.9g}', flush=True)
    return result


with concurrent.futures.ThreadPoolExecutor(6) as pool:
    values = list(pool.map(scales, range(36)))
(ROOT / 'artifacts/original-expert-scales.json').write_text(json.dumps({'repo': REPO, 'revision': REV, 'layers': values}, indent=2) + '\n')
