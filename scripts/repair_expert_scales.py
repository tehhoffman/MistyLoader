#!/usr/bin/env python3
"""Repair omitted FP8 expert scales in a separate copy of the identified Q4 artifact.

This script is specific to the checksum below. It scales the Q4_K d/dmin and
Q6_K d fields, retaining the quantized values and all non-expert tensor bytes.
The extra FP16 rounding makes this an experimental repair, not a re-conversion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from gguf_header import read_header

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads((ROOT / 'configs/model-provenance.json').read_text())
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('source', type=Path, help='Original GGUF matching the recorded source SHA256.')
ap.add_argument('--output', type=Path, default=ROOT / 'models/mistral-small-4-uncensored-scale-repaired-Q4_K_M.gguf')
ap.add_argument('--scales', type=Path, default=ROOT / 'artifacts/original-expert-scales.json')
args = ap.parse_args()
SOURCE = args.source.expanduser().resolve(strict=True)
EXPECTED = PROVENANCE['source_sha256']
OUT = args.output.expanduser().resolve()
PART = OUT.with_suffix('.gguf.partial')
MANIFEST = OUT.with_suffix('.repair.json')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


if OUT.exists() or PART.exists():
    raise FileExistsError('repair output already exists; do not apply the scales twice')
print('Verifying the source checksum before repair...', flush=True)
if digest(SOURCE) != EXPECTED:
    raise ValueError('source checksum mismatch')
header = read_header(SOURCE)
scales = json.loads(args.scales.read_text())
if scales['repo'] != PROVENANCE['official_repository'] or scales['revision'] != PROVENANCE['official_revision']:
    raise ValueError('unexpected scale source revision')
assert len(scales['layers']) == 36
assert header['metadata']['mistral4.expert_count'] == 128
expert_names = {n for n in header['tensors'] if '_exps.' in n}
expected_names = {f'blk.{i}.ffn_{kind}_exps.weight' for i in range(36) for kind in ['gate_up', 'down']}
if expert_names != expected_names:
    raise ValueError('unexpected expert tensor set')
OUT.parent.mkdir(exist_ok=True)
print('Copying the model; the original stays unchanged...', flush=True)
shutil.copyfile(SOURCE, PART)
report = {'source': str(SOURCE), 'source_sha256': EXPECTED, 'output': str(OUT),
          'scales_source': {'repo': scales['repo'], 'revision': scales['revision']},
          'method': 'Multiply existing quantization scale fields by original per-expert FP8 scale; round to FP16.',
          'tensors': []}
for layer in scales['layers']:
    il = layer['layer']
    for kind in ['gate_up', 'down']:
        name = f'blk.{il}.ffn_{kind}_exps.weight'
        t = header['tensors'][name]
        if t['shape'][2] != 128:
            raise ValueError('unexpected expert dimension')
        if t['type'] == 12:  # Q4_K
            block_size, field_offsets = 144, [0, 2]
        elif t['type'] == 14:  # Q6_K
            block_size, field_offsets = 210, [208]
        else:
            raise ValueError(f'unsupported quantization type: {t}')
        elements = int(np.prod(t['shape'][:2]))
        if elements % 256:
            raise ValueError('unaligned expert matrix')
        blocks = elements // 256
        row_bytes = blocks * block_size
        buffer = np.memmap(PART, mode='r+', dtype=np.uint8, offset=t['offset'], shape=(128 * row_bytes,))
        underflow = 0
        nonzero = 0
        for expert, scale in enumerate(layer[kind]):
            if not (0 < scale < 1):
                raise ValueError('invalid scale')
            for field in field_offsets:
                view = np.ndarray(shape=(blocks,), dtype='<f2', buffer=buffer,
                                  offset=expert * row_bytes + field, strides=(block_size,))
                values = view.astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f'nonfinite source scales: {name}, expert {expert}')
                corrected = (values * np.float32(scale)).astype(np.float16)
                if not np.isfinite(corrected).all():
                    raise ValueError('nonfinite repaired scales')
                nonzero += int(np.count_nonzero(values))
                underflow += int(np.count_nonzero((values != 0) & (corrected == 0)))
                view[:] = corrected
                del view
        buffer.flush()
        buffer._mmap.close()
        row = {'name': name, 'type': t['type'], 'bytes': 128 * row_bytes,
               'scale_fields_changed': 128 * blocks * len(field_offsets),
               'nonzero_source_scales': nonzero, 'underflow_to_zero': underflow}
        report['tensors'].append(row)
        print(f'Repaired {name}; underflow fields: {underflow}/{nonzero}', flush=True)

# Check every byte outside the expert tensors, including all metadata and padding.
intervals = sorted((header['tensors'][r['name']]['offset'], header['tensors'][r['name']]['offset'] + r['bytes']) for r in report['tensors'])
unchanged_bytes = 0
with SOURCE.open('rb') as original, PART.open('rb') as repaired:
    cursor = 0
    for start, end in intervals + [(SOURCE.stat().st_size, SOURCE.stat().st_size)]:
        original.seek(cursor)
        repaired.seek(cursor)
        remaining = start - cursor
        while remaining:
            n = min(16 * 1024 * 1024, remaining)
            if original.read(n) != repaired.read(n):
                raise ValueError('non-expert bytes changed')
            unchanged_bytes += n
            remaining -= n
        cursor = end
report['verified_unchanged_non_expert_bytes'] = unchanged_bytes
print('Hashing the repaired file...', flush=True)
report['output_sha256'] = digest(PART)
if report['output_sha256'] != PROVENANCE['tested_sha256']:
    raise ValueError('repaired checksum differs from the tested model; partial file retained')
PART.replace(OUT)
MANIFEST.write_text(json.dumps(report, indent=2) + '\n')
print(f'Completed experimental repair: {OUT}', flush=True)
