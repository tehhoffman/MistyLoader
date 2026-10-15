#!/usr/bin/env python3
"""Summarize the final decode window in an Nsight Systems SQLite export."""
import argparse
import json
from pathlib import Path
import sqlite3


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlap_summary(db, start, end, tokens):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='NVTX_EVENTS'").fetchone():
        return None
    cpu = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='mistral.cpu.experts' "
                     'AND start>=? AND end<=? ORDER BY start', (start, end)).fetchall()
    gpu = []
    for table in ['CUPTI_ACTIVITY_KIND_GRAPH_TRACE', 'CUPTI_ACTIVITY_KIND_KERNEL']:
        if db.execute('SELECT 1 FROM sqlite_master WHERE name=?', (table,)).fetchone():
            gpu += db.execute(f'SELECT max(start,?),min(end,?) FROM {table} '
                              'WHERE end>? AND start<?', (start, end, start, end)).fetchall()
    # Union both devices and graph/kernel records so overlapping intervals are counted once.
    gpu = merge_intervals(gpu)
    total_overlap = 0
    overlapping_ranges = 0
    index = 0
    for a, b in cpu:
        while index < len(gpu) and gpu[index][1] <= a:
            index += 1
        j, overlap = index, 0
        while j < len(gpu) and gpu[j][0] < b:
            overlap += max(0, min(b, gpu[j][1]) - max(a, gpu[j][0]))
            j += 1
        total_overlap += overlap
        overlapping_ranges += overlap > 0
    return {'cpu_ranges': len(cpu), 'cpu_ranges_overlapping_gpu_compute': overlapping_ranges,
            'cpu_expert_ms_per_step': sum(b-a for a, b in cpu) / 1e6 / tokens,
            'cpu_gpu_compute_overlap_ms_per_step': total_overlap / 1e6 / tokens,
            'gpu_compute_union_ms_per_step': sum(b-a for a, b in gpu) / 1e6 / tokens,
            'definition': 'Intersection of owned CPU expert NVTX ranges with actual CUDA graph/kernel intervals, excluding DMA-only overlap.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace', type=Path)
    ap.add_argument('--tokens', type=int, default=128)
    args = ap.parse_args()
    if args.tokens < 1:
        ap.error('--tokens must be positive')
    db = sqlite3.connect(args.trace.resolve().as_uri() + '?mode=ro', uri=True)
    # This target exports one F32 logits row (131072 vocabulary entries) per decode step.
    logits = db.execute('SELECT end FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                        'WHERE bytes=524288 AND copyKind=2 ORDER BY end').fetchall()
    if len(logits) <= args.tokens:
        raise ValueError('not enough logits transfers for the requested decode window')
    start, end = logits[-args.tokens - 1][0], logits[-1][0]
    runtime = db.execute('SELECT s.value, count(*), sum(r.end-r.start)/1e6 '
                         'FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId '
                         'WHERE r.start>=? AND r.end<=? GROUP BY s.value ORDER BY 3 DESC', (start, end)).fetchall()
    copies = db.execute('SELECT copyKind, count(*), sum(bytes), sum(end-start)/1e6 '
                        'FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? GROUP BY copyKind',
                        (start, end)).fetchall()
    kernels = db.execute('SELECT s.value, count(*), sum(k.end-k.start)/1e6 '
                         'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName '
                         'WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY 3 DESC', (start, end)).fetchall()
    graphs = []
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_GRAPH_TRACE'").fetchone():
        graphs = db.execute('SELECT deviceId, count(*), sum(end-start)/1e6 '
                            'FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE start>=? AND end<=? GROUP BY deviceId',
                            (start, end)).fetchall()
    result = {
        'trace': str(args.trace), 'decode_steps': args.tokens,
        'window_start_ns': start, 'window_end_ns': end, 'window_ms': (end - start) / 1e6,
        'profiled_ms_per_step': (end - start) / 1e6 / args.tokens,
        'runtime': [{'name': n, 'calls_per_step': c / args.tokens, 'ms_per_step': t / args.tokens}
                    for n, c, t in runtime],
        'copies': [{'kind': k, 'count_per_step': c / args.tokens, 'bytes_per_step': b / args.tokens,
                    'ms_per_step': t / args.tokens} for k, c, b, t in copies],
        'kernels': [{'name': n, 'calls_per_step': c / args.tokens, 'ms_per_step': t / args.tokens}
                    for n, c, t in kernels],
        'cuda_graphs': [{'device': d, 'calls_per_step': c / args.tokens, 'ms_per_step': t / args.tokens}
                        for d, c, t in graphs],
        'expert_overlap': overlap_summary(db, start, end, args.tokens),
        'limitations': ['Profiler overhead is included; use unprofiled runs for throughput.',
                       'Assumes the last request has more generated tokens than this window, no speculation, and this vocabulary.',
                       'Runtime and GPU times overlap and must not be added; sync time includes work being waited for.'],
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
