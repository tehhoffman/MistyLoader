#!/usr/bin/env python3
"""Replay recorded routing IDs; report cache traffic, not predicted token throughput."""
import argparse
from collections import Counter, OrderedDict
import csv
import json


def replay(rows, policy, warm=64, period=1, decay=0):
    groups = {}
    total = Counter()
    post = Counter()
    for r in rows:
        il, step, ig, capacity, size = (int(r[k]) for k in ['layer', 'step', 'group', 'slots', 'expert_bytes'])
        keys = [(il, int(e)) for e in dict.fromkeys(r['ids'].split())]
        g = groups.setdefault(ig, {'cache': OrderedDict(), 'freq': Counter(), 'decay_step': 0})
        cache, freq = g['cache'], g['freq']
        if decay and step // decay > g['decay_step']:
            for k in freq:
                freq[k] *= 0.5
            g['decay_step'] = step // decay
        freq.update(keys)
        before = sum(k in cache for k in keys)
        for k in keys:
            if k in cache:
                cache.move_to_end(k)
        upload = 0
        update = step < warm or (policy.startswith('adaptive') and step % period == 0)
        if update:
            for k in sorted(keys):
                if k in cache:
                    continue
                if len(cache) < capacity:
                    victim = None
                else:
                    available = [v for v in cache if v not in keys]
                    if not available:
                        continue
                    victim = min(available, key=lambda v: freq[v]) if 'lfu' in policy else available[0]
                    if 'lfu' in policy and freq[k] <= freq[victim]:
                        continue
                if victim is not None:
                    del cache[victim]
                cache[k] = None
                upload += size
        after = sum(k in cache for k in keys)
        sample = {'selected': len(keys), 'hits_before': before, 'gpu_experts': after,
                  'upload_bytes': upload, 'cpu_bytes': (len(keys) - after)*size, 'layer_steps': 1}
        total.update(sample)
        if step >= warm:
            post.update(sample)
    def report(x):
        # 36 layers per token for this model; trace includes startup/smoke.
        tokens = x['layer_steps'] / 36
        return {'hit_pct': 100*x['hits_before']/max(1,x['selected']),
                'gpu_pct': 100*x['gpu_experts']/max(1,x['selected']),
                'upload_mib_per_token': x['upload_bytes']/2**20/max(1,tokens),
                'cpu_mib_per_token': x['cpu_bytes']/2**20/max(1,tokens),
                'layer_steps': x['layer_steps']}
    return {'policy': policy, 'warm': warm, 'period': period, 'decay': decay,
            'all': report(total), 'post_warm': report(post)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace')
    args = ap.parse_args()
    with open(args.trace) as f:
        rows = list(csv.DictReader(f))
    results = [replay(rows, p) for p in ['frozen-lru', 'frozen-lfu']]
    for p in ['adaptive-lru', 'adaptive-lfu']:
        for interval in [4, 8, 16, 32]:
            for decay in ([0, 64, 128] if 'lfu' in p else [0]):
                results.append(replay(rows, p, period=interval, decay=decay))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
