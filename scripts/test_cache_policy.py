#!/usr/bin/env python3
"""Compare the MistyLoader cache planner with an independent routing replay."""
import csv
import json
from pathlib import Path
import subprocess
import tempfile

from cache_policy_replay import replay

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'vendor/llama.cpp/src/llama-moe-cache.cpp').read_text()
planner = source[source.index('struct moe_cache_lru {'):source.index('// gate, up, down or gate_up, down')]
harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#define GGML_ASSERT assert
PLANNER
int main(int argc, char ** argv) {
    const bool lfu = std::string(argv[1]) == "lfu";
    const int period = std::stoi(argv[2]);
    const int warm = std::stoi(argv[3]);
    std::map<int, moe_cache_lru> groups;
    std::map<int, std::vector<std::vector<int32_t>>> maps;
    uint64_t hit = 0, selected = 0, gpu = 0, bytes = 0, cpu = 0, steps = 0;
    std::string line;
    std::getline(std::cin, line);
    while (std::getline(std::cin, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int il, step, ig, capacity;
        uint64_t expert_bytes;
        row >> il >> step >> ig >> capacity >> expert_bytes;
        auto & g = groups[ig];
        if (!g.n_slots) {
            g.init(36, 128, capacity);
            auto & m = maps[ig];
            m.assign(36, std::vector<int32_t>(128, -1));
            for (int i = 0; i < 36; ++i) g.slot_map[i] = m[i].data();
        }
        std::vector<int32_t> ids;
        int id;
        while (row >> id) ids.push_back(id);
        std::vector<moe_cache_lru::fill> fills;
        size_t hits;
        assert(g.plan(il, ids.data(), ids.size(), fills, hits,
                      step < warm || (period > 0 && step % period == 0), lfu));
        size_t resident = 0;
        for (auto e : g.uniq) resident += g.slot_map[il][e] >= 0;
        assert(resident == hits + fills.size());
        for (int s = 0; s < capacity; ++s) {
            const int key = g.key_of[s];
            if (key >= 0) assert(g.slot_map[key / 128][key % 128] == s);
        }
        for (int layer = 0; layer < 36; ++layer) {
            const auto & m = maps[ig][layer];
            for (int e = 0; e < 128; ++e) {
                if (m[e] >= 0) assert(g.key_of[m[e]] == layer*128 + e);
            }
        }
        if (step >= warm) {
            hit += hits; selected += g.uniq.size(); gpu += resident;
            bytes += fills.size()*expert_bytes;
            cpu += (g.uniq.size()-resident)*expert_bytes;
            steps++;
        }
    }
    std::cout << hit << ' ' << selected << ' ' << gpu << ' ' << bytes << ' ' << cpu << ' ' << steps << '\n';
}
'''.replace('PLANNER', planner)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace', type=Path)
    args = ap.parse_args()
    trace = args.trace.read_text()
    rows = list(csv.DictReader(trace.splitlines()))
    with tempfile.TemporaryDirectory(prefix='mistral-cache-test-') as d:
        src, binary = Path(d)/'test.cpp', Path(d)/'test'
        src.write_text(harness)
        subprocess.run(['c++', '-O2', '-std=c++17', str(src), '-o', str(binary)], check=True)
        for policy, period in [('frozen-lru', 0), ('frozen-lfu', 0), ('adaptive-lru', 8), ('adaptive-lfu', 4)]:
            for warm in [0, 64]:
                p = subprocess.run([str(binary), 'lfu' if 'lfu' in policy else 'lru', str(period), str(warm)],
                                   input=trace, capture_output=True, text=True, check=True)
                hit, selected, gpu, uploaded, cpu, steps = map(int, p.stdout.split())
                observed = {'hit_pct': 100*hit/selected, 'gpu_pct': 100*gpu/selected,
                            'upload_mib_per_token': uploaded/2**20/(steps/36),
                            'cpu_mib_per_token': cpu/2**20/(steps/36), 'layer_steps': steps}
                expected = replay(rows, policy, warm=warm, period=max(1, period))['post_warm']
                assert observed == expected, (policy, warm, observed, expected)
                print(json.dumps({'policy': policy, 'warm': warm, 'status': 'pass', **observed}), flush=True)


if __name__ == '__main__':
    main()
