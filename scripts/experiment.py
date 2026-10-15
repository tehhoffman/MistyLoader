#!/usr/bin/env python3
"""Run one isolated, reproducible Mistral server experiment with MistyLoader."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/mistral-small-4-uncensored-scale-repaired-Q4_K_M.gguf'
GPU_IDS = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1')


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def capture(argv):
    p = subprocess.run(argv, text=True, capture_output=True, timeout=15)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--server', type=Path, default=ROOT / 'build/mistyloader/bin/llama-server')
    ap.add_argument('--source', type=Path, default=ROOT / 'vendor/llama.cpp', help='Source checkout corresponding to --server.')
    ap.add_argument('--engine', choices=['llama', 'ik'], default='llama')
    ap.add_argument('--prompt-file', type=Path, help='Reuse exact benchmark tokens from an earlier prompt.json.')
    ap.add_argument('--ctx', type=int, default=65536)
    ap.add_argument('--cache-mib', type=int, default=0)
    ap.add_argument('--spec', choices=['none', 'ngram-simple', 'ngram-mod', 'draft-dflash'], default='none')
    ap.add_argument('--draft-model', type=Path)
    ap.add_argument('--draft-max', type=int, default=7)
    ap.add_argument('--draft-device', default='CUDA1')
    ap.add_argument('--draft-p-min', type=float, default=0.0, help='Truncate a learned proposal when its token confidence falls below this value.')
    ap.add_argument('--ngram-short', action='store_true', help='For ngram-simple: 4-token lookup, 8-token draft, 1 required hit.')
    ap.add_argument('--prompt-tokens', type=int, default=512)
    ap.add_argument('--workload', choices=['retrieval', 'story'], default='story')
    ap.add_argument('--story-suite', action='store_true', help='Cycle fantasy, science fiction, and detective prompts within one model session.')
    ap.add_argument('--predict', type=int, default=128)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--threads-batch', type=int, help='Prompt threads; defaults to generation threads.')
    ap.add_argument('--cpu-mask', help='llama CPU affinity mask, with strict placement.')
    ap.add_argument('--cpu-mask-batch', help='Separate affinity mask for prompt processing.')
    ap.add_argument('--ubatch', type=int, default=256)
    ap.add_argument('--kv', default='q8_0', choices=['f16', 'q8_0', 'q4_0'])
    ap.add_argument('--port', type=int, default=18090)
    ap.add_argument('--reverse-gpus', action='store_true', help='Put the GPU without the desktop first in CUDA device order.')
    ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--reuse-prefix', action='store_true', help='Also measure a repeated request with the prompt KV cache enabled.')
    ap.add_argument('--smoke-only', action='store_true')
    ap.add_argument('--gpu-layers', type=int, default=99)
    ap.add_argument('--flash', choices=['on', 'off'], default='on')
    ap.add_argument('--no-cuda-graphs', action='store_true')
    ap.add_argument('--pin-host', action='store_true', help='MistyLoader only: register cached expert source weights with CUDA.')
    ap.add_argument('--cuda-f32-down', action='store_true', help='MistyLoader only: cache down experts and use the F32 activation CUDA kernel.')
    ap.add_argument('--hybrid-cache', action='store_true', help='MistyLoader: GPU cache hits, CPU misses after a fixed cache warmup.')
    ap.add_argument('--cache-warm-steps', type=int, default=64, help='Per-layer cache updates before hybrid mode freezes residency.')
    ap.add_argument('--cache-policy', choices=['frozen', 'lfu', 'lru'], default='frozen', help='Hybrid cache residency policy.')
    ap.add_argument('--cache-refresh-steps', type=int, default=4, help='Per-layer interval between adaptive cache admissions.')
    ap.add_argument('--cache-max-tokens', type=int, default=1, help='Largest verification batch allowed to use the hybrid cache.')
    ap.add_argument('--cache-quant-batch', action='store_true', help='Experimental SM86 path: quantized gate/up CUDA kernel for verification batches up to eight tokens.')
    ap.add_argument('--cache-whole-ffn', action='store_true', help='MistyLoader experiment: finish CPU/GPU expert branches before merging.')
    ap.add_argument('--cpu-skip-masked', action='store_true', help='Skip CPU matvec setup when all selected expert IDs are masked.')
    ap.add_argument('--cpu-concurrent', action='store_true', help='Run whole CPU expert misses on an owned worker alongside GPU hits.')
    ap.add_argument('--cpu-concurrent-serial', action='store_true', help='Serial control using the same private CPU executor and buffers.')
    ap.add_argument('--trace-routing', action='store_true', help='Save decode routing IDs for offline cache-policy analysis.')
    args = ap.parse_args()
    if args.ngram_short and args.spec != 'ngram-simple':
        ap.error('--ngram-short requires --spec ngram-simple')
    if (args.spec == 'draft-dflash') != bool(args.draft_model):
        ap.error('--spec draft-dflash requires --draft-model, and vice versa')
    if not 1 <= args.draft_max <= 7:
        ap.error('--draft-max must be between 1 and 7 for this draft')
    if not 0.0 <= args.draft_p_min <= 1.0:
        ap.error('--draft-p-min must be between 0 and 1')
    if not args.name or Path(args.name).name != args.name:
        ap.error('name must be a single directory name')
    if args.prompt_tokens < 256 or args.prompt_tokens + args.predict + 16 > args.ctx:
        ap.error('prompt must be at least 256 tokens and fit in context with generation headroom')
    if min(args.reps, args.predict, args.threads, args.ubatch) < 1 or args.cache_mib < 0:
        ap.error('invalid experiment size')
    if args.hybrid_cache and (args.cache_mib == 0 or args.cache_warm_steps < 0):
        ap.error('--hybrid-cache requires a cache budget and nonnegative warmup steps')
    if args.cache_refresh_steps < 1 or (args.cache_policy != 'frozen' and not args.hybrid_cache):
        ap.error('adaptive cache requires --hybrid-cache and a positive refresh interval')
    if not 1 <= args.cache_max_tokens <= 32:
        ap.error('--cache-max-tokens must be between 1 and 32')
    if args.cache_quant_batch and not args.hybrid_cache:
        ap.error('--cache-quant-batch requires --hybrid-cache')
    if args.cache_whole_ffn and not (args.hybrid_cache and args.cuda_f32_down and args.cache_mib):
        ap.error('--cache-whole-ffn requires --hybrid-cache, --cuda-f32-down, and a cache budget')
    if args.cpu_skip_masked and not args.hybrid_cache:
        ap.error('--cpu-skip-masked requires --hybrid-cache')
    if args.cpu_concurrent and not args.cache_whole_ffn:
        ap.error('--cpu-concurrent requires --cache-whole-ffn')
    if args.cpu_concurrent_serial and not args.cpu_concurrent:
        ap.error('--cpu-concurrent-serial requires --cpu-concurrent')
    if args.cpu_concurrent and args.cpu_mask:
        ap.error('--cpu-mask is not yet supported by the owned concurrent CPU pool')
    if args.trace_routing and (args.engine != 'llama' or not args.cache_mib):
        ap.error('--trace-routing requires the MistyLoader with a cache budget')
    if args.story_suite and (args.prompt_file or args.reuse_prefix or args.workload != 'story'):
        ap.error('--story-suite requires story workload without --prompt-file or --reuse-prefix')
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    with (ROOT / 'artifacts/experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


def run(args):
    model = args.model.resolve(strict=True)
    service = capture(['systemctl', '--user', 'is-active', 'strata.service']) if shutil.which('systemctl') else {'stdout': ''}
    if service['stdout'].strip() == 'active':
        raise RuntimeError('Stop strata.service before running an experiment')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', args.port))
    out = ROOT / 'artifacts' / args.name
    out.mkdir(exist_ok=False)
    url = f'http://127.0.0.1:{args.port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, data=None, timeout=None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(url + path, data=body, headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=timeout or args.timeout) as response:
            return json.load(response)

    cmd = [str(args.server), '-m', str(model), '--host', '127.0.0.1', '--port', str(args.port),
           '-c', str(args.ctx), '-np', '1', '-ngl', str(args.gpu_layers), '--cpu-moe', '-sm', 'layer', '-ts', '1,1',
           '--fit', 'off', '-fa', args.flash, '-ctk', args.kv, '-ctv', args.kv,
           '-b', str(max(2048, args.ubatch)), '-ub', str(args.ubatch), '-t', str(args.threads), '-tb', str(args.threads),
           '--moe-cache-mib', str(args.cache_mib), '--cache-ram', '0', '--no-context-shift',
           '--jinja', '--spec-type', args.spec, '--perf', '-lv', '4']
    cmd[cmd.index('-tb') + 1] = str(args.threads_batch or args.threads)
    if args.engine == 'ik':
        if args.cache_mib or args.pin_host or args.cpu_mask or args.cpu_mask_batch or args.cuda_f32_down or args.hybrid_cache:
            raise ValueError('ik adapter does not support the llama cache or CPU mask options')
        for flag, count in [('--cpu-moe', 1), ('--fit', 2), ('--moe-cache-mib', 2), ('--perf', 1), ('-lv', 2)]:
            index = cmd.index(flag)
            del cmd[index:index + count]
        # This installed version's --cpu-moe pattern misses fused gate/up tensors.
        cmd += ['-ot', '.*ffn_.*_exps.*=CPU', '--reasoning', 'off']
    if args.cpu_mask:
        cmd += ['--cpu-mask', args.cpu_mask, '--cpu-strict', '1']
    if args.cpu_mask_batch:
        cmd += ['--cpu-mask-batch', args.cpu_mask_batch, '--cpu-strict-batch', '1']
    if args.gpu_layers == 0:
        cmd += ['--device', 'none', '--no-op-offload']
    if args.ngram_short:
        cmd += ['--spec-ngram-simple-size-n', '4', '--spec-ngram-simple-size-m', '8',
                '--spec-ngram-simple-min-hits', '1']
    if args.draft_model:
        cmd += ['--model-draft', str(args.draft_model.resolve(strict=True)),
                '--spec-draft-n-max', str(args.draft_max), '--spec-draft-n-min', '1',
                '--spec-draft-p-min', str(args.draft_p_min),
                '--spec-draft-device', args.draft_device, '--spec-draft-ngl', '99',
                '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0']
    env = os.environ.copy()
    env.pop('MISTRAL_CACHE_WHOLE_FFN', None)
    env.pop('MISTRAL_CPU_SKIP_MASKED', None)
    env.pop('MISTRAL_CPU_CONCURRENT', None)
    env.pop('MISTRAL_CPU_CONCURRENT_SERIAL', None)
    # Checkpoints must load their own libraries, not the build directory's RUNPATH.
    library_dir = str(args.server.resolve().parent)
    env['LD_LIBRARY_PATH'] = library_dir + (os.pathsep + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    gpu_ids = ','.join(reversed(GPU_IDS.split(','))) if args.reverse_gpus else GPU_IDS
    env['CUDA_VISIBLE_DEVICES'] = gpu_ids
    if args.pin_host:
        env['GGML_CUDA_REGISTER_HOST'] = '1'
    if args.cuda_f32_down:
        env['MISTRAL_CUDA_F32_DOWN'] = '1'
    if args.hybrid_cache:
        env['MISTRAL_CACHE_HYBRID'] = '1'
        env['MISTRAL_CACHE_WARM_STEPS'] = str(args.cache_warm_steps)
        env['MISTRAL_CACHE_POLICY'] = args.cache_policy
        env['MISTRAL_CACHE_REFRESH_STEPS'] = str(args.cache_refresh_steps)
        env['MISTRAL_CACHE_MAX_TOKENS'] = str(args.cache_max_tokens)
    if args.trace_routing:
        env['MISTRAL_CACHE_TRACE'] = str(out / 'routing.csv')
    if args.cache_quant_batch:
        env['MISTRAL_CACHE_QUANT_BATCH'] = '1'
    if args.cache_whole_ffn:
        env['MISTRAL_CACHE_WHOLE_FFN'] = '1'
    if args.cpu_skip_masked:
        env['MISTRAL_CPU_SKIP_MASKED'] = '1'
    if args.cpu_concurrent:
        env['MISTRAL_CPU_CONCURRENT'] = '1'
    if args.cpu_concurrent_serial:
        env['MISTRAL_CPU_CONCURRENT_SERIAL'] = '1'
    if args.no_cuda_graphs:
        env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    metadata = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'command': cmd, 'model_path': str(model), 'model_first_file_bytes': model.stat().st_size,
                'model_mtime_ns': model.stat().st_mtime_ns,
                'engine_commit': capture(['git', '-C', str(args.source), 'rev-parse', 'HEAD']),
                'engine_diff': capture(['git', '-C', str(args.source), 'diff', '--stat']),
                'gpu_order': gpu_ids, 'library_path': env['LD_LIBRARY_PATH'],
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'cuda_environment': {k: env.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'GGML_CUDA_REGISTER_HOST',
                                     'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_NO_PINNED', 'GGML_CUDA_P2P',
                                     'GGML_OP_OFFLOAD_MIN_BATCH', 'GGML_CUDA_ENABLE_UNIFIED_MEMORY',
                                     'MISTRAL_CUDA_F32_DOWN', 'MISTRAL_CACHE_HYBRID', 'MISTRAL_CACHE_WARM_STEPS',
                                     'MISTRAL_CACHE_POLICY', 'MISTRAL_CACHE_REFRESH_STEPS', 'MISTRAL_CACHE_MAX_TOKENS', 'MISTRAL_CACHE_QUANT_BATCH', 'MISTRAL_CACHE_WHOLE_FFN', 'MISTRAL_CPU_SKIP_MASKED', 'MISTRAL_CPU_CONCURRENT', 'MISTRAL_CPU_CONCURRENT_SERIAL', 'MISTRAL_CACHE_TRACE']},
                'before_gpu': capture(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv']),
                'before_memory': capture(['free', '-b']), 'requests': []}
    save(out / 'result.json', metadata)
    patch = capture(['git', '-C', str(args.source), 'diff'])['stdout']
    # Include new source files too, so experimental kernels are reproducible before staging.
    new_files = capture(['git', '-C', str(args.source), 'ls-files', '--others', '--exclude-standard'])['stdout']
    for name in new_files.splitlines():
        path = args.source / name
        if path.is_file() and path.suffix in {'.cu', '.cuh', '.cpp', '.h', '.hpp'} and path.stat().st_size < 1024 * 1024:
            patch += capture(['git', '-C', str(args.source), 'diff', '--no-index', '--', '/dev/null', name])['stdout']
    (out / 'engine.patch').write_text(patch)
    stop = threading.Event()

    def monitor():
        with (out / 'resources.jsonl').open('w') as f:
            while not stop.is_set():
                sample = {'time': time.time(), 'gpu': capture(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,power.draw,temperature.gpu', '--format=csv,noheader,nounits'])}
                if process.poll() is None:
                    try:
                        sample['process_status'] = Path(f'/proc/{process.pid}/status').read_text()
                        sample['process_io'] = Path(f'/proc/{process.pid}/io').read_text()
                    except (FileNotFoundError, ProcessLookupError, PermissionError) as e:
                        # /proc can disappear or revoke access while the server exits.
                        sample['process_monitor_error'] = str(e)
                f.write(json.dumps(sample) + '\n')
                f.flush()
                stop.wait(2)

    with (out / 'server.log').open('w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        observer = threading.Thread(target=monitor, daemon=True)
        observer.start()
        started = time.monotonic()
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited with {process.returncode}; see {out / "server.log"}')
                try:
                    if request('/health', timeout=2).get('status') == 'ok':
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() - started > 600:
                    raise TimeoutError('model did not become ready within 10 minutes')
                time.sleep(1)
            metadata['load_seconds'] = time.monotonic() - started
            print(f'{args.name}: loaded in {metadata["load_seconds"]:.1f}s', flush=True)
            save(out / 'props.json', request('/props'))
            smoke = request('/v1/chat/completions', {
                'messages': [{'role': 'user', 'content': 'What is the capital of France? Reply with only the city name.'}],
                'temperature': 0, 'max_tokens': 32, 'seed': 1234,
                'reasoning_effort': 'none', 'cache_prompt': False})
            save(out / 'smoke.json', smoke)
            smoke_text = smoke['choices'][0]['message'].get('content') or ''
            metadata['smoke_content'] = smoke_text
            metadata['smoke_pass'] = 'paris' in smoke_text.lower()
            print('Smoke:', repr(smoke_text), flush=True)
            if args.cache_whole_ffn:
                metadata['whole_ffn_active'] = b'whole-expert CPU/GPU branches enabled' in (out / 'server.log').read_bytes()
                if not metadata['whole_ffn_active']:
                    raise RuntimeError('whole-expert path was requested but its graph activation marker is absent')
            if not metadata['smoke_pass']:
                plain = request('/completion', {'prompt': 'The capital of France is', 'n_predict': 16,
                                'temperature': 0, 'cache_prompt': False, 'return_tokens': True})
                save(out / 'plain-smoke.json', plain)
                metadata['plain_smoke_content'] = plain.get('content')
                print('Plain completion:', repr(plain.get('content')), flush=True)
                raise RuntimeError('native chat smoke test failed; speed results would not be a usable baseline')
            if args.smoke_only:
                metadata['status'] = 'completed'
                return
            if args.prompt_file:
                fixture = json.loads(args.prompt_file.read_text())
                tokens, expected = fixture['tokens'], fixture['expected_retrieval']
                if len(tokens) != args.prompt_tokens:
                    raise ValueError('--prompt-tokens does not match --prompt-file')
            else:
                tokens, expected = make_prompt(request, args.prompt_tokens, args.workload)
            save(out / 'prompt.json', {'tokens': tokens, 'expected_retrieval': expected,
                                      'description': 'Synthetic varied records with three inserted facts; optional story continuation. Not a natural conversation quality benchmark.'})
            metadata['prompt_sha256'] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            fixtures = [(tokens, expected)]
            if args.story_suite:
                fixtures += [make_prompt(request, args.prompt_tokens, args.workload, theme)
                             for theme in ['station', 'detective']]
                for i, (suite_tokens, suite_expected) in enumerate(fixtures):
                    save(out / f'prompt-suite-{i}.json', {'tokens': suite_tokens, 'expected_retrieval': suite_expected})
            for rep in range(args.reps + int(args.reuse_prefix)):
                tokens, expected = fixtures[rep % len(fixtures)]
                reuse = rep == args.reps
                body = {'prompt': tokens, 'n_predict': args.predict, 'temperature': 0,
                        'seed': 1234, 'cache_prompt': reuse, 'return_tokens': True,
                        'ignore_eos': False, 'stream': False}
                wall = time.monotonic()
                response = request('/completion', body)
                elapsed = time.monotonic() - wall
                save(out / f'response-{rep}.json', response)
                content = response.get('content', '')
                summary = {'rep': rep, 'reuse_prefix': reuse, 'wall_seconds': elapsed,
                           'prompt_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                           'timings': response.get('timings'), 'tokens_evaluated': response.get('tokens_evaluated'),
                           'tokens_predicted': response.get('tokens_predicted'),
                           'truncated': response.get('truncated'),
                           'retrieval_all_codes_present': all(v in content for v in expected.values()),
                           'content': content, 'tokens': response.get('tokens')}
                metadata['requests'].append(summary)
                save(out / 'result.json', metadata)
                print(json.dumps({k: v for k, v in summary.items() if k not in ('content', 'tokens')}), flush=True)
            metadata['status'] = 'completed'
        except BaseException as e:
            metadata['status'] = 'failed'
            metadata['error'] = repr(e)
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            stop.set()
            observer.join(timeout=20)
            metadata['server_exit_code'] = process.returncode
            metadata['after_gpu'] = capture(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv'])
            save(out / 'result.json', metadata)


def make_prompt(request, target, workload, theme='palace'):
    def tokenize(text):
        return request('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']

    expected = {'amber': '731942', 'cobalt': '528163', 'jade': '946285'}
    marker = 'STRATA_EXPERIMENT_RECORDS_MARKER'
    stories = {
        'palace': ('Then write a tense fantasy scene of about 150 words: Mara, a disgraced archivist, '
            'must retrieve the jade ledger from a flooded palace. Her former friend Iven guards the door. '
            'Show their history through dialogue and concrete details. End on a discovery, without resolving it.'),
        'station': ('Then write a tense science fiction scene of about 150 words: engineer Suri is alone '
            'on an abandoned orbital station when her dead brother answers over the radio. '
            'Use dialogue, believable mechanical details, and restrained emotion. End on a discovery, without resolving it.'),
        'detective': ('Then write a noir detective scene of about 150 words: retired investigator Daniel '
            'finds his own handwriting on a ransom note from twenty years ago. His daughter recognizes the paper. '
            'Use dialogue and concrete details to reveal their mistrust. End on a discovery, without resolving it.')}
    task = (stories[theme] if workload == 'story' else
            'Then explain in about 150 words how to preserve these facts during a long conversation.')
    content = ('Read these records from the palace archive. Later I will ask for three verification codes.\n'
               + marker + '\nQuestion: First return the verification codes for amber, cobalt, and jade, in that order. ' + task)
    formatted = request('/apply-template', {'messages': [{'role': 'user', 'content': content}],
                        'add_generation_prompt': True,
                        'chat_template_kwargs': {'enable_thinking': False, 'reasoning_effort': 'none'}})['prompt']
    before, after = formatted.split(marker)
    # apply-template omits BOS when the model's tokenizer adds it. Integer-array
    # completion bypasses that automatic insertion, so add it on this first segment.
    prefix = request('/tokenize', {'content': before, 'add_special': True, 'parse_special': True})['tokens']
    suffix = tokenize(after)
    facts = [tokenize(f'\nVerified project {key} verification code: {value}.\n') for key, value in expected.items()]
    rng = random.Random(1015)
    lines = []
    while len(lines) < max(40, target // 12):
        i = len(lines)
        lines.append(f'Record {i}: {rng.choice(["pump", "valve", "motor", "sensor", "relay", "filter"])} {rng.randrange(10000,99999)} at station {rng.randrange(1,80)}; {rng.choice(["inspected", "replaced", "calibrated", "approved", "scheduled"])} by team {rng.choice(["north", "south", "east", "west"])}; reading {rng.randrange(10,900)}; ticket {rng.randrange(100000,999999)}.\n')
    filler = tokenize(''.join(lines))
    remaining = target - len(prefix) - len(suffix) - sum(map(len, facts))
    if remaining < 0 or len(filler) < remaining:
        raise ValueError('prompt construction failed')
    parts = [remaining // 10, remaining * 4 // 10, remaining * 4 // 10]
    parts.append(remaining - sum(parts))
    tokens = prefix[:]
    offset = 0
    for i, count in enumerate(parts):
        tokens += filler[offset:offset + count]
        offset += count
        if i < len(facts):
            tokens += facts[i]
    tokens += suffix
    assert len(tokens) == target
    return tokens, expected


if __name__ == '__main__':
    main()
