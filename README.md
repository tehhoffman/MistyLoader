# MistyLoader

I started this after getting roughly 33-43 tok/s out of Flash Next with [Strata](https://github.com/attentiondotnet/Strata). Seeing that run on my machine made me want to try a similar approach with Mistral Small 4. I know they're different architectures, but it seemed worth finding out how much more we could get out of Mistral on consumer hardware.

My main use case is storytelling and long conversations. The plan is to get Q4 moving as fast as I can first, then see whether Q8 is practical. Q5 is the fallback. This is the first public checkpoint of that work.

**Right now, the selected configuration averages about 26.4 tok/s after warmup on my short storytelling tests, with individual requests reaching 27.6 tok/s.** That's up from 23.6 tok/s at the previous checkpoint. The 33-43 tok/s Strata result is the inspiration, not a matched benchmark between the two models.

This is an **AI-assisted project**. I've been using OpenAI Codex heavily for implementation, debugging, testing, and benchmarking. I want to be upfront about that. I'm sharing the code, the pinned source revision, and the measurements so there's something concrete to inspect and build on. This is experimental software, and this release is a checkpoint rather than a claim that every model, quant, or machine is supported.

## What it does

MistyLoader builds on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp). For this release, I'm keeping the changes as a patch against one exact upstream commit instead of carrying a second copy of the whole repository.

The current path combines:

- A GPU expert cache with frequency-based admissions that continue across requests.
- Whole expert branches, so gate/up, SwiGLU, and down stay together on their assigned processor.
- Actual CPU/GPU concurrency: CPU misses run in an owned worker while the GPU processes cached experts, then the results join before the next layer.
- An F32-input CUDA down-projection path for the large activations encountered in this model.
- A shortcut for CPU work when every selected expert is already covered by the GPU.

All four selected routed experts still contribute. The concurrent worker owns its input and scratch buffers, and the existing scheduler memory-reuse fences stay in place. The runtime patch also retains the earlier speculative-decoding experiment, but speculation is off in the released profile because it didn't improve net speed in my tests.

## The checkpoint

My test machine has a Ryzen 9 5950X, 128 GB RAM, and two RTX 3060 12 GB GPUs. The GPUs run over PCIe 4 x8, without NVLink. The selected profile uses an 8K context allocation, a 16 GiB combined expert cache, 12 decode threads, 16 prefill threads, and Q8 KV cache.

| Configuration | Fixed-prompt warmed mean | Storytelling warmed mean |
| --- | ---: | ---: |
| Previous whole-expert profile, 16 threads | 23.12 tok/s | 23.62 tok/s |
| Concurrent execution, 16 threads | 25.93 tok/s | 26.09 tok/s |
| Selected concurrent profile, 12 threads | **26.38 tok/s** | **26.44 tok/s** |

The fixed test uses a 512-token prompt and generates 256 tokens over three requests. The story test cycles fantasy, science fiction, and detective prompts twice, with 512 prompt tokens and up to 384 output tokens. The warmed means exclude the first request. These are decode rates; prompt processing is measured separately. The selected profile's first story request, including cache warmup, decoded at 22.58 tok/s.

The storytelling improvement is about **12%**. Twelve threads only beat sixteen by about 1.4% in that comparison, so I wouldn't assume that thread count wins everywhere. All matched output-token sequences were identical across the previous and concurrent story profiles, and all retrieval checks passed.

A sixteen-thread profiler run measured about 4.2 ms of CPU/GPU compute overlap per generated token. The serial control, using the same new executor and buffers, measured zero overlap and stayed around 23 tok/s. The CPU executor also passed 200 bitwise comparisons covering all-hit, mixed, and all-miss routing, buffer reuse, and shutdown with work pending. Earlier CPU/CUDA tests cover the unchanged arithmetic kernels.

The selected story run peaked at about 67.67 GiB of process RSS, with zero model swap. Sampled GPU use was 11,263 / 9,551 MiB, including desktop use. The per-request numbers, model hashes, test scope, and profiler summaries are in [benchmarks/checkpoint.json](benchmarks/checkpoint.json).

Before publishing, I also built the packaged source from a fresh upstream checkout, reran the 200 CPU comparisons and 12 CPU/CUDA expert checks, and checked the launcher with the real model. Those release checks are recorded in [benchmarks/release-verification.json](benchmarks/release-verification.json).

## Build

This release is **source-only**. Model weights and prebuilt CUDA binaries aren't included.

The tested setup is Linux with Git, Python 3, CMake, Ninja, a C++ compiler, and the NVIDIA CUDA toolkit. The development build used CUDA 13.1 and GCC 14.2. The default CUDA target is SM86, matching the RTX 3060s; other hardware hasn't been validated for this checkpoint.

```bash
git clone https://github.com/tehhoffman/MistyLoader.git
cd MistyLoader
git checkout v0.1.0
bash scripts/build.sh
```

The build script fetches llama.cpp at `b76b13379d3b7d67ab0bc424516bb6fd2522bd8e`, applies the patch, and verifies the changed source files against [upstream.lock.json](upstream.lock.json). It refuses to reset or overwrite an existing source checkout that differs from the checkpoint. The fetched upstream tree is local build input and is ignored by this repository.

If `nvcc` isn't on your path, set `CUDACXX` to its location. `BUILD_JOBS` defaults to 8, and `CUDA_ARCHITECTURES` defaults to 86. Additional arguments to `scripts/build.sh` are passed to CMake. To verify an existing checkout without downloading anything:

```bash
python3 scripts/bootstrap.py --check
```

The executable remains named `llama-server` and is built under `build/mistyloader/bin/`. This build provides the API server; it doesn't bundle a chat UI.

## Run

Supply a compatible GGUF separately. The exact model used for the numbers above is described below.

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve.sh /absolute/path/to/model.gguf
```

The server listens on `http://127.0.0.1:18090`, with an OpenAI-compatible API at `/v1` and model alias `mistyloader`. Ctrl+C unloads it. The launcher respects your GPU selection and refuses to run alongside an active `strata.service` or another experiment using this project's lock.

| Environment variable | Default |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1` |
| `DECODE_THREADS` | `12` |
| `PREFILL_THREADS` | `16` |
| `CACHE_MIB` | `16384`, combined across both GPUs |
| `CONTEXT_SIZE` | `8192` |
| `PORT` | `18090` |

Those defaults reproduce the current speed profile. Raising the context allocation changes the memory budget; it doesn't establish that an occupied long conversation will run at the rates above. The concurrent path is currently limited to single-token Mistral Small 4 decoding with the tested Q4_K/Q6_K expert layouts, without adapters, precision overrides, or Hadamard transforms. The CPU worker uses the initial decode thread count; custom decode CPU affinity isn't supported yet.

## About the model I used

The benchmark target is a repaired copy of the Q4_K_M file from [timteh673's Mistral Small 4 Uncensored GGUF repository](https://huggingface.co/timteh673/Mistral-Small-4-119B-Uncensored-GGUF). The supplied file matched its published checksum, but the expert scaling investigation led to an experimental repair using the original model's per-expert FP8 scale factors. The repair changes quantization scale fields and preserves the other tensor bytes. It produces coherent output in the tests, but it isn't equivalent to doing a fresh conversion from the original weights.

I don't want the benchmark numbers to imply that every Mistral Small 4 GGUF has been tested. The exact source revision, original checksum, repaired checksum, and official model revision are in [configs/model-provenance.json](configs/model-provenance.json).

If you have that **exact original file** and want to reproduce the repair:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-repair.txt
.venv/bin/python scripts/fetch_expert_scales.py
.venv/bin/python scripts/repair_expert_scales.py /absolute/path/to/original.gguf
```

The fetcher reads small scale tensors from the pinned official model revision. It doesn't download the full original model. The repair script checks the original file's SHA256, writes a separate copy, verifies that non-expert bytes are unchanged, and checks the repaired SHA256. It needs enough disk space for another roughly 72 GB file. It refuses other source checksums or an existing output file; this is not a general repair tool for arbitrary GGUFs.

## Reproduce the tests

Run the CPU executor checks without loading a model:

```bash
build/mistyloader/bin/test-mistral-moe-cpu
MISTRAL_CPU_SKIP_MASKED=1 build/mistyloader/bin/test-mistral-moe-cpu
```

Check the full expert arithmetic against CUDA:

```bash
MISTRAL_CPU_SKIP_MASKED=1 MISTRAL_CUDA_F32_DOWN=1 \
  build/mistyloader/bin/test-backend-ops test -b CUDA0 -o MISTRAL_EXPERT_FFN
```

The benchmark harness needs `nvidia-smi`, a free localhost port, and a new run name. It unloads its own server when finished. For the fixed fixture, use the repaired model with the recorded tokenizer:

```bash
python3 scripts/experiment.py \
  --name my-checkpoint-test --model /absolute/path/to/repaired-model.gguf \
  --ctx 8192 --threads 12 --threads-batch 16 --cache-mib 16384 \
  --pin-host --cuda-f32-down --hybrid-cache \
  --cache-policy lfu --cache-refresh-steps 4 \
  --cache-whole-ffn --cpu-skip-masked --cpu-concurrent \
  --predict 256 --reps 3 --prompt-file benchmarks/fixed-prompt.json
```

For the changing-topic test, replace `--prompt-file ... --predict 256 --reps 3` with `--story-suite --predict 384 --reps 6`. Results go into the ignored `artifacts/` directory. `--cpu-concurrent-serial` runs the same owned executor synchronously for comparison. Omitting `--cpu-concurrent` selects the earlier whole-expert schedule.

Optional NVTX tracing is enabled at build time with `MISTY_NVTX=ON`. Use Nsight Systems with `--trace=cuda,nvtx`, export the trace to SQLite, and run `scripts/summarize_decode_trace.py` on it. The summarizer uses this model's logits transfers to identify the final decode window. Profiled timings include overhead; the throughput results above come from unprofiled runs.

## Where I want to take it

The next speed work is reducing CPU expert processing cost and testing whether sending a measured share of cache misses to the GPU helps. I'm still aiming for 33-43 tok/s, but this checkpoint hasn't reached that range.

The longer-term target is good storytelling at 64K-128K occupied context, then a higher quant. Earlier cold long-context tests retrieved the test facts, but prompt reuse had problems. The concurrent speed profile hasn't been validated at those occupied lengths. Q8 is still the goal, with Q5 as a fallback; neither higher target quant has been downloaded or benchmarked for this project. They also need work beyond the current Q4_K/Q6_K down-projection path.

## Credit

- **[Strata](https://github.com/attentiondotnet/Strata), Niko1221, and its contributors**: the inspiration for the project and a valuable reference for expert caching, CPU/GPU overlap, and memory ownership. I studied revision `a0116b2782a5b82622358c04a06496485d12b2d6` during this work.
- **[llama.cpp / ggml and their contributors](https://github.com/ggml-org/llama.cpp)**: the actual foundation of this runtime, including model loading, inference, quantization, CPU/CUDA kernels, and the existing MoE cache. MistyLoader is an extension of that work.
- **[Mistral AI](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)**: Mistral Small 4 and the original model material used in the scaling investigation.
- **[timteh673](https://huggingface.co/timteh673/Mistral-Small-4-119B-Uncensored-GGUF)**: the community GGUF that started this experiment.
- **[Red Hat AI](https://huggingface.co/RedHatAI/Mistral-Small-4-119B-2603.dflash)**: the draft model used in the speculative-decoding investigation, which is deferred for now.
- **OpenAI Codex**: AI assistance throughout the implementation and testing process.

MistyLoader's code is distributed under the [MIT license](LICENSE), with upstream attribution retained in [NOTICE](NOTICE) and `licenses/`. Models keep their own licenses. This is an independent project, not an official release from any of the projects or organizations above.
