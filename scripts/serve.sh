#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model="${1:-${MISTY_MODEL:-}}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ -z "$model" || ! -f "$model" ]]; then
  echo 'Usage: bash scripts/serve.sh /path/to/model.gguf [extra llama-server arguments]' >&2
  exit 2
fi
server="${MISTY_SERVER:-$PWD/build/mistyloader/bin/llama-server}"
if [[ ! -x "$server" ]]; then
  echo 'Build MistyLoader first: bash scripts/build.sh' >&2
  exit 2
fi
if command -v systemctl >/dev/null && systemctl --user is-active --quiet strata.service 2>/dev/null; then
  echo 'Stop strata.service before starting MistyLoader.' >&2
  exit 1
fi
mkdir -p artifacts
exec 9>artifacts/experiment.lock
if ! flock -n 9; then
  echo 'Another MistyLoader experiment or server is running.' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export LD_LIBRARY_PATH="$(dirname "$server")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_CUDA_REGISTER_HOST=1
export MISTRAL_CUDA_F32_DOWN=1
export MISTRAL_CACHE_HYBRID=1
export MISTRAL_CACHE_WARM_STEPS=64
export MISTRAL_CACHE_POLICY=lfu
export MISTRAL_CACHE_REFRESH_STEPS=4
export MISTRAL_CACHE_MAX_TOKENS=1
export MISTRAL_CACHE_WHOLE_FFN=1
export MISTRAL_CPU_SKIP_MASKED=1
export MISTRAL_CPU_CONCURRENT=1
unset MISTRAL_CACHE_QUANT_BATCH MISTRAL_CPU_CONCURRENT_SERIAL
exec "$server" \
  -m "$model" --alias mistyloader \
  --host 127.0.0.1 --port "${PORT:-18090}" \
  -c "${CONTEXT_SIZE:-8192}" -np 1 -ngl 99 --cpu-moe -sm layer -ts 1,1 \
  --fit off -fa on -ctk q8_0 -ctv q8_0 \
  -b 2048 -ub 256 -t "${DECODE_THREADS:-12}" -tb "${PREFILL_THREADS:-16}" \
  --moe-cache-mib "${CACHE_MIB:-16384}" --cache-ram 0 --no-context-shift \
  --jinja --chat-template-kwargs '{"reasoning_effort":"none"}' \
  --spec-type none --perf "$@"
