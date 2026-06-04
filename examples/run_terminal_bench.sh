#!/usr/bin/env bash
# Launch deepagent-hermes against Terminal-Bench 2.0.
# Assumes:
#   - $OPENROUTER_API_KEY is already in env
#   - WSL venv at ~/.venvs/dah-bench has harbor + deepagent-hermes installed
#   - Docker daemon is reachable
#
# Routes hermes through OpenRouter's OpenAI-compatible API by setting
# OPENAI_BASE_URL + OPENAI_API_KEY, then handing langchain the model id
# `openai:<openrouter-style-id>`. The langchain ChatOpenAI client uses
# the base URL, OpenRouter routes to Anthropic underneath.
#
# Usage:
#   examples/run_terminal_bench.sh <n_tasks> [model]
#   examples/run_terminal_bench.sh 1                                 # smoke, default model
#   examples/run_terminal_bench.sh 89 anthropic/claude-sonnet-4.5    # full run
set -euo pipefail

N_TASKS="${1:-1}"
# Default to direct Anthropic, not OpenRouter — Anthropic supports
# prompt caching natively, and hermes's system prompt is large enough
# (~30k tokens with the skill index + bundled tool docs) that paying
# full price every turn explodes the cost. The first smoke through
# OpenRouter cost ~$15/task because cache_read was 0; the same task
# through Anthropic with caching should land at $1-2.
PROVIDER="${HERMES_BENCH_PROVIDER:-anthropic}"
case "$PROVIDER" in
    anthropic)
        MODEL="${2:-anthropic:claude-sonnet-4-5-20250929}"
        ;;
    openrouter)
        OR_MODEL="${2:-anthropic/claude-sonnet-4.5}"
        MODEL="openai:$OR_MODEL"
        if [ -z "${OPENROUTER_API_KEY:-}" ]; then
            echo "OPENROUTER_API_KEY not set — aborting." >&2
            exit 2
        fi
        export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
        export OPENAI_API_KEY="$OPENROUTER_API_KEY"
        ;;
    *)
        echo "Unknown HERMES_BENCH_PROVIDER=$PROVIDER (expected: anthropic | openrouter)" >&2
        exit 2
        ;;
esac

N_CONCURRENT="${HERMES_BENCH_N_CONCURRENT:-4}"
DATASET="${HERMES_BENCH_DATASET:-terminal-bench/terminal-bench-2}"
REPO="${HERMES_BENCH_REPO:-/mnt/c/Users/Kedar/Documents/Code/deepagent-hermes}"
VENV="${HERMES_BENCH_VENV:-$HOME/.venvs/dah-bench}"
JOBS_DIR="${HERMES_BENCH_JOBS_DIR:-$HOME/hermes-bench/jobs}"

if [ "$PROVIDER" = "anthropic" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ANTHROPIC_API_KEY not set — aborting." >&2
    exit 2
fi

export PYTHONPATH="$REPO/examples:${PYTHONPATH:-}"

# Cost-per-megatoken estimates (Anthropic list price; OpenRouter charges
# the same plus a small markup). Override via env for exact accounting.
# Sonnet 4.5 list: $3/MTok fresh input, $0.30/MTok cache-read, $15/MTok output.
export HERMES_BENCH_COST_PER_INPUT_MTOK="${HERMES_BENCH_COST_PER_INPUT_MTOK:-3.00}"
export HERMES_BENCH_COST_PER_CACHE_READ_MTOK="${HERMES_BENCH_COST_PER_CACHE_READ_MTOK:-0.30}"
export HERMES_BENCH_COST_PER_OUTPUT_MTOK="${HERMES_BENCH_COST_PER_OUTPUT_MTOK:-15.00}"

# shellcheck disable=SC1090
source "$VENV/bin/activate"

# Hermes routes through `init_chat_model("openai:...")`, which uses
# langchain-openai. The base deepagent-hermes install doesn't pull it
# in (only the optional `[openai]` extra does). Fail fast if missing
# — uv venvs don't include pip by default, so we error rather than
# trying to install on the user's behalf.
python -c "import langchain_openai" 2>/dev/null || {
    echo "[run_terminal_bench] langchain-openai not installed in $VENV." >&2
    echo "[run_terminal_bench] Install with: uv pip install langchain-openai" >&2
    exit 3
}

mkdir -p "$JOBS_DIR"
cd "$JOBS_DIR"

echo "[run_terminal_bench] provider=$PROVIDER  model=$MODEL"
echo "[run_terminal_bench] dataset=$DATASET  n_tasks=$N_TASKS  n_concurrent=$N_CONCURRENT"
echo "[run_terminal_bench] jobs_dir=$JOBS_DIR"

harbor run \
    --agent-import-path "terminal_bench:DeepagentHermesAgent" \
    --model "$MODEL" \
    --env docker \
    --dataset "$DATASET" \
    --n-tasks "$N_TASKS" \
    --n-concurrent "$N_CONCURRENT"
