#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [ -d "${REPO_ROOT}/.venv" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

if [ ! -d "${REPO_ROOT}/assets/hf/gpt2" ]; then
    echo "Missing GPT-2 tokenizer at assets/hf/gpt2."
    echo "Run: .venv/bin/python scripts/download_hf_assets.py --repo_id openai-community/gpt2 --assets tokenizer"
    exit 1
fi

if [ ! -f "${REPO_ROOT}/data/climbmix_50m/train.jsonl" ] || [ ! -f "${REPO_ROOT}/data/climbmix_50m/validation.jsonl" ]; then
    echo "Missing local ClimbMix subset at data/climbmix_50m."
    echo "Run: .venv/bin/python scripts/download_climbmix_subset.py"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RUN_GROUP="${WANDB_RUN_GROUP:-qwen3-10m-climbmix-50m-${TIMESTAMP}}"
WANDB_PROJECT="${WANDB_PROJECT:-linear-attention-climbmix}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/ablations/${RUN_GROUP}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
REPORT_PATH="${OUTPUT_ROOT}/report.md"
NGPU="${NGPU:-1}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

mkdir -p "${LOG_DIR}"

CONFIGS=(
    "qwen3 qwen3_10m_climbmix_dense"
    "linear_attention.qwen3 qwen3_10m_climbmix_gdn_dense_3to1"
    "linear_attention.qwen3 qwen3_10m_climbmix_kda_dense_3to1"
    "linear_attention.qwen3 qwen3_10m_climbmix_kda_dense_1to1"
    "linear_attention.qwen3 qwen3_10m_climbmix_swa_dense_3to1"
    "linear_attention.qwen3 qwen3_10m_climbmix_swa_gdn_1to1"
    "linear_attention.qwen3 qwen3_10m_climbmix_swa_kda_1to1"
)

{
    echo "# Qwen3 10M ClimbMix Ablation"
    echo
    echo "- Started: $(date -Is)"
    echo "- Run group: \`${RUN_GROUP}\`"
    echo "- W&B project: \`${WANDB_PROJECT}\`"
    echo "- NGPU: \`${NGPU}\`"
    echo "- Output root: \`${OUTPUT_ROOT}\`"
    echo "- Extra args: \`$*\`"
    echo
    echo "## Dataset"
    echo
    echo "- Train: \`data/climbmix_50m/train.jsonl\`"
    echo "- Validation: \`data/climbmix_50m/validation.jsonl\`"
    echo "- Metadata: \`data/climbmix_50m/metadata.json\`"
    echo
    echo "## Runs"
    echo
    echo "| Status | Module | Config | Seconds | Log |"
    echo "| --- | --- | --- | ---: | --- |"
} > "${REPORT_PATH}"

failures=0

for item in "${CONFIGS[@]}"; do
    module="${item%% *}"
    config="${item#* }"
    run_name="${config}"
    log_path="${LOG_DIR}/${run_name}.log"

    echo
    echo "=== Running ${module} / ${config} ==="
    echo "Log: ${log_path}"

    run_args=()
    if [ "${ENABLE_WANDB}" = "1" ]; then
        run_args+=("--metrics.enable_wandb")
    fi
    run_args+=("$@")

    start_seconds="${SECONDS}"
    env \
        NGPU="${NGPU}" \
        MODULE="${module}" \
        CONFIG="${config}" \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        WANDB_RUN_GROUP="${RUN_GROUP}" \
        WANDB_RUN_NAME="${run_name}" \
        ./run_train.sh "${run_args[@]}" 2>&1 | tee "${log_path}"
    status="${PIPESTATUS[0]}"
    elapsed="$((SECONDS - start_seconds))"

    if [ "${status}" -eq 0 ]; then
        result="pass"
    else
        result="fail(${status})"
        failures=$((failures + 1))
    fi

    printf '| %s | `%s` | `%s` | %s | `%s` |\n' \
        "${result}" "${module}" "${config}" "${elapsed}" "${log_path}" \
        >> "${REPORT_PATH}"

    if [ "${status}" -ne 0 ] && [ "${STOP_ON_FAILURE}" = "1" ]; then
        break
    fi
done

{
    echo
    echo "## Finished"
    echo
    echo "- Finished: $(date -Is)"
    echo "- Failures: ${failures}"
} >> "${REPORT_PATH}"

echo
echo "Report: ${REPORT_PATH}"

if [ "${failures}" -ne 0 ]; then
    exit 1
fi
