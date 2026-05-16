#!/usr/bin/env bash
# Convenience entrypoint for research workflows. The canonical setup logic lives
# in scripts/setup_linear_attention_env.sh.

set -euo pipefail

RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${RESEARCH_DIR}/../scripts/setup_linear_attention_env.sh"
