#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export MAMBA_FORCE_BUILD="${MAMBA_FORCE_BUILD:-TRUE}"

uv sync \
  --extra dev \
  --extra linear-attention \
  --extra mamba3 \
  --no-build-isolation \
  --index-strategy unsafe-best-match \
  --prerelease allow
