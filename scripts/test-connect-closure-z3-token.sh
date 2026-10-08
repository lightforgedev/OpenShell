#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workflow="$root/.github/workflows/publish-lightforge-connect-closure.yml"
dockerfile="$root/deploy/docker/Dockerfile.cli-macos"

rg -q 'READ_ONLY_GITHUB_TOKEN: \$\{\{ secrets\.GITHUB_TOKEN \}\}' "$workflow"
rg -q -- '--secret id=READ_ONLY_GITHUB_TOKEN,env=READ_ONLY_GITHUB_TOKEN' "$workflow"
rg -q -- '--mount=type=secret,id=READ_ONLY_GITHUB_TOKEN,required=false' "$dockerfile"
rg -q 'export READ_ONLY_GITHUB_TOKEN="\$\(cat /run/secrets/READ_ONLY_GITHUB_TOKEN\)"' "$dockerfile"

if rg -q '^(ARG|ENV) READ_ONLY_GITHUB_TOKEN' "$dockerfile"; then
  echo 'READ_ONLY_GITHUB_TOKEN must remain a BuildKit secret, not an image setting.' >&2
  exit 1
fi
