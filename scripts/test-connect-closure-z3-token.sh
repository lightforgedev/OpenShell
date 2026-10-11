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
rg -q 'package: openshell-cli' "$workflow"
rg -q 'triple: aarch64-unknown-linux-musl' "$workflow"
rg -q 'artifact-name: connect-agent-openshell-cli-linux-arm64' "$workflow"
rg -q 'package: openshell-sandbox' "$workflow"
rg -q 'artifact-name: connect-openshell-supervisor-linux-arm64' "$workflow"
rg -q 'package: openshell-gateway' "$workflow"
rg -q 'package: openshell-supervisor' "$workflow"

if rg -q 'rust-native-build\.yml|docker-build\.yml' "$workflow"; then
  echo 'Connect closure must use the maintained build actions, not removed workflows.' >&2
  exit 1
fi

if sed -n '/package: openshell-cli/,/artifact-name: connect-agent-openshell-cli-linux-arm64/p' "$workflow" | rg -q 'bundled-z3'; then
  echo 'openshell-cli no longer owns the bundled-z3 feature.' >&2
  exit 1
fi

if rg -q '^(ARG|ENV) READ_ONLY_GITHUB_TOKEN' "$dockerfile"; then
  echo 'READ_ONLY_GITHUB_TOKEN must remain a BuildKit secret, not an image setting.' >&2
  exit 1
fi
