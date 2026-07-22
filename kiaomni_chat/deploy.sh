#!/usr/bin/env bash
# KiaOmni Chat — deploy helper
#
# Usage:
#   ./deploy.sh deploy      # full deploy (slow first time, builds image)
#   ./deploy.sh demo-test   # run the demo acceptance test
#   ./deploy.sh vram-test   # run the no-GC VRAM accumulation test
#   ./deploy.sh status      # check deploy status
#   ./deploy.sh stop        # stop the deployed app
#
# No HF token required: Qwen2.5-7B-Instruct is Apache 2.0 / ungated.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

cmd="${1:-help}"
shift || true

case "$cmd" in
  deploy)
    echo "==> Deploying kiaomni-chat to Modal (A100-40GB)..."
    modal deploy kiaomni_chat/modal_app.py
    echo
    echo "==> Done. Open the URL printed above to chat."
    echo "    (First request triggers ~3 min cold start.)"
    ;;
  demo-test)
    echo "==> Running demo acceptance test (4 tasks × kiaomni_gaussian @ B=512)..."
    modal run kiaomni_chat/modal_app.py::test_demo_acceptance
    ;;
  q1q9-test)
    budget="${2:-512}"
    echo "==> Running Q1–Q9 regression test (kiaomni_gaussian @ B=${budget})..."
    modal run kiaomni_chat/modal_app.py::test_q1_q9_acceptance --budget "$budget"
    ;;
  vram-test)
    n="${1:-100}"
    echo "==> Running VRAM accumulation test (${n} requests, no GC)..."
    modal run kiaomni_chat/modal_app.py::test_vram_accumulation --n-requests "$n"
    ;;
  status)
    modal app show kiaomni-chat 2>/dev/null || modal app list | grep kiaomni || echo "not deployed"
    ;;
  stop)
    echo "==> Stopping kiaomni-chat (it will respawn on next request)..."
    modal app stop kiaomni-chat 2>/dev/null || true
    ;;
  help|*)
    cat <<EOF
kiaomni_chat deploy helper

  deploy      Deploy to Modal (A100-40GB). First run builds image.
  demo-test   Run the 4 demo tasks × kiaomni_gaussian @ B=512.
  vram-test   Run the no-GC VRAM accumulation stress test.
  status      Show app status.
  stop        Stop the deployed app.

Qwen2.5-7B-Instruct is Apache 2.0 / ungated — no HF token required.
EOF
    ;;
esac
