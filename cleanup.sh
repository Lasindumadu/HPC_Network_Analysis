#!/bin/bash
# cleanup.sh — Remove build artifacts, logs, and generated files
# EC7207 High Performance Computing
# Usage: chmod +x cleanup.sh && ./cleanup.sh [--all]
#
# Without --all : removes logs, charts, and timing files only
# With    --all : also removes compiled binaries (requires rebuild)

set -euo pipefail

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

info() { echo -e "${GREEN}[clean]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $1"; }

CLEAN_ALL=false
if [[ "${1:-}" == "--all" ]]; then
    CLEAN_ALL=true
fi

echo "=========================================="
echo "  HPC Network Analysis — Cleanup"
echo "=========================================="
echo ""

# ── Always clean: logs, charts, timing files ──────────────────
info "Removing unwanted
log files..."
rm -rf results/logs/
mkdir -p results/logs
info "  results/logs/ cleared"

info "Removing unwanted 
timing files..."
rm -f results/serial_time.txt
rm -f results/openmp_time.txt
rm -f results/pthreads_time.txt
rm -f results/mpi_time.txt
rm -f results/hybrid_time.txt
rm -f results/cuda_time.txt
info "  results/*.txt cleared"

info "Removing generated charts..."
rm -rf charts/
mkdir -p charts
info "  charts/ cleared"

info "Removing webapp results history..."
rm -f webapp/results_history.json
info "  webapp/results_history.json removed"

info "Removing Python cache..."
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null || true
info "  __pycache__ removed"

# ── Optional: remove binaries ─────────────────────────────────
if $CLEAN_ALL; then
    info "Removing compiled binaries (--all)..."
    make clean 2>/dev/null || true
    info "  binaries removed — run 'make all' to rebuild"
else
    warn "Binaries kept. Use './cleanup.sh --all' to remove them too."
fi

echo ""
echo "=========================================="
info "Cleanup complete."
if ! $CLEAN_ALL; then
    echo ""
    echo "  To rebuild:       make all"
    echo "  To run again:     ./run_all.sh"
fi
echo "========================================================="
