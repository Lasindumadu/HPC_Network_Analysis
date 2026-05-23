#!/bin/bash
# setup.sh — Install dependencies and build all HPC implementations
# EC7207 High Performance Computing
# Usage: chmod +x setup.sh && ./setup.sh

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[setup]${NC} $1"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $1"; }
error()   { echo -e "${RED}[error]${NC} $1"; }

echo "=========================================="
echo "  HPC Network Analysis — Setup"
echo "  EC7207 High Performance Computing"
echo "=========================================="
echo ""

# ── 1. System packages ─────────────────────────────────────────
info "Updating package list..."
sudo apt-get update -qq

info "Installing GCC, MPI, Python, and build tools..."
sudo apt-get install -y \
    gcc \
    make \
    libopenmpi-dev \
    openmpi-bin \
    python3-pip \
    python3-dev \
    --no-install-recommends

# ── 2. Python packages ─────────────────────────────────────────
info "Installing Python dependencies..."
pip3 install -r webapp/requirements.txt --quiet

# ── 3. Check for nvcc (optional) ──────────────────────────────
echo ""
if command -v nvcc &> /dev/null; then
    info "nvcc found: $(nvcc --version | grep release | awk '{print $6}')"
    info "CUDA implementation will be compiled."
else
    warn "nvcc not found — CUDA binary will be skipped."
    warn "To compile CUDA: install CUDA toolkit and re-run 'make results/cuda'"
    warn "On Google Colab: nvcc is pre-installed (Tesla T4 available)."
fi

# ── 4. Verify compilers ───────────────────────────────────────
echo ""
info "Checking compilers..."
gcc_ver=$(gcc --version | head -n1)
mpi_ver=$(mpicc --version | head -n1)
py_ver=$(python3 --version)
info "  gcc:    $gcc_ver"
info "  mpicc:  $mpi_ver"
info "  python: $py_ver"

# ── 5. Build ──────────────────────────────────────────────────
echo ""
info "Building all implementations..."
make clean
make all

# ── 6. Check dataset ──────────────────────────────────────────
echo ""
DATASET="data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv"
if [ -f "$DATASET" ]; then
    info "Dataset found: $DATASET"
else
    warn "Dataset not found at: $DATASET"
    warn "Download from: https://drive.google.com/drive/folders/1tqNgeGTsgRTTDDsr46wnJ4Gt4gdxUfzN"
    warn "Then place it at: $DATASET"
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "=========================================="
info "Setup complete."
echo ""
echo "  Run benchmarks:   ./run_all.sh"
echo "  Web dashboard:    cd webapp && python3 app.py"
echo "  Generate charts:  python3 generate_charts_all.py"
echo "=========================================="
