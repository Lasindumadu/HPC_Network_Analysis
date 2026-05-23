#!/bin/bash
# HPC Network Traffic Analysis — Run All Implementations
# EC7207 High Performance Computing
# Runs: Serial, OpenMP, Pthreads, MPI, Hybrid MPI+OpenMP, CUDA
# Usage: chmod +x run_all.sh && ./run_all.sh

set -euo pipefail

DATA="data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv"
BLOCK_SIZE=256

mkdir -p results/logs

# ============================================================
echo "=========================================="
echo " Building all implementations..."
echo "=========================================="
make clean
make all

# ============================================================
echo ""
echo "=========================================="
echo " Serial (Baseline)"
echo "=========================================="
./results/serial "$DATA" | tee results/logs/serial.log

# ============================================================
echo ""
echo "=========================================="
echo " OpenMP (1, 2, 4, 8, 16 threads)"
echo "=========================================="
for t in 1 2 4 8 16; do
  echo "--- OpenMP: $t thread(s) ---"
  OMP_NUM_THREADS=$t ./results/openmp "$DATA" | tee results/logs/openmp_${t}t.log
done

# ============================================================
echo ""
echo "=========================================="
echo " Pthreads (1, 2, 4, 8, 16 threads)"
echo "=========================================="
for t in 1 2 4 8 16; do
  echo "--- Pthreads: $t thread(s) ---"
  ./results/pthreads "$DATA" $t | tee results/logs/pthreads_${t}t.log
done

# ============================================================
echo ""
echo "=========================================="
echo " MPI (1, 2, 4, 8, 16 processes)"
echo "=========================================="
for p in 1 2 4 8 16; do
  echo "--- MPI: $p process(es) ---"
  mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi "$DATA" | tee results/logs/mpi_${p}p.log
done

# ============================================================
echo ""
echo "=========================================="
echo " Hybrid MPI + OpenMP"
echo "=========================================="
hybrid_configs="2x2 2x4 2x8 4x2 4x4 8x2 1x16 2x16 4x8 8x4 16x1"
for cfg in $hybrid_configs; do
  np=$(echo $cfg | cut -dx -f1)
  nt=$(echo $cfg | cut -dx -f2)
  echo "--- Hybrid: $np MPI rank(s) x $nt OpenMP thread(s) ---"
  OMP_NUM_THREADS=$nt mpirun --allow-run-as-root --oversubscribe -np $np ./results/hybrid "$DATA" | tee results/logs/hybrid_${np}p_${nt}t.log
done

# ============================================================
echo ""
echo "=========================================="
echo " CUDA (GPU)"
echo "=========================================="
./results/cuda "$DATA" $BLOCK_SIZE | tee results/logs/cuda.log

# ============================================================
echo ""
echo "=========================================="
echo " ALL IMPLEMENTATIONS COMPLETED"
echo "=========================================="
echo "Logs: results/logs/"
echo ""
echo "Generate charts:"
echo "  python3 generate_charts.py"
echo "  python3 generate_charts2.py"
echo "  python3 generate_charts3.py"
echo ""
echo "Verify correctness:"
echo "  python3 verify_results.py"
