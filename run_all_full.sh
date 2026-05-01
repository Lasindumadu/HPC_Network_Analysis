#!/bin/bash
# HPC Network Analysis - Run All Variants (Full Dataset, All Implementations)
# Supports: Serial, OpenMP(1,2,4,8,16), Pthreads(1,2,4,8,16), MPI(1,2,4,8,16), Hybrid
# Execute: chmod +x run_all_full.sh && ./run_all_full.sh

set -euo pipefail

DATA="data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv"

mkdir -p results/logs

echo "=========================================="
echo "Building all binaries..."
echo "=========================================="
make clean
make all

echo ""
echo "=========================================="
echo "Running Serial Baseline..."
echo "=========================================="
./results/serial "$DATA" | tee results/logs/serial.log

echo ""
echo "=========================================="
echo "Running OpenMP (1, 2, 4, 8, 16 threads)..."
echo "=========================================="
for t in 1 2 4 8 16; do
  echo "--- OpenMP with $t threads ---"
  OMP_NUM_THREADS=$t ./results/openmp "$DATA" | tee results/logs/openmp_${t}t.log
done

echo ""
echo "=========================================="
echo "Running Pthreads (1, 2, 4, 8, 16 threads)..."
echo "=========================================="
for t in 1 2 4 8 16; do
  echo "--- Pthreads with $t threads ---"
  ./results/pthreads "$DATA" $t | tee results/logs/pthreads_${t}t.log
done

echo ""
echo "=========================================="
echo "Running MPI (1, 2, 4, 8, 16 processes)..."
echo "=========================================="
for p in 1 2 4 8 16; do
  echo "--- MPI with $p processes ---"
  mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi "$DATA" | tee results/logs/mpi_${p}p.log
done

echo ""
echo "=========================================="
echo "Running Hybrid MPI + OpenMP..."
echo "=========================================="
# Hybrid configurations (np x nt) where np*nt <= 16
hybrid_configs="2x2 2x4 2x8 4x2 4x4 8x2 1x16 2x16 4x8 8x4 16x1"
for cfg in $hybrid_configs; do
  np=$(echo $cfg | cut -dx -f1)
  nt=$(echo $cfg | cut -dx -f2)
  echo "--- Hybrid: $np MPI ranks x $nt OpenMP threads ---"
  OMP_NUM_THREADS=$nt mpirun --allow-run-as-root --oversubscribe -np $np ./results/hybrid "$DATA" | tee results/logs/hybrid_${cfg}.log
done

echo ""
echo "=========================================="
echo "ALL VARIANTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Logs available at: results/logs/"
echo "To generate charts: python3 generate_charts.py"
