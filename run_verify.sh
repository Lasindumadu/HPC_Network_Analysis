#!/bin/bash
set -euo pipefail
DATA="data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv"
OUT="results/verification"
mkdir -p "$OUT"

echo "=== RUNNING SERIAL ==="
./results/serial "$DATA" | tee "$OUT/serial.log"

echo ""
echo "=== RUNNING OPENMP ==="
for t in 1 2 4 8 16; do
  echo ""
  echo "--- OpenMP $t threads ---"
  OMP_NUM_THREADS=$t ./results/openmp "$DATA" | tee "$OUT/openmp_${t}t.log"
done

echo ""
echo "=== RUNNING PTHREADS ==="
for t in 1 2 4 8 16; do
  echo ""
  echo "--- Pthreads $t threads ---"
  ./results/pthreads "$DATA" $t | tee "$OUT/pthreads_${t}t.log"
done

echo ""
echo "=== RUNNING MPI ==="
for p in 1 2 4 8 16; do
  echo ""
  echo "--- MPI $p processes ---"
  mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi "$DATA" | tee "$OUT/mpi_${p}p.log"
done

echo ""
echo "=== RUNNING HYBRID ==="
for cfg in 2x2 2x4 2x8 4x2 4x4 8x2 1x16 2x16 4x8 8x4 16x1; do
  echo ""
  echo "--- Hybrid $cfg ---"
  np=$(echo $cfg | cut -dx -f1)
  nt=$(echo $cfg | cut -dx -f2)
  OMP_NUM_THREADS=$nt mpirun --allow-run-as-root --oversubscribe -np $np ./results/hybrid "$DATA" | tee "$OUT/hybrid_${cfg}.log"
done

echo ""
echo "=== ALL RUNS COMPLETE ==="
