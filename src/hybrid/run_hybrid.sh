#!/bin/bash
# HPC Network Analysis - Hybrid MPI + OpenMP Run Script
# Supports multiple hybrid configurations up to 16 total parallelism
# Execute: chmod +x src/hybrid/run_hybrid.sh && ./src/hybrid/run_hybrid.sh [dataset_path]

DATA=${1:-data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv}

echo "=========================================="
echo "Hybrid MPI + OpenMP Configurations"
echo "=========================================="

# Hybrid configurations (np x nt) where np*nt <= 16
configs=(
  "1:16"
  "2:2"
  "2:4"
  "2:8"
  "2:16"
  "4:2"
  "4:4"
  "4:8"
  "8:2"
  "8:4"
  "16:1"
)

for cfg in "${configs[@]}"; do
  np=$(echo $cfg | cut -d: -f1)
  nt=$(echo $cfg | cut -d: -f2)
  total=$((np * nt))
  echo ""
  echo "--- Hybrid: $np MPI ranks x $nt OpenMP threads = $total total parallelism ---"
  OMP_NUM_THREADS=$nt mpirun --allow-run-as-root --oversubscribe -np $np ./results/hybrid "$DATA" | tee results/logs/hybrid_${np}x${nt}.log
done

echo ""
echo "=========================================="
echo "All hybrid configurations completed!"
echo "=========================================="
