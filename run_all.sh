#!/bin/bash
DATA="data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv"

echo "==============================="
echo "SERIAL"
echo "==============================="
./results/serial "$DATA"

echo "==============================="
echo "OPENMP — 1 2 4 8 THREADS"
echo "==============================="
for t in 1 2 4 8; do
    OMP_NUM_THREADS=$t ./results/openmp "$DATA"
done

echo "==============================="
echo "MPI — 1 2 4 8 PROCESSES"
echo "==============================="
for p in 1 2 4 8; do
    mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi "$DATA"
done
