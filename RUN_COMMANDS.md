# HPC Network Analysis — Terminal Commands (Non-CUDA)

**Dataset:** `data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv`

---

## 1. Build All Non-CUDA Implementations

```bash
make clean
make all
```

This produces:
- `results/serial`
- `results/openmp`
- `results/pthreads`
- `results/mpi`
- `results/hybrid`

---

## 2. Serial Baseline

```bash
./results/serial data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv
```

**What to verify:** Accuracy, Precision, Recall, F1, RMSE, Confusion Matrix, Protocol Stats.  
These values are the **correctness baseline** for all parallel runs.

---

## 3. OpenMP (1, 2, 4, 8, 16 Threads)

```bash
for t in 1 2 4 8 16; do
  echo "========================================"
  echo "OpenMP with $t threads"
  echo "========================================"
  OMP_NUM_THREADS=$t ./results/openmp data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv
done
```

---

## 4. Pthreads (1, 2, 4, 8, 16 Threads)

```bash
for t in 1 2 4 8 16; do
  echo "========================================"
  echo "Pthreads with $t threads"
  echo "========================================"
  ./results/pthreads data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv $t
done
```

---

## 5. MPI (1, 2, 4, 8, 16 Processes)

```bash
for p in 1 2 4 8 16; do
  echo "========================================"
  echo "MPI with $p processes"
  echo "========================================"
  mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv
done
```

---

## 6. Hybrid MPI + OpenMP

```bash
hybrid_configs="2x2 2x4 2x8 4x2 4x4 8x2 1x16 2x16 4x8 8x4 16x1"
for cfg in $hybrid_configs; do
  np=$(echo $cfg | cut -dx -f1)
  nt=$(echo $cfg | cut -dx -f2)
  echo "========================================"
  echo "Hybrid: $np MPI ranks x $nt OpenMP threads"
  echo "========================================"
  OMP_NUM_THREADS=$nt mpirun --allow-run-as-root --oversubscribe -np $np ./results/hybrid data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv
done
```

---

## 7. Quick Correctness Check

After each run, compare these lines against the serial baseline:

```bash
grep -E "Accuracy:|Precision:|Recall:|F1 Score:|RMSE:" results/logs/serial.log
```

All parallel variants should produce **identical** Accuracy, Precision, Recall, F1, and RMSE values because the detection engine is deterministic.

---

## 8. Batch Run Script (Alternative)

Save as `run_verification.sh` and execute:

```bash
#!/bin/bash
set -euo pipefail
DATA="data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv"
mkdir -p results/logs

echo "=== SERIAL ==="
./results/serial "$DATA" | tee results/logs/serial_verify.log

echo "=== OPENMP ==="
for t in 1 2 4 8 16; do
  OMP_NUM_THREADS=$t ./results/openmp "$DATA" | tee results/logs/openmp_${t}t_verify.log
done

echo "=== PTHREADS ==="
for t in 1 2 4 8 16; do
  ./results/pthreads "$DATA" $t | tee results/logs/pthreads_${t}t_verify.log
done

echo "=== MPI ==="
for p in 1 2 4 8 16; do
  mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi "$DATA" | tee results/logs/mpi_${p}p_verify.log
done

echo "=== HYBRID ==="
for cfg in 2x2 2x4 2x8 4x2 4x4 8x2 1x16 2x16 4x8 8x4 16x1; do
  np=$(echo $cfg | cut -dx -f1)
  nt=$(echo $cfg | cut -dx -f2)
  OMP_NUM_THREADS=$nt mpirun --allow-run-as-root --oversubscribe -np $np ./results/hybrid "$DATA" | tee results/logs/hybrid_${cfg}_verify.log
done
```

---

> **Note:** CUDA is intentionally excluded as per requirements.
