# HPC Network Analysis — New Machine Setup Guide

Complete step-by-step guide to clone, build, and run all non-CUDA implementations on a fresh machine.

---

## 1. System Requirements

### Minimum Hardware
- CPU: 4+ cores (8+ recommended for 16-thread testing)
- RAM: 8GB minimum
- Disk: 10GB free space
- OS: Linux (Ubuntu 20.04+ recommended)

### Required Software
```bash
sudo apt update
sudo apt install -y gcc libomp-dev openmpi-bin libopenmpi-dev python3 python3-pip
pip3 install matplotlib numpy
```

Verify installations:
```bash
gcc --version          # Should support C11
mpicc --version        # MPI compiler
mpirun --version       # MPI runtime
```

---

## 2. Get the Project

```bash
git clone <repository-url>
cd HPC_Network_Analysis
```

---

## 3. Prepare the Dataset

**Dataset file:** `UNSW-NB15_1_with_header.csv`

Place it at:
```
HPC_Network_Analysis/data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv
```

Verify:
```bash
ls -lah data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv
```

Expected: ~700,000 records with header row containing column names.

---

## 4. Build All Implementations

```bash
make clean
make all
```

This creates:
- `results/serial`
- `results/openmp`
- `results/pthreads`
- `results/mpi`
- `results/hybrid`

---

## 5. Run All Implementations

### Dataset Path Variable
```bash
DATA="data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv"
```

### 5.1 Serial (Baseline)
```bash
./results/serial "$DATA"
```

### 5.2 OpenMP (1, 2, 4, 8, 16 threads)
```bash
for t in 1 2 4 8 16; do
  echo "=== OpenMP $t threads ==="
  OMP_NUM_THREADS=$t ./results/openmp "$DATA"
done
```

### 5.3 POSIX Threads (1, 2, 4, 8, 16 threads)
```bash
for t in 1 2 4 8 16; do
  echo "=== Pthreads $t threads ==="
  ./results/pthreads "$DATA" $t
done
```

### 5.4 MPI (1, 2, 4, 8, 16 processes)
```bash
for p in 1 2 4 8 16; do
  echo "=== MPI $p processes ==="
  mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi "$DATA"
done
```

### 5.5 Hybrid MPI + OpenMP
```bash
for cfg in 2x2 2x4 2x8 4x2 4x4 8x2 1x16 2x16 4x8 8x4 16x1; do
  echo "=== Hybrid $cfg ==="
  np=$(echo $cfg | cut -dx -f1)
  nt=$(echo $cfg | cut -dx -f2)
  OMP_NUM_THREADS=$nt mpirun --allow-run-as-root --oversubscribe -np $np ./results/hybrid "$DATA"
done
```

---

## 6. Automated Verification

All implementations should produce **identical** accuracy metrics. Run the verification script:

```bash
python3 verify_results.py
```

Expected output: `26/26 implementations PASSED`

Correct baseline values:
- **Accuracy:** 97.072%
- **Precision:** 53.154%
- **Recall:** 65.087%
- **F1 Score:** 58.518%
- **RMSE:** 0.171126
- **Confusion Matrix:** TP=14459, TN=665043, FP=12743, FN=7756

---

## 7. Generate Performance Charts

```bash
python3 generate_charts.py
```

Output saved to `charts/`:
- `speedup.png`
- `efficiency.png`
- `execution_time.png`
- `throughput.png`
- `all_charts.png`

---

## 8. Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `mpicc: command not found` | `sudo apt install openmpi-bin libopenmpi-dev` |
| `libomp: not found` | `sudo apt install libomp-dev` |
| `Permission denied` on scripts | `chmod +x run_all.sh run_verify.sh` |
| `Dataset not found` | Check path: `data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv` |
| `mpirun` fails with 16 processes | Add `--oversubscribe` flag |

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `SETUP_NEW_MACHINE.md` | This guide |
| `README.md` | Full project documentation |
| `PROJECT_GUIDE.md` | Detailed technical guide |
| `RUN_COMMANDS.md` | Command cheat sheet |
| `Makefile` | Build system |
| `run_verify.sh` | Automated batch run script |
| `verify_results.py` | Correctness verification |

---

## One-Command Full Run

Save time by running everything automatically:
```bash
bash run_verify.sh && python3 verify_results.py
```

> **Note:** CUDA is intentionally excluded from this guide as per project requirements.
