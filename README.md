# HPC Network Traffic Analysis
## EC7207 — High Performance Computing

**Team Members:**
- EG/2021/4426 — Bandara AWLM
- EG/2021/4432 — Bandara KMTON
- EG/2021/4433 — Bandara LRTD

---

## Project Overview

This project implements parallel computing techniques for real-time network traffic anomaly detection using the **UNSW-NB15 dataset**. The serial baseline processes traffic features to detect anomalies, and parallel implementations using **OpenMP**, **POSIX Threads**, **MPI**, and **Hybrid MPI+OpenMP** achieve significant speedup for real-time cybersecurity monitoring.

### Key Features
- **Traffic feature-based anomaly detection** (no labels used for prediction)
- **RMSE validation** against ground-truth labels
- **Speedup & efficiency** calculation against serial baseline
- **Support for up to 16 threads/processes** across all parallel implementations

---

## Implemented Variants

| Variant | Paradigm | File | Workers Supported |
|---------|----------|------|-------------------|
| Serial | Baseline | `src/serial/network_analysis_serial.c` | 1 |
| OpenMP | Shared Memory | `src/openmp/network_analysis_openmp.c` | 1, 2, 4, 8, 16 |
| Pthreads | Shared Memory | `src/pthreads/network_analysis_pthread.c` | 1, 2, 4, 8, 16 |
| MPI | Distributed Memory | `src/mpi/network_analysis_mpi.c` | 1, 2, 4, 8, 16 |
| Hybrid | MPI + OpenMP | `src/hybrid/network_analysis_hybrid.c` | Combos up to 16 total |

---

## Quick Start

### Build All Implementations
```bash
make clean
make all
```

### Run All Variants (1, 2, 4, 8, 16 workers)
```bash
chmod +x run_all.sh
./run_all.sh
```

### Generate Performance Charts
```bash
python3 generate_charts.py
```

Charts are saved to `charts/`:
- `speedup.png` — Speedup vs workers
- `efficiency.png` — Parallel efficiency
- `execution_time.png` — Time per pass
- `throughput.png` — Records processed per second
- `all_charts.png` — Combined 2×2 dashboard

---

## Individual Build & Run

### Serial
```bash
make results/serial
./results/serial data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

### OpenMP
```bash
make results/openmp
OMP_NUM_THREADS=4 ./results/openmp data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
# Or for 16 threads:
OMP_NUM_THREADS=16 ./results/openmp data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

### POSIX Threads
```bash
make results/pthreads
./results/pthreads data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv 4
# Or for 16 threads:
./results/pthreads data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv 16
```

### MPI
```bash
make results/mpi
mpirun --allow-run-as-root --oversubscribe -np 4 ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
# Or for 16 processes:
mpirun --allow-run-as-root --oversubscribe -np 16 ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

### Hybrid MPI + OpenMP
```bash
make results/hybrid
# 4 MPI ranks × 4 OpenMP threads = 16 total parallelism
OMP_NUM_THREADS=4 mpirun --allow-run-as-root --oversubscribe -np 4 ./results/hybrid data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

---

## Dataset

**UNSW-NB15** — Network intrusion detection dataset

Place the dataset at:
```
data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

The CSV is parsed dynamically — column indices are auto-detected from the header. Required columns include: `state`, `proto`, `service`, `spkts`, `dpkts`, `rate`, `sttl`, `dttl`, `sload`, `sloss`, `dloss`, `sjit`, `djit`, `ct_srv_src`, `ct_state_ttl`, `ct_src_dport_ltm`, `label`.

---

## Detection Engine

The `detect()` function scores each record based on traffic features:

**Positive signals** (increase attack score):
- `state=INT` → +5
- `dttl=60/253` → +5
- `proto=unas/sctp/any/gre/ospf` → +5
- `service=pop3/ssl/snmp` → +5
- `ct_state_ttl=2` → +3
- High `sload`, `rate`, `jitter` → +1 to +2 each

**Negative signals** (decrease attack score):
- `sttl=31` → −4
- `dttl=29` → −4
- `ct_state_ttl=0` → −3
- `state=CON/REQ` → −2

Threshold = 4. Score ≥ 4 → classified as attack.

---

## Performance Metrics

All implementations output:
- **Execution time** per pass (with REPEAT_FACTOR=50)
- **Throughput** (records/second)
- **Speedup** vs serial baseline
- **Efficiency** (Speedup / Workers × 100%)
- **Confusion Matrix** (TP, TN, FP, FN)
- **RMSE** (Root Mean Square Error)
- **Accuracy, Precision, Recall, F1 Score**

---

## Project Structure

```
HPC_Network_Analysis/
├── Makefile                          # Build system (all 5 variants)
├── README.md                         # This file
├── PROJECT_GUIDE.md                  # Detailed guide
├── generate_charts.py                # Chart generation script
├── run_all.sh                        # Run all variants
├── run_all_non_cuda_full.sh          # Full non-CUDA run
├── run_all_full.sh                   # Full run script
├── src/
│   ├── serial/
│   │   └── network_analysis_serial.c
│   ├── openmp/
│   │   └── network_analysis_openmp.c
│   ├── pthreads/
│   │   └── network_analysis_pthread.c
│   ├── mpi/
│   │   └── network_analysis_mpi.c
│   ├── hybrid/
│   │   ├── network_analysis_hybrid.c
│   │   └── run_hybrid.sh
│   └── analysis/
│       └── generate_test_data.c
├── data/                             # Dataset directory
├── results/                          # Binaries & timing files
│   └── logs/                         # Execution logs
└── charts/                           # Generated PNG charts
```

---

## Requirements

### Software
- GCC (`gcc`) with C11 support
- OpenMP (`libomp-dev` or built-in)
- MPI (`openmpi-bin`, `libopenmpi-dev`)
- Python 3 with `matplotlib`, `numpy`

### Install on Ubuntu
```bash
sudo apt update
sudo apt install gcc libomp-dev openmpi-bin libopenmpi-dev python3-pip
pip3 install matplotlib numpy
```

---

## Notes

- The `--oversubscribe` flag in `mpirun` allows running more MPI processes than physical CPU cores, which is essential for testing 16-process configurations on systems with fewer cores.
- All parallel implementations use the **identical** `detect()` function, guaranteeing identical confusion matrices across all variants.
- Labels are used **only for validation** (RMSE, accuracy metrics) — the detection engine operates purely on traffic features.
