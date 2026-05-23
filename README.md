# HPC Network Traffic Analysis
**EC7207 — High Performance Computing**  
**Team:** EG/2021/4426 · EG/2021/4432 · EG/2021/4433

---

## Overview

This project implements a **network intrusion detection system** across six parallelisation strategies and benchmarks their performance on the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) dataset (700,001 records).

Each implementation applies an identical scoring engine to classify network flows as *normal* or *attack* using traffic features only — no labels used during detection. Labels are used only for validation (accuracy, confusion matrix).

---

## Implementations

| Implementation | Source | Parallelism Model |
|---|---|---|
| Serial | `src/serial/network_analysis_serial.c` | Baseline — single thread |
| OpenMP | `src/openmp/network_analysis_openmp.c` | Shared memory, fork-join |
| Pthreads | `src/pthreads/network_analysis_pthread.c` | Shared memory, manual threads |
| MPI | `src/mpi/network_analysis_mpi.c` | Distributed memory, message passing |
| Hybrid | `src/hybrid/network_analysis_hybrid.c` | MPI ranks × OpenMP threads |
| CUDA | `src/cuda/network_analysis_cuda.cu` | GPU — one thread per record |

All six produce **identical confusion matrix results** — the scoring engine (`detect()`) is deterministic and stateless per record.

---

## Dataset

The UNSW-NB15 dataset is **not included** in this repository (157 MB exceeds the submission limit).

**Download:** https://drive.google.com/drive/folders/1tqNgeGTsgRTTDDsr46wnJ4Gt4gdxUfzN?usp=sharing

After downloading, place the file at:

```
HPC_Network_Analysis/
└── data/
    └── UNSW-NB15_1.csv/
        └── UNSW-NB15_1_with_header.csv   ← place here
```

---

## Quick Start

### 1. Install Dependencies

```bash
chmod +x setup.sh && ./setup.sh
```

Or manually:

```bash
sudo apt update
sudo apt install -y gcc libopenmpi-dev mpirun python3-pip
pip3 install -r webapp/requirements.txt
```

### 2. Build All Implementations

```bash
make all
```

To build a single implementation:

```bash
make results/serial
make results/openmp
make results/pthreads
make results/mpi
make results/hybrid
make results/cuda      # requires nvcc + CUDA-capable GPU
```

### 3. Run Benchmarks

**All at once (shell script):**

```bash
chmod +x run_all.sh && ./run_all.sh
```

**Individual runs:**

```bash
# Serial baseline (run this first — other implementations read its time)
./results/serial data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv

# OpenMP — set thread count via environment variable
OMP_NUM_THREADS=8 ./results/openmp data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv

# Pthreads — thread count is a command-line argument
./results/pthreads data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv 8

# MPI — process count via mpirun
mpirun --allow-run-as-root -np 4 ./results/mpi data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv

# Hybrid — MPI ranks × OpenMP threads (example: 4 × 4 = 16-way)
OMP_NUM_THREADS=4 mpirun --allow-run-as-root -np 4 ./results/hybrid data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv

# CUDA — optional block size argument (default 256)
./results/cuda data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv 256
```

### 4. Web Dashboard

```bash
cd webapp && python3 app.py
# Open http://localhost:5000
```

The dashboard lets you run any implementation interactively, view live output, and plot speedup/efficiency/throughput charts.

### 5. Generate Charts (PNG)

```bash
python3 generate_charts_all.py     # generates all chart variants
# or individually:
python3 generate_charts.py         # uses log files, CUDA simulated if no GPU
python3 generate_charts2.py        # includes hardcoded Tesla T4 CUDA results
python3 generate_charts3.py        # CPU-only charts (no CUDA)
```

Charts are saved to `charts/`.

### 6. Verify Correctness

```bash
python3 verify_results.py
```

Compares all parallel implementations against the serial baseline. All confusion matrix values must match exactly.

---

## Results Summary (UNSW-NB15, 700,001 records, Repeat ×50)

| Implementation | Workers | Time (s/pass) | Speedup | Efficiency | Throughput (rec/s) |
|---|---|---|---|---|---|
| Serial | 1 | 0.4320 | 1.00× | 100% | 1,620,360 |
| OpenMP | 8 | 0.0758 | 7.09× | 88.6% | 9,234,284 |
| Pthreads | 8 | 0.0632 | 8.50× | 106.3% | 11,075,275 |
| Hybrid | 8×16 | 0.0095 | 56.81× | 44.4% | 8,701,575 |
| CUDA (Tesla T4) | GPU | 0.0022 | 532.40× | — | 321,824,052 |

**Detection accuracy:** 97.072% on UNSW-NB15 (EXCELLENT)

---

## Project Structure

```
HPC_Network_Analysis/
├── src/
│   ├── serial/
│   │   ├── network_analysis_serial.c
│   │   └── analyze_features.py        # feature distribution analysis
│   ├── openmp/
│   │   └── network_analysis_openmp.c
│   ├── pthreads/
│   │   └── network_analysis_pthread.c
│   ├── mpi/
│   │   └── network_analysis_mpi.c
│   ├── hybrid/
│   │   └── network_analysis_hybrid.c
│   └── cuda/
│       └── network_analysis_cuda.cu
├── webapp/
│   ├── app.py                         # Flask backend
│   ├── requirements.txt
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/                              # ← dataset goes here (not in repo)
├── results/                           # compiled binaries + timing files
│   └── logs/                          # per-run log files read by charts
├── charts/                            # generated PNG charts
├── docs/
│   └── design_decisions.md            # scoring weight rationale
├── Makefile
├── run_all.sh
├── setup.sh
├── cleanup.sh
├── generate_charts.py
├── generate_charts2.py
├── generate_charts3.py
├── generate_charts_all.py
├── verify_results.py
└── README.md
```

---

## Compile-Time Constants (`src/config.h`)

Key parameters shared across all implementations:

| Constant | Value | Description |
|---|---|---|
| `REPEAT_FACTOR` | 50 | Dataset passes per run (for stable timing) |
| `ATTACK_THRESHOLD` | 4 | Score ≥ this → classified as attack |
| `MAX_RECORDS` | 750,000 | Buffer size for in-memory loading |
| `MAX_FIELDS` | 50 | Max CSV columns per row |
| `FIELD_LEN` | 64 | Max characters per CSV field |

See `docs/design_decisions.md` for the rationale behind these values.

---

## Environment

- **CPU benchmarks:** VMware Ubuntu 22.04, GCC 11, OpenMPI 4.1
- **CUDA benchmarks:** Google Colab, Tesla T4 GPU, CUDA 12.2, nvcc 12.2
- **Python:** 3.10+, Flask 2.x, matplotlib 3.x, numpy 1.x
