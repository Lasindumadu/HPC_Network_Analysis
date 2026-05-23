# HPC Network Traffic Analysis
## EC7207 — High Performance Computing

**Team Members:**
| Index No. | Name |
|-----------|------|
| EG/2021/4426 | Bandara AWMLM |
| EG/2021/4432 | Bandara KMTON |
| EG/2021/4433 | Bandara LRTD |

---

## Project Overview

Parallel computing techniques applied to real-time network traffic anomaly detection using the **UNSW-NB15 dataset** (700,001 records). Six implementations — Serial, OpenMP, POSIX Threads, MPI, Hybrid MPI+OpenMP, and CUDA — are benchmarked for speedup, efficiency, and classification accuracy.

Detection runs purely on traffic features. Ground-truth labels are used **only for validation** (accuracy, RMSE, confusion matrix).

---

## Implementations

| Variant | Paradigm | Source File | Workers |
|---------|----------|-------------|---------|
| Serial | Baseline | `src/serial/network_analysis_serial.c` | 1 |
| OpenMP | Shared Memory | `src/openmp/network_analysis_openmp.c` | 1, 2, 4, 8, 16 |
| Pthreads | Shared Memory | `src/pthreads/network_analysis_pthread.c` | 1, 2, 4, 8, 16 |
| MPI | Distributed Memory | `src/mpi/network_analysis_mpi.c` | 1, 2, 4, 8, 16 |
| Hybrid | MPI + OpenMP | `src/hybrid/network_analysis_hybrid.c` | 2×2, 2×4, 4×2, 4×4, 8×2, … |
| CUDA | GPU (massively parallel) | `src/cuda/network_analysis_cuda.cu` | GPU blocks × threads |

---

## Measured Performance Results

**Dataset:** `UNSW-NB15_1_with_header.csv` — 700,001 records, REPEAT_FACTOR = 50

| Implementation | Workers | Time (s/pass) | Speedup | Efficiency |
|----------------|---------|---------------|---------|------------|
| Serial | 1 | 0.4405 | 1.00× | 100% |
| OpenMP | 1 | 0.298 | 1.48× | 148% |
| OpenMP | 2 | 0.139 | 3.16× | 158% |
| OpenMP | 4 | 0.078 | 5.64× | 141% |
| OpenMP | 8 | 0.065 | 6.74× | 84% |
| OpenMP | 16 | 0.063 | 7.02× | 44% |
| Pthreads | 1 | 0.298 | 1.48× | 148% |
| Pthreads | 2 | 0.151 | 2.91× | 146% |
| Pthreads | 4 | 0.084 | 5.22× | 131% |
| Pthreads | 8 | 0.061 | 7.19× | 90% |
| Pthreads | 16 | 0.058 | 7.56× | 47% |
| MPI | 1 | 0.319 | 1.38× | 138% |
| MPI | 2 | 0.166 | 2.66× | 133% |
| MPI | 4 | 0.092 | 4.80× | 120% |
| MPI | 8 | 0.071 | 6.17× | 77% |
| MPI | 16 | 0.074 | 5.97× | 37% |
| Hybrid 2×2 | 4 | 0.115 | 3.83× | 96% |
| Hybrid 4×2 | 8 | 0.065 | 6.82× | 85% |
| Hybrid 4×4 | 16 | 0.071 | 6.20× | 39% |
| CUDA | Tesla T4 | 0.0022 | 532.40× | — |

> Super-linear speedups (>100%) at low worker counts occur because parallel implementations load the full dataset into RAM once, while serial re-reads from disk each repeat pass.

**Accuracy (all implementations, identical):**
| Metric | Value |
|--------|-------|
| Accuracy | 97.072% |
| Precision | 53.154% |
| Recall | 65.087% |
| F1 Score | 58.518% |
| RMSE | 0.171126 |
| TP / TN / FP / FN | 14,459 / 665,043 / 12,743 / 7,756 |

---

## Detection Engine

Each record is scored using traffic features only. Threshold = 4; score ≥ 4 → classified as attack.

**Score rules (same across all 6 implementations):**

| Feature | Condition | Score |
|---------|-----------|-------|
| state | INT | +5 |
| dttl | 60 or 253 | +5 |
| proto | unas/sctp/any/gre/ospf | +5 |
| service | pop3/ssl/snmp | +5 |
| ct_state_ttl | == 2 | +3 |
| dttl | == 0 | +3 |
| sttl | 254 or 255 | +2 |
| service | dns | +2 |
| ct_src_dport_ltm | > 10 | +2 |
| sload | > 1M / 10M / 50M | +1 each |
| rate | > 100K / 166K | +1 each |
| dpkts==0 && spkts>2 | | +1 |
| ct_srv_src | > 20 | +1 |
| sjit or djit | > 1000 | +1 |
| sttl | == 31 | −4 |
| dttl | == 29 | −4 |
| ct_state_ttl | == 0 | −3 |
| state | CON or REQ | −2 |

---

## Quick Start

### Prerequisites

```bash
# Ubuntu / WSL
sudo apt update
sudo apt install -y gcc libomp-dev openmpi-bin libopenmpi-dev python3 python3-pip

# CUDA (requires NVIDIA GPU)
sudo apt install nvidia-cuda-toolkit
# or download from https://developer.nvidia.com/cuda-downloads

pip3 install flask matplotlib numpy
```

### Build

```bash
make clean
make all        # builds Serial, OpenMP, Pthreads, MPI, Hybrid, CUDA
```

**Individual build commands:**
```bash
# Serial
gcc -Wall -O2 -std=c11 -lm -o results/serial src/serial/network_analysis_serial.c

# OpenMP
gcc -Wall -O2 -std=c11 -fopenmp -lm -o results/openmp src/openmp/network_analysis_openmp.c

# Pthreads
gcc -Wall -O2 -std=c11 -pthread -lm -o results/pthreads src/pthreads/network_analysis_pthread.c

# MPI
mpicc -Wall -O2 -std=c11 -lm -o results/mpi src/mpi/network_analysis_mpi.c

# Hybrid MPI + OpenMP
mpicc -Wall -O2 -std=c11 -fopenmp -lm -o results/hybrid src/hybrid/network_analysis_hybrid.c

# CUDA
nvcc -O2 -std=c++11 -o results/cuda src/cuda/network_analysis_cuda.cu
```

---

## Run via Web Dashboard (Recommended)

```bash
cd webapp
pip3 install -r requirements.txt
python3 app.py
# Open http://localhost:5000
```

Click **Run All Benchmarks** — all six implementations run in sequence, charts update live with speedup, efficiency, execution time, and throughput graphs.

---

## Run via Terminal

```bash
DATA="data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv"

# 1. Serial (run first — generates serial_time.txt for speedup baseline)
./results/serial "$DATA"

# 2. OpenMP
for t in 1 2 4 8 16; do
  OMP_NUM_THREADS=$t ./results/openmp "$DATA"
done

# 3. Pthreads
for t in 1 2 4 8 16; do
  ./results/pthreads "$DATA" $t
done

# 4. MPI
for p in 1 2 4 8 16; do
  mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi "$DATA"
done

# 5. Hybrid
for cfg in 2x2 2x4 2x8 4x2 4x4 8x2 1x16 2x16 4x8 8x4 16x1; do
  np=$(echo $cfg | cut -dx -f1)
  nt=$(echo $cfg | cut -dx -f2)
  OMP_NUM_THREADS=$nt mpirun --allow-run-as-root --oversubscribe -np $np ./results/hybrid "$DATA"
done

# 6. CUDA
./results/cuda "$DATA" 256
```

### Automated Script

```bash
chmod +x run_all.sh
./run_all.sh
```

### Generate Static Charts

```bash
python3 generate_charts.py
# Charts saved to charts/
```

### Verify Correctness

```bash
python3 verify_results.py
```

All implementations must produce identical accuracy metrics (97.072%) — this confirms the parallel detection engines are all correct.

---

## Dataset

**UNSW-NB15** — network intrusion detection benchmark dataset · 700,001 records · ~157 MB

> **The dataset is NOT included in this submission** (file size 157 MB > 10 MB limit).
> Download it from Google Drive and place it at the path below before running.
>
> **Google Drive:** https://drive.google.com/drive/folders/1tqNgeGTsgRTTDDsr46wnJ4Gt4gdxUfzN?usp=sharing
>
> See **[IMPORTANT_NOTE.md](IMPORTANT_NOTE.md)** for full setup instructions.

Primary dataset path (required):
```
data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv
```

The CSV header is parsed dynamically — no hardcoded column indices. Required fields: `state`, `proto`, `service`, `spkts`, `dpkts`, `rate`, `sttl`, `dttl`, `sload`, `sloss`, `dloss`, `sjit`, `djit`, `ct_srv_src`, `ct_state_ttl`, `ct_src_dport_ltm`, `label`.

---

## CUDA Implementation

The CUDA kernel runs one thread per record — massively parallel scoring on GPU.

**Compilation:**
```bash
nvcc -O2 -std=c++11 -o results/cuda src/cuda/network_analysis_cuda.cu
```

**Run:**
```bash
./results/cuda data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv 256
```

The block size argument (256) sets CUDA threads per block. Typical values: 128, 256, 512.

---

## Project Structure

```
HPC_Network_Analysis/
├── Makefile
├── README.md
├── run_all.sh
├── generate_charts.py
├── verify_results.py
├── src/
│   ├── serial/network_analysis_serial.c
│   ├── openmp/network_analysis_openmp.c
│   ├── pthreads/network_analysis_pthread.c
│   ├── mpi/network_analysis_mpi.c
│   ├── hybrid/network_analysis_hybrid.c
│   └── cuda/network_analysis_cuda.cu
├── data/                              ← NOT in submission (157 MB > limit)
│   └── UNSW-NB15_1.csv/               ← download from Google Drive
│       └── UNSW-NB15_1_with_header.csv
├── results/
│   ├── serial / openmp / pthreads / mpi / hybrid / cuda
│   ├── serial_time.txt
│   └── logs/
├── charts/
│   ├── speedup.png
│   ├── efficiency.png
│   ├── execution_time.png
│   ├── throughput.png
│   └── all_charts.png
└── webapp/
    ├── app.py
    ├── requirements.txt
    ├── templates/index.html
    └── static/
        ├── css/style.css
        └── js/app.js
```

---

## Performance Analysis

### Speedup & Efficiency

```
Speedup    = T_serial / T_parallel
Efficiency = (Speedup / Workers) × 100%
```

### Strong Scaling

All tests use a fixed problem size (700,001 records × 50 passes) with increasing worker count. Super-linear speedup at low counts is expected — parallel variants cache the dataset in RAM across passes, while serial re-reads from disk.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mpicc: command not found` | `sudo apt install openmpi-bin libopenmpi-dev` |
| `libomp: not found` | `sudo apt install libomp-dev` |
| `nvcc: command not found` | `sudo apt install nvidia-cuda-toolkit` |
| Permission denied on scripts | `chmod +x run_all.sh` |
| Dataset not found | Download from [Google Drive](https://drive.google.com/drive/folders/1tqNgeGTsgRTTDDsr46wnJ4Gt4gdxUfzN?usp=sharing) → place at `data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv` |
| MPI fails at 16 processes | Add `--oversubscribe` flag to mpirun |
| CUDA out of memory | Reduce block size: `./results/cuda "$DATA" 128` |

---

## Requirements

### Build Dependencies
```bash
sudo apt install gcc libomp-dev openmpi-bin libopenmpi-dev nvidia-cuda-toolkit
```

### Python Dependencies
```bash
pip3 install flask matplotlib numpy
```

### Hardware
- Multi-core CPU (4+ cores recommended)
- NVIDIA GPU with CUDA compute capability ≥ 3.5 (for CUDA implementation)
- 8 GB RAM minimum
- 10 GB disk space

---

## References

1. UNSW-NB15 Dataset — https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
2. OpenMP — https://www.openmp.org/
3. MPI Standard — https://www.mpi-forum.org/
4. CUDA Toolkit — https://developer.nvidia.com/cuda-toolkit
5. Chart.js — https://www.chartjs.org/
