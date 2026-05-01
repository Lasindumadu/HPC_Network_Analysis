# HPC Network Traffic and Packet Analysis Project

## Project Overview
- **Course:** EC7207 – High Performance Computing
- **Team Members:**
  - EG/2021/4426 - Bandara AWLM
  - EG/2021/4432 - Bandara KMTON
  - EG/2021/4433 - Bandara LRTD

## Project Description
This project implements parallel computing techniques for analyzing network traffic data from the UNSW-NB15 dataset. The implementations cover various HPC paradigms including shared memory (OpenMP, POSIX Threads), distributed memory (MPI), and hybrid (MPI + OpenMP) approaches, with support for up to **16 threads/processes**.

## Implementations

### 1. Serial (Baseline)
- **File:** `src/serial/network_analysis_serial.c`
- **Description:** Single-threaded baseline implementation with dynamic column detection
- **Features:** Traffic feature-based anomaly detection, RMSE validation, protocol statistics
- **Compilation:** `make results/serial`

### 2. OpenMP (Shared Memory)
- **File:** `src/openmp/network_analysis_openmp.c`
- **Description:** Parallel implementation using OpenMP directives with `#pragma omp parallel for`
- **Features:** 
  - Static schedule for deterministic work distribution
  - Thread-local counters with OpenMP `reduction` clause
  - Configurable thread count (1, 2, 4, 8, 16)
- **Compilation:** `make results/openmp`

### 3. POSIX Threads (Manual Thread Management)
- **File:** `src/pthreads/network_analysis_pthread.c`
- **Description:** Manual thread management using pthreads
- **Features:**
  - Thread-local accumulators (no mutex in hot loop)
  - Results merged after `pthread_join`
  - Configurable thread count (1, 2, 4, 8, 16)
- **Compilation:** `make results/pthreads`

### 4. MPI (Distributed Memory)
- **File:** `src/mpi/network_analysis_mpi.c`
- **Description:** Distributed parallel implementation using MPI
- **Features:**
  - Rank 0 reads CSV and broadcasts data to all processes
  - `MPI_Reduce` for aggregating results
  - Configurable process count (1, 2, 4, 8, 16)
- **Compilation:** `make results/mpi`

### 5. Hybrid MPI + OpenMP
- **File:** `src/hybrid/network_analysis_hybrid.c`
- **Description:** Two-level parallelism combining MPI and OpenMP
- **Features:**
  - MPI distributes data across compute nodes
  - OpenMP parallelizes within each MPI rank
  - `MPI_Init_thread` with `MPI_THREAD_FUNNELED`
  - Supports combinations up to 16 total parallelism (e.g., 4×4, 8×2, 16×1)
- **Compilation:** `make results/hybrid`

## Building the Project

### Using Makefile (Recommended)
```bash
cd HPC_Network_Analysis
make clean
make all          # Build all 5 implementations
```

### Individual Compilation
```bash
# Serial
gcc -Wall -O2 -std=c11 -lm -o results/serial src/serial/network_analysis_serial.c

# OpenMP
gcc -Wall -O2 -std=c11 -fopenmp -lm -o results/openmp src/openmp/network_analysis_openmp.c

# POSIX Threads
gcc -Wall -O2 -std=c11 -pthread -lm -o results/pthreads src/pthreads/network_analysis_pthread.c

# MPI (requires OpenMPI/MPICH)
mpicc -Wall -O2 -std=c11 -lm -o results/mpi src/mpi/network_analysis_mpi.c

# Hybrid MPI + OpenMP
mpicc -Wall -O2 -std=c11 -fopenmp -lm -o results/hybrid src/hybrid/network_analysis_hybrid.c
```

## Running the Implementations

### Run All Variants
```bash
chmod +x run_all.sh
./run_all.sh
```

### Run Serial
```bash
./results/serial data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

### Run OpenMP
```bash
# With 4 threads
OMP_NUM_THREADS=4 ./results/openmp data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv

# With 16 threads
OMP_NUM_THREADS=16 ./results/openmp data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

### Run POSIX Threads
```bash
# With 4 threads
./results/pthreads data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv 4

# With 16 threads
./results/pthreads data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv 16
```

### Run MPI
```bash
# With 4 processes
mpirun --allow-run-as-root --oversubscribe -np 4 ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv

# With 16 processes
mpirun --allow-run-as-root --oversubscribe -np 16 ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

### Run Hybrid
```bash
# 4 MPI ranks × 4 OpenMP threads = 16 total parallelism
OMP_NUM_THREADS=4 mpirun --allow-run-as-root --oversubscribe -np 4 ./results/hybrid data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv

# 2 MPI ranks × 8 OpenMP threads = 16 total parallelism
OMP_NUM_THREADS=8 mpirun --allow-run-as-root --oversubscribe -np 2 ./results/hybrid data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
```

### Run Hybrid Configurations Script
```bash
chmod +x src/hybrid/run_hybrid.sh
./src/hybrid/run_hybrid.sh
```

## Generate Charts
```bash
python3 generate_charts.py
```
Forces re-run of all programs and regenerates charts:
```bash
python3 generate_charts.py --rerun
```

## Dataset

### UNSW-NB15 Dataset
The UNSW-NB15 dataset is a comprehensive network intrusion detection dataset.

**Download from:** https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

**Required files:**
- `UNSW_NB15_training-set.csv`

Place the dataset file in the `data/` directory.

### Dataset Format
The implementation auto-detects column indices from the CSV header. Key fields used:
- `state` — Connection state
- `proto` — Protocol
- `service` — Network service
- `spkts`, `dpkts` — Source/destination packets
- `rate` — Packet rate
- `sttl`, `dttl` — Source/destination TTL
- `sload` — Source load
- `sloss`, `dloss` — Packet loss
- `sjit`, `djit` — Jitter
- `ct_srv_src` — Connection count to same service
- `ct_state_ttl` — Connection state-TTL count
- `ct_src_dport_ltm` — Source-port connection count
- `label` — Ground truth (0=normal, 1=attack)

## Performance Metrics

### Speedup Calculation
```
Speedup = T_serial / T_parallel
```

### Efficiency Calculation
```
Efficiency = Speedup / Number_of_Workers × 100%
```

### Expected Results (UNSW-NB15, ~82,332 records, Repeat ×50)
| Implementation | Workers | Expected Speedup | Expected Efficiency |
|---------------|---------|------------------|---------------------|
| Serial        | 1       | 1.0×             | 100.0%              |
| OpenMP        | 4       | 3.5–4.5×         | 90–110%             |
| OpenMP        | 8       | 5.0–6.5×         | 65–80%              |
| OpenMP        | 16      | 6.0–8.0×         | 40–50%              |
| Pthreads      | 4       | 3.5–4.5×         | 90–110%             |
| Pthreads      | 8       | 5.0–6.5×         | 65–80%              |
| Pthreads      | 16      | 6.0–8.0×         | 40–50%              |
| MPI           | 4       | 3.5–4.5×         | 90–110%             |
| MPI           | 8       | 5.5–7.0×         | 70–90%              |
| MPI           | 16      | 7.0–10.0×        | 45–65%              |
| Hybrid        | 4×4     | 5.0–7.0×         | 35–45%              |
| Hybrid        | 8×2     | 6.0–8.0×         | 40–50%              |

## Anomaly Detection Features

The implementations detect:
1. **Traffic Volume Spikes** — Unusually high packet counts
2. **High-Risk Protocols/Services** — Protocols with high attack ratios
3. **Suspicious TTL Patterns** — TTL values strongly correlated with attacks
4. **Connection State Anomalies** — States like INT with high attack probability

## Accuracy Validation

All implementations compute:
- **Confusion Matrix** (TP, TN, FP, FN)
- **RMSE** (Root Mean Square Error)
- **Accuracy, Precision, Recall, F1 Score**

> **Important:** Labels are used **ONLY for validation**. The `detect()` engine operates exclusively on traffic features — this is genuine unsupervised anomaly detection.

## Project Structure
```
HPC_Network_Analysis/
├── Makefile
├── README.md
├── PROJECT_GUIDE.md
├── generate_charts.py
├── run_all.sh
├── run_all_non_cuda_full.sh
├── run_all_full.sh
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
│   ├── cuda/
│   │   └── network_analysis_cuda.cu
│   └── analysis/
│       └── generate_test_data.c
├── data/
│   └── UNSW_NB15_training-set.csv/
├── results/
│   ├── serial
│   ├── openmp
│   ├── pthreads
│   ├── mpi
│   ├── hybrid
│   └── logs/
└── charts/
    ├── speedup.png
    ├── efficiency.png
    ├── execution_time.png
    ├── throughput.png
    └── all_charts.png
```

## Requirements

### Software Requirements
- GCC compiler (7.0+ with C11 support)
- OpenMP support (`libomp-dev` or built-in)
- POSIX Threads (standard on Linux)
- MPI implementation (OpenMPI or MPICH)
- Python 3 with `matplotlib`, `numpy`

### Hardware Requirements
- Multi-core CPU (4+ cores recommended)
- 8GB RAM minimum
- 10GB disk space

## Troubleshooting

### MPI Not Installed
```bash
# Install OpenMPI
sudo apt install openmpi-bin libopenmpi-dev

# Verify installation
mpicc --version
mpirun --version
```

### OpenMP Not Working
```bash
# Install OpenMP library
sudo apt install libomp-dev

# Verify
echo '#include <omp.h>' | gcc -fopenmp -x c - -o /dev/null
```

### Permission Denied on Run Scripts
```bash
chmod +x run_all.sh run_all_non_cuda_full.sh run_all_full.sh src/hybrid/run_hybrid.sh
```

### Dataset Not Found
Ensure the dataset is in the correct location:
```bash
ls -la data/
```

## Performance Analysis

### Strong Scaling
Test with fixed problem size, increasing worker count:
```bash
# OpenMP
for t in 1 2 4 8 16; do
    OMP_NUM_THREADS=$t ./results/openmp data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
done

# MPI
for p in 1 2 4 8 16; do
    mpirun --allow-run-as-root --oversubscribe -np $p ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
done
```

### Weak Scaling
Test with problem size proportional to worker count by varying REPEAT_FACTOR.

## References

1. UNSW-NB15 Dataset: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
2. OpenMP: https://www.openmp.org/
3. MPI Standard: https://www.mpi-forum.org/
4. POSIX Threads: https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/pthread.h.html
5. Chart.js: https://www.chartjs.org/
