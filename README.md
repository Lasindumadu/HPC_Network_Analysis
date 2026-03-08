# EC7207 – High Performance Computing
## Group 12 | Network Traffic Anomaly Detection

**Students:** EG/2021/4426 | EG/2021/4432 | EG/2021/4433

---

## Project Overview
Parallel implementation of Z-score anomaly detection on the CICIDS-2017
network traffic dataset using 5 parallelization strategies.

## Dataset
CICIDS-2017 – Download from:
https://www.kaggle.com/datasets/cicdataset/cicids2017

Place all CSV files in `./dataset/` before running.

## Implementations
| # | File | Technique |
|---|------|-----------|
| 1 | `src/serial_baseline.c`  | Single-threaded baseline |
| 2 | `src/openmp_analysis.c`  | OpenMP shared-memory |
| 3 | `src/pthreads_analysis.c`| POSIX threads |
| 4 | `src/mpi_analysis.c`     | MPI distributed memory |
| 5 | `src/hybrid_mpi_omp.c`   | Hybrid MPI + OpenMP |

## Setup

### Install Dependencies
```bash
sudo apt install -y build-essential gcc make libopenmpi-dev openmpi-bin python3 python3-pip
pip3 install pandas numpy scikit-learn matplotlib --break-system-packages
```

### Prepare Data
```bash
python3 src/prepare_data.py
```

### Build
```bash
make all
```

### Run
```bash
# Serial (run first)
./bin/serial ./data/train_data.bin

# OpenMP
./bin/omp ./data/train_data.bin 4

# pthreads
./bin/pthreads ./data/train_data.bin 4

# MPI
mpirun -np 4 --oversubscribe ./bin/mpi ./data/train_data.bin

# Hybrid
OMP_NUM_THREADS=4 mpirun -np 2 --oversubscribe ./bin/hybrid ./data/train_data.bin
```

### Validate & Benchmark
```bash
python3 eval/validate_rmse.py
python3 eval/benchmark.py
```

## Results
Speedup plot saved to `results/speedup_curves.png` after running benchmark.
