# EC7207 HPC Project - Group 12
# Makefile  –  Build all parallel implementations
#
# Prerequisites:
#   gcc      (sudo apt install gcc)
#   mpicc    (sudo apt install openmpi-bin libopenmpi-dev)
#   nvcc     (CUDA Toolkit: https://developer.nvidia.com/cuda-downloads)
#
# Usage:
#   make all         - build everything
#   make serial      - build serial baseline only
#   make omp         - build OpenMP version
#   make pthreads    - build POSIX Threads version
#   make mpi         - build MPI version
#   make cuda        - build CUDA version
#   make hybrid      - build Hybrid MPI+OpenMP version
#   make clean       - remove all binaries

CC      = gcc
MPICC   = mpicc
NVCC    = nvcc

CFLAGS  = -O2 -Wall -Wextra -I./src
LDFLAGS = -lm

# Detect GPU architecture (default sm_75 for T4/Turing; change if needed)
# T4/Turing: sm_75  |  V100: sm_70  |  A100: sm_80  |  RTX 3090: sm_86
CUDA_ARCH = sm_75

BIN_DIR = bin
SRC_DIR = src

.PHONY: all serial omp pthreads mpi cuda hybrid clean data dirs run_serial \
        run_omp run_pthreads run_mpi run_cuda run_hybrid benchmark validate

all: dirs serial omp pthreads mpi cuda hybrid
	@echo ""
	@echo "=== Build complete. Binaries in ./bin/ ==="

dirs:
	@mkdir -p $(BIN_DIR) data results

# ── Serial baseline ───────────────────────────────────────────────────────────
serial: dirs
	$(CC) $(CFLAGS) -o $(BIN_DIR)/serial $(SRC_DIR)/serial_baseline.c $(LDFLAGS)
	@echo "[OK] serial"

# ── OpenMP ────────────────────────────────────────────────────────────────────
omp: dirs
	$(CC) $(CFLAGS) -fopenmp -o $(BIN_DIR)/omp $(SRC_DIR)/openmp_analysis.c $(LDFLAGS)
	@echo "[OK] omp"

# ── POSIX Threads ─────────────────────────────────────────────────────────────
pthreads: dirs
	$(CC) $(CFLAGS) -pthread -o $(BIN_DIR)/pthreads $(SRC_DIR)/pthreads_analysis.c $(LDFLAGS)
	@echo "[OK] pthreads"

# ── MPI ───────────────────────────────────────────────────────────────────────
mpi: dirs
	$(MPICC) $(CFLAGS) -o $(BIN_DIR)/mpi $(SRC_DIR)/mpi_analysis.c $(LDFLAGS)
	@echo "[OK] mpi"

# ── CUDA ──────────────────────────────────────────────────────────────────────
cuda: dirs
	$(NVCC) -O2 -arch=$(CUDA_ARCH) -I./src \
	    -o $(BIN_DIR)/cuda $(SRC_DIR)/cuda_analysis.cu $(LDFLAGS)
	@echo "[OK] cuda (arch=$(CUDA_ARCH))"

# ── Hybrid MPI + OpenMP ───────────────────────────────────────────────────────
hybrid: dirs
	$(MPICC) $(CFLAGS) -fopenmp -o $(BIN_DIR)/hybrid \
	    $(SRC_DIR)/hybrid_mpi_omp.c $(LDFLAGS)
	@echo "[OK] hybrid"

# ── Without CUDA (if no GPU available) ───────────────────────────────────────
no_cuda: dirs serial omp pthreads mpi hybrid
	@echo "=== Built all except CUDA ==="

# ── Data preparation ──────────────────────────────────────────────────────────
data:
	python3 $(SRC_DIR)/prepare_data.py

# ── Individual runs ───────────────────────────────────────────────────────────
run_serial: serial
	$(BIN_DIR)/serial data/train_data.bin

run_omp: omp
	@echo "--- 1 thread ---"
	$(BIN_DIR)/omp data/train_data.bin 1
	@echo "--- 4 threads ---"
	$(BIN_DIR)/omp data/train_data.bin 4
	@echo "--- 8 threads ---"
	$(BIN_DIR)/omp data/train_data.bin 8

run_pthreads: pthreads
	@echo "--- 1 thread ---"
	$(BIN_DIR)/pthreads data/train_data.bin 1
	@echo "--- 4 threads ---"
	$(BIN_DIR)/pthreads data/train_data.bin 4
	@echo "--- 8 threads ---"
	$(BIN_DIR)/pthreads data/train_data.bin 8

run_mpi: mpi
	@echo "--- 1 rank ---"
	mpirun -np 1 $(BIN_DIR)/mpi data/train_data.bin
	@echo "--- 4 ranks ---"
	mpirun -np 4 $(BIN_DIR)/mpi data/train_data.bin

run_cuda: cuda
	$(BIN_DIR)/cuda data/train_data.bin

run_hybrid: hybrid
	@echo "--- 2 ranks x 4 threads ---"
	OMP_NUM_THREADS=4 mpirun -np 2 $(BIN_DIR)/hybrid data/train_data.bin
	@echo "--- 4 ranks x 4 threads ---"
	OMP_NUM_THREADS=4 mpirun -np 4 $(BIN_DIR)/hybrid data/train_data.bin

# ── Benchmark & validate ──────────────────────────────────────────────────────
benchmark:
	python3 eval/benchmark.py

validate:
	python3 eval/validate_rmse.py

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f $(BIN_DIR)/*
	@echo "Cleaned binaries."

clean_data:
	rm -f data/*.bin data/*.txt
	@echo "Cleaned data files."

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "EC7207 HPC Project - Build Targets:"
	@echo "  make all         Build all implementations"
	@echo "  make no_cuda     Build all except CUDA"
	@echo "  make data        Run prepare_data.py"
	@echo "  make run_serial  Run serial baseline"
	@echo "  make run_omp     Run OpenMP (1/4/8 threads)"
	@echo "  make run_mpi     Run MPI (1/4 ranks)"
	@echo "  make run_cuda    Run CUDA"
	@echo "  make run_hybrid  Run Hybrid"
	@echo "  make benchmark   Run full benchmark suite"
	@echo "  make validate    Run RMSE validation"
	@echo "  make clean       Remove binaries"
