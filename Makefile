# Makefile for HPC Network Traffic Analysis
# Course: EC7207 - High Performance Computing

CC = gcc
CFLAGS = -Wall -O2 -std=c11 -lm
OMPFLAGS = -fopenmp
PTHREADFLAGS = -pthread

# Directories
SRC_DIR = src
SERIAL_DIR = $(SRC_DIR)/serial
MPI_DIR = $(SRC_DIR)/mpi
PTHREADS_DIR = $(SRC_DIR)/pthreads
RESULTS_DIR = results

# Executables
SERIAL = $(RESULTS_DIR)/serial
OPENMP = $(RESULTS_DIR)/openmp
PTHREADS = $(RESULTS_DIR)/pthreads
MPI = $(RESULTS_DIR)/mpi
HYBRID = $(RESULTS_DIR)/hybrid

# Source files
SERIAL_SRC = $(SERIAL_DIR)/network_analysis_serial.c
OPENMP_SRC = $(SRC_DIR)/openmp/network_analysis_openmp.c
PTHREADS_SRC = $(PTHREADS_DIR)/network_analysis_pthread.c
MPI_SRC = $(MPI_DIR)/network_analysis_mpi.c
HYBRID_SRC = $(SRC_DIR)/hybrid/network_analysis_hybrid.c

all: directories $(SERIAL) $(OPENMP) $(PTHREADS) $(MPI) $(HYBRID)

directories:
	@mkdir -p $(RESULTS_DIR)

# Serial implementation
$(SERIAL): $(SERIAL_SRC)
	$(CC) $(CFLAGS) -o $@ $< -lm

# OpenMP implementation
$(OPENMP): $(OPENMP_SRC)
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< -lm

# POSIX Threads implementation
$(PTHREADS): $(PTHREADS_SRC)
	$(CC) $(CFLAGS) $(PTHREADFLAGS) -o $@ $< -lm

# MPI implementation
$(MPI): $(MPI_SRC)
	mpicc $(CFLAGS) -o $@ $< -lm

# Hybrid MPI + OpenMP implementation
$(HYBRID): $(HYBRID_SRC)
	mpicc $(CFLAGS) $(OMPFLAGS) -o $@ $< -lm

# Run serial baseline first (required for speedup calculation)
run-serial: $(SERIAL)
	@echo "Running Serial Baseline..."
	@$(SERIAL) data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv

# Run OpenMP
run-openmp: $(OPENMP)
	@echo "Running OpenMP..."
	@for t in 1 2 4 8 16; do \
		echo "  OpenMP with $$t threads"; \
		OMP_NUM_THREADS=$$t $(OPENMP) data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv; \
	done

# Run Pthreads
run-pthreads: $(PTHREADS)
	@echo "Running Pthreads..."
	@for t in 1 2 4 8 16; do \
		echo "  Pthreads with $$t threads"; \
		$(PTHREADS) data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv $$t; \
	done

# Run MPI
run-mpi: $(MPI)
	@echo "Running MPI..."
	@for p in 1 2 4 8 16; do \
		echo "  MPI with $$p processes"; \
		mpirun --allow-run-as-root --oversubscribe -np $$p $(MPI) data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv; \
	done

# Run Hybrid MPI + OpenMP
run-hybrid: $(HYBRID)
	@echo "Running Hybrid..."
	@for combo in "2,2" "2,4" "2,8" "4,2" "4,4" "8,2" "1,16" "2,16" "4,8" "8,4" "16,1"; do \
		np=$$(echo $$combo | cut -d, -f1); \
		nt=$$(echo $$combo | cut -d, -f2); \
		echo "  Hybrid: $$np MPI ranks x $$nt OpenMP threads"; \
		OMP_NUM_THREADS=$$nt mpirun --allow-run-as-root --oversubscribe -np $$np $(HYBRID) data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv; \
	done

# Clean
clean:
	rm -f $(SERIAL) $(OPENMP) $(PTHREADS) $(MPI) $(HYBRID)

.PHONY: all directories clean run-serial run-openmp run-pthreads run-mpi run-hybrid
