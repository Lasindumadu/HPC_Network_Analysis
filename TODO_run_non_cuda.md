# TODO: Run All Non-CUDA HPC Network Analysis Variants (Full Dataset, 16 Workers)

## Steps:
1. [x] Create and verify `run_all_non_cuda_full.sh` (supports 1,2,4,8,16 workers)
2. [x] `make clean && make all` (builds all 5 implementations)
3. [ ] `chmod +x run_all_non_cuda_full.sh && ./run_all_non_cuda_full.sh`
4. [ ] `python3 generate_charts.py --rerun`
5. [ ] Check `charts/` for results with all 5 implementations

## 16-Worker Support Added
- **OpenMP:** `OMP_NUM_THREADS=16 ./results/openmp ...`
- **Pthreads:** `./results/pthreads ... 16`
- **MPI:** `mpirun --allow-run-as-root --oversubscribe -np 16 ./results/mpi ...`
- **Hybrid:** `OMP_NUM_THREADS=8 mpirun --allow-run-as-root --oversubscribe -np 2 ./results/hybrid ...`

## Hybrid Configurations (np × nt ≤ 16)
| Config | MPI Ranks | OpenMP Threads | Total Parallelism |
|--------|-----------|----------------|-------------------|
| 2×2    | 2         | 2              | 4                 |
| 2×4    | 2         | 4              | 8                 |
| 2×8    | 2         | 8              | 16                |
| 4×2    | 4         | 2              | 8                 |
| 4×4    | 4         | 4              | 16                |
| 8×2    | 8         | 2              | 16                |
| 1×16   | 1         | 16             | 16                |
| 2×16   | 2         | 16             | 32*               |
| 4×8    | 4         | 8              | 32*               |
| 8×4    | 8         | 4              | 32*               |
| 16×1   | 16        | 1              | 16                |

*\*Note: Configurations with total > 16 may be limited by available cores.*

**Notes:**  
- Excludes CUDA (`network_analysis_cuda.cu`).  
- Logs in `results/logs/`.  
- Dataset at `data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv` (adjust if needed).
- `--oversubscribe` flag enables running more MPI processes than physical cores.
