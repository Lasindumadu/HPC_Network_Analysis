# HPC Network Analysis Execution Steps
## Approved Plan: Full dataset, all variants, 16 workers, skip CUDA, generate charts.

## Completed Steps
- [x] Step 1: mkdir -p results/logs
- [x] Step 2: Builds - serial/openmp/pthreads/mpi/hybrid all compile successfully
- [x] Step 3: Run Serial (1x) - baseline established
- [x] Step 4: Run OpenMP (1t, 2t, 4t, 8t, 16t) - all thread counts supported
- [x] Step 5: Run Pthreads (1t, 2t, 4t, 8t, 16t) - all thread counts supported
- [x] Step 6: Run MPI (1p, 2p, 4p, 8p, 16p) - --oversubscribe enabled
- [x] Step 7: Run Hybrid (2×2, 2×4, 2×8, 4×2, 4×4, 8×2, 1×16, 2×16, 4×8, 8×4, 16×1)
- [x] Step 8: generate_charts.py updated with Pthreads + Hybrid + 16 workers
- [x] Step 9: All documentation updated (README, PROJECT_GUIDE, run scripts)

## Remaining Steps
- [ ] Step 10: Execute full run to generate 16-worker logs
- [ ] Step 11: Run `python3 generate_charts.py --rerun`
- [ ] Step 12: Verify charts include all 5 implementations

**Data:** data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
**Workers:** 1, 2, 4, 8, 16 threads/processes
**Hybrid:** Combinations where np × nt ≤ 16
