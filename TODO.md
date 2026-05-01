# HPC Network Analysis - Execution TODO

## Plan Breakdown (Approved: full dataset, run all variants, 16 workers supported)
- [x] Step 1: mkdir -p results/logs
- [x] Step 2: Compile all variants (Serial, OpenMP, Pthreads, MPI, Hybrid)
- [x] Step 3: Run Serial (1x)
- [x] Step 4: Run OpenMP (1, 2, 4, 8, 16 threads)
- [x] Step 5: Run Pthreads (1, 2, 4, 8, 16 threads)
- [x] Step 6: Run MPI (1, 2, 4, 8, 16 processes) with --oversubscribe
- [x] Step 7: Run Hybrid (np×nt combos up to 16 total parallelism)
- [x] Step 8: Skip CUDA (no GPU/nvcc - noted in documentation)
- [x] Step 9: Update generate_charts.py with Pthreads + Hybrid support
- [x] Step 10: Update all documentation (README, PROJECT_GUIDE, run scripts)
- [ ] Step 11: Execute full run to generate 16-worker logs
- [ ] Step 12: python3 generate_charts.py --rerun
- [ ] Step 13: Verify results/charts

**Data:** data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
**Notes:** 
- Serial first for baseline. MPI uses --allow-run-as-root --oversubscribe for 16 processes.
- All run scripts now support 1, 2, 4, 8, 16 workers.
- Hybrid configs: 2×2, 2×4, 2×8, 4×2, 4×4, 8×2, 1×16, 2×16, 4×8, 8×4, 16×1
