# HPC Network Analysis — Execution TODO (16-Worker Support + Missing Parts)

## Plan Items
- [x] 1. Fix Makefile (add OpenMP, Hybrid targets)
- [x] 2. Update `run_all.sh` (16 workers, pthreads loop, hybrid configs)
- [x] 3. Update `run_all_non_cuda_full.sh` (16 workers, fix pthreads, hybrid)
- [x] 4. Update `run_all_full.sh` (reconstruct with 16-worker support)
- [x] 5. Update `src/hybrid/run_hybrid.sh` (more combos, --oversubscribe)
- [x] 6. Update `generate_charts.py` (add Pthreads + Hybrid, extend to 16)
- [x] 7. Update `README.md` (comprehensive docs)
- [x] 8. Update `PROJECT_GUIDE.md` (correct paths, 16-worker info, Hybrid section)
- [x] 9. Update `TODO.md`, `TODO_steps.md`, `TODO_run_non_cuda.md`
- [x] 10. Build test with `make clean && make all`
