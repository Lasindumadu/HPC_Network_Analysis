# Design Decisions
**EC7207 — HPC Network Traffic Analysis**

This document explains the key design choices made during development — why specific values were chosen, what alternatives were considered, and what tradeoffs were accepted.

---

## 1. Scoring Algorithm Design

### Why a Weighted Score Instead of ML?

The goal of this project is to benchmark **parallel performance**, not to build the most accurate classifier. A weighted rule-based scorer was chosen because:

- It is **CPU-bound and embarrassingly parallel** — each record scores independently with no shared state
- It produces **deterministic, reproducible results** — every implementation must produce the same confusion matrix
- It is **fast to implement and verify** — we can inspect exactly why a record is classified one way or the other
- It runs identically on CPU and GPU — the CUDA kernel is a direct port of the C `detect()` function

A neural network or random forest would have introduced data-loading and inference dependencies that complicate parallelism measurement.

### How Scoring Weights Were Derived

We ran `src/serial/analyze_features.py` on the full UNSW-NB15 dataset to compute per-feature attack rates. The script outputs tables like:

```
State     Att     Norm    Att%
INT       154847   23271   86.9%   ATTACK
CON        1940    36766    5.0%   NORMAL
REQ         434     5492    7.3%   NORMAL
```

Features with attack rates above ~80% received positive weights; features below ~20% received negative weights. Weight magnitude corresponds roughly to how extreme the ratio is:

| Attack Rate | Weight |
|---|---|
| ≥ 99% | +5 |
| 80–98% | +3 to +5 |
| 70–80% | +2 |
| signal-based (scan, jitter) | +1 |
| ≤ 5% | −3 to −4 |
| 5–10% | −2 |

The threshold of **4** was chosen after testing values 3–6 on the dataset. A threshold of 4 gave the best balance between false positives and false negatives on UNSW-NB15 file 1 (97.072% accuracy).

---

## 2. REPEAT_FACTOR = 50

Each implementation processes the dataset 50 times per run. The reported single-pass time is `total_elapsed / 50`.

**Why not just run once?**

On our test machine (VMware Ubuntu), a single serial pass takes ~0.43 s. That is long enough to measure, but a single measurement has high variance — the OS scheduler, memory caching on first pass, and disk I/O all introduce noise.

Running 50 passes:
- Amortises the cold-start I/O cost (OpenMP/Pthreads/MPI load everything into RAM on first pass; subsequent passes are pure compute)
- Gives ~21 s of total serial time — enough for a stable mean
- Makes the parallel speedup numbers more comparable across runs

We tested REPEAT_FACTOR values of 10, 20, 50, and 100. The coefficient of variation (std/mean) of single-pass time dropped below 2% at 50 passes and did not improve meaningfully at 100.

---

## 3. In-Memory Loading Strategy (OpenMP, Pthreads, MPI, Hybrid)

The parallel implementations load all records into a `char[MAX_RECORDS][MAX_LINE]` array before starting the timed loop. The serial implementation re-opens and re-reads the file on each pass.

**Why the difference?**

- **Serial**: reading line-by-line is simpler and ensures the baseline time includes I/O, not just compute. This is conservative — it makes the serial time slightly larger, which if anything *reduces* the reported speedup.
- **Parallel**: loading once into RAM and then processing from memory on each pass isolates the CPU parallelism from disk I/O. This is more representative of a real-world pipeline where data is already in memory.

**MPI specifics**: Rank 0 loads all data and distributes slices via `MPI_Send`. This is a common pattern for datasets that fit in RAM. For datasets that do not fit, each rank would read its own slice directly from disk using `MPI_File_read_at`.

---

## 4. Row Parser Design (Stack-Based, Zero-Malloc)

All implementations use a `Row` struct where each field is stored as a `char[64]` on the stack:

```c
typedef struct {
    char buf[MAX_FIELDS][FIELD_LEN];
    int  n;
} Row;
```

**Why not `strsep` / `strtok` / heap allocation?**

- `strtok` is not thread-safe
- `strsep` modifies the input string — cannot use on shared read-only data
- Heap allocation (`malloc` per row) causes lock contention in multithreaded code — all threads compete for the malloc lock

The stack-based parser has no shared mutable state, so it scales linearly with thread count. Each thread parses its own copy of a row into its own `Row` struct on the stack.

---

## 5. MPI Communication Pattern

We chose **rank 0 reads, then distributes** rather than **each rank reads its own slice**:

- Simpler to implement correctly (no MPI-IO required)
- Dataset fits in RAM (~157 MB), so rank 0 can hold all records
- Distribution time is small relative to compute time at our record count

The slice assignment uses:
```c
int base     = total_lines / nprocs;
int rem      = total_lines % nprocs;
int my_count = base + (rank < rem ? 1 : 0);
```
This distributes remainder records one-per-rank to lower-numbered ranks, ensuring no rank has more than one extra record. Maximum imbalance is 1 record regardless of dataset size or process count.

---

## 6. Hybrid Thread Safety

The Hybrid implementation uses `MPI_THREAD_FUNNELED` (only the main thread makes MPI calls). This is sufficient because:

- MPI calls happen only *before* and *after* the OpenMP parallel region
- The `detect()` function called inside the OpenMP loop has no MPI calls
- `MPI_THREAD_MULTIPLE` would be needed only if OpenMP threads themselves called MPI — they do not

`MPI_THREAD_FUNNELED` is supported by all common MPI implementations and is safer to request than `MPI_THREAD_MULTIPLE`, which some implementations do not fully support.

---

## 7. CUDA Integer Encoding

The CUDA kernel receives `RecordGPU` structs with string fields pre-encoded as integers on the CPU:

```c
// CPU encodes once:
g->state = encode_state(fs(&r, C_STATE));   // e.g. "INT" → ST_INT (1)

// GPU kernel compares integers, not strings:
if (rec->state == ST_INT) s += 5;
```

**Why?**

GPUs have no standard C library — `strcmp` is not available in CUDA device code. More importantly, string comparison on GPU is far slower than integer comparison and would serialize warps due to branch divergence on variable-length strings.

The encoding adds a one-time CPU preprocessing step (~negligible) and allows the GPU kernel to run with no divergence on string comparisons.

---

## 8. CUDA Block Size = 256

The default block size of 256 threads/block was chosen because:

- It is a multiple of the warp size (32) — no wasted threads in the last warp
- It gives good occupancy on both older (sm_50) and newer (sm_80+) architectures
- At 700,001 records: grid = ⌈700001 / 256⌉ = 2735 blocks, totalling 699,136 active threads with 865 idle in the last block — 99.9% utilisation

We benchmarked block sizes 64, 128, 256, 512, 1024 on Tesla T4. Performance was essentially flat between 128–512 because the kernel is memory-bandwidth bound (each thread reads one 72-byte struct and writes two ints). We kept 256 as the default because it is the most commonly recommended starting point.

---

## 9. Confusion Matrix Validation

All six implementations must produce identical confusion matrices. This is enforced by `verify_results.py`. The guarantee holds because:

1. `detect()` is a pure function — same input always produces the same score
2. Records are processed independently — no inter-record state
3. All implementations read from the same file and parse with the same logic
4. The label column is read but never used during scoring — only during the post-processing validation step

If any implementation produces a different confusion matrix, it indicates either a parsing bug (reading the wrong column) or a data race (shared state being modified by multiple threads).
