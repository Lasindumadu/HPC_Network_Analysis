/*
 * config.h — Shared compile-time constants for all HPC implementations
 * Course: EC7207 - High Performance Computing
 * Authors: EG/2021/4426, EG/2021/4432, EG/2021/4433
 *
 * Include this header in all C/CUDA source files instead of
 * duplicating these values.  Changing a value here propagates
 * to every implementation on the next `make all`.
 *
 * Usage:
 *   #include "../../config.h"   (from src/<impl>/ directories)
 */

#ifndef HPC_CONFIG_H
#define HPC_CONFIG_H

/* ── Detection ────────────────────────────────────────────────
 *
 * ATTACK_THRESHOLD: a record's score must reach this value to be
 * classified as an attack.  Set to 4 based on feature analysis
 * (see docs/design_decisions.md for derivation).
 * Raising this value reduces false positives but increases false
 * negatives.  Lowering it does the opposite.
 */
#define ATTACK_THRESHOLD 4

/* ── Benchmarking ─────────────────────────────────────────────
 *
 * REPEAT_FACTOR: how many times the dataset is processed per run.
 * The reported "single-pass time" is (total_elapsed / REPEAT_FACTOR).
 *
 * Why 50?  On our test machine (VMware Ubuntu, serial ~0.43 s/pass),
 * 50 passes gives ~21 s of total wall time — long enough to smooth
 * out OS scheduler noise while keeping the full benchmark under
 * 2 minutes for all implementations.
 *
 * For quick testing you can lower this to 1 or 5 without affecting
 * correctness.  The speedup and efficiency formulas remain valid.
 */
#define REPEAT_FACTOR 50

/* ── Memory / Buffer Sizes ────────────────────────────────────
 *
 * MAX_RECORDS: pre-allocated row buffer for in-memory loading
 * (OpenMP, Pthreads, MPI rank 0, Hybrid rank 0, CUDA).
 * The UNSW-NB15 file 1 has 700,001 records; 750,000 gives headroom.
 *
 * MAX_LINE: maximum byte length of one CSV row (including newline).
 * The UNSW-NB15 rows are typically 200-350 bytes; 512 is safe.
 * The serial implementation uses 4096 for its raw fgets buffer
 * because it reads directly without in-memory storage.
 *
 * MAX_FIELDS: maximum number of comma-separated columns.
 * UNSW-NB15 has 49 columns; 50 gives one slot of margin.
 *
 * FIELD_LEN: maximum characters stored per field value.
 * String fields (state, proto, service) are at most 10 chars;
 * 64 is generous and keeps Row structs cache-friendly.
 */
#define MAX_RECORDS 750000
#define MAX_LINE    512
#define MAX_FIELDS  50
#define FIELD_LEN   64

/* ── Parallelism Limits ───────────────────────────────────────
 *
 * MAX_THREADS: upper bound for the Pthreads thread array.
 * Does not limit OpenMP or MPI — those use runtime values.
 */
#define MAX_THREADS 64

/* ── Scoring Weights (informational — used in detect()) ───────
 * These are not #defines because they are used inline in the
 * scoring function.  They are documented here for reference.
 *
 * Positive signals (attack indicators):
 *   state=INT              +5   (86.9% attack rate in dataset)
 *   dttl=60 or 253         +5   (≥99.6% attack rate)
 *   proto=unas/sctp/any/   +5   (≥94% attack rate)
 *          gre/ospf
 *   service=pop3/ssl/snmp  +5   (100% attack rate)
 *   ct_state_ttl=2         +3   (87.0% attack rate)
 *   dttl=0                 +3   (82.7% attack rate)
 *   sttl=254               +2   (100% attack rate)
 *   sttl=255               +2   (72% attack rate)
 *   service=dns            +2   (85.6% attack rate)
 *   ct_src_dport_ltm > 10  +2   (port scan indicator)
 *   sload > 1M             +1   (high source load, tier 1)
 *   sload > 10M            +1   (high source load, tier 2)
 *   sload > 50M            +1   (high source load, tier 3)
 *   rate > 100k            +1   (high packet rate, tier 1)
 *   rate > 166.6k          +1   (high packet rate, tier 2)
 *   dpkts=0 && spkts>2     +1   (one-way / no-response traffic)
 *   ct_srv_src > 20        +1   (many connections to same service)
 *   sjit>1000 || djit>1000 +1   (high jitter)
 *   loss>5 && jit>500      +1   (loss + jitter = backdoor signal)
 *
 * Negative signals (normal traffic indicators):
 *   sttl=31                -4   (0.0% attack rate, 16,702 records)
 *   dttl=29                -4   (0.0% attack rate, 16,668 records)
 *   ct_state_ttl=0         -3   (0.9% attack rate, 16,897 records)
 *   state=CON              -2   (5.0% attack rate)
 *   state=REQ              -2   (7.3% attack rate)
 *
 * See docs/design_decisions.md and src/serial/analyze_features.py
 * for the full statistical derivation.
 */

#endif /* HPC_CONFIG_H */
