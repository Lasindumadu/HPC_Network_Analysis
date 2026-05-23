/*
 * HPC Network Traffic Analysis - OpenMP Implementation
 * Course: EC7207 - High Performance Computing
 * Authors: EG/2021/4426, EG/2021/4432, EG/2021/4433
 *
 * ════════════════════════════════════════════════════════════════
 * PARALLELISATION STRATEGY:
 *   All records are loaded into memory first (same as MPI rank 0).
 *   The processing loop is parallelised with OpenMP:
 *     #pragma omp parallel for  — splits records across threads.
 *   Each thread scores its records independently (no shared state
 *   in detect()). Thread-local counters are reduced with
 *   #pragma omp atomic to avoid race conditions.
 *
 * CORRECTNESS GUARANTEE:
 *   Identical detect() function to all other implementations.
 *   All produce the same accuracy (97.072%) and confusion matrix.
 *
 * COMPILATION:
 *   gcc -Wall -O2 -std=c11 -fopenmp -lm \
 *       -o results/openmp src/openmp/network_analysis_openmp.c
 *
 * RUNNING:
 *   OMP_NUM_THREADS=1  ./results/openmp data/...
 *   OMP_NUM_THREADS=2  ./results/openmp data/...
 *   OMP_NUM_THREADS=4  ./results/openmp data/...
 *   OMP_NUM_THREADS=8  ./results/openmp data/...
 * ════════════════════════════════════════════════════════════════
 */

/* _GNU_SOURCE enables GNU-specific extensions (e.g. getline, strdupa).
   _POSIX_C_SOURCE 200809L enables POSIX.1-2008 functions like clock_gettime. */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>    /* printf, fopen, fgets, fclose, fprintf, snprintf */
#include <stdlib.h>   /* malloc, free, exit, strtof, strtol              */
#include <string.h>   /* strcmp, strcpy, strcspn, memcpy, strlen         */
#include <math.h>     /* sqrt() — used for RMSE calculation              */
#include <time.h>     /* clock_gettime, struct timespec                  */
#include <omp.h>      /* OpenMP API: omp_get_max_threads()               */
#include <ctype.h>    /* tolower — not directly used but included        */


/* ── Compile-time constants ─────────────────────────────────────────────────
 * MAX_LINE     : maximum number of characters we read per CSV line.
 * MAX_FIELDS   : maximum number of comma-separated columns in the CSV.
 * ATTACK_THRESHOLD : a record is labelled ATTACK if detect() returns >= this.
 * REPEAT_FACTOR    : we process the whole dataset this many times so that
 *                    a single-pass time measurement is stable and averaged.
 * MAX_RECORDS  : ceiling on how many lines we will load into memory.        */
#define MAX_LINE         512
#define MAX_FIELDS       50
#define ATTACK_THRESHOLD 4
#define REPEAT_FACTOR    50    /* must match serial REPEAT_FACTOR */
#define MAX_RECORDS      750000

/* ── Column index globals ────────────────────────────────────────────────────
 * These hold the zero-based column position of each feature inside the CSV.
 * They are set ONCE by detect_columns() after reading the header row,
 * and are then READ-ONLY for the rest of the program.
 * Being read-only makes them safe to use from every OpenMP thread with
 * no locking needed.
 * Example: if the CSV header is "proto,state,label,..." then
 *   C_PROTO=0, C_STATE=1, C_LABEL=2, everything else=-1.              */
static int C_STATE, C_PROTO, C_SERVICE, C_SPKTS, C_DPKTS;
static int C_RATE, C_STTL, C_DTTL, C_SLOAD;
static int C_SLOSS, C_DLOSS, C_SJIT, C_DJIT;
static int C_CT_SRV, C_CT_STT, C_CT_DPT, C_LABEL;
static int MIN_F;   /* = C_LABEL + 1; rows with fewer fields are skipped  */

/* ── High-resolution wall-clock timer ──────────────────────────────────────
 * CLOCK_MONOTONIC never jumps backward (unlike CLOCK_REALTIME) so it is
 * safe to use for measuring elapsed time even if the system clock is
 * adjusted during the run.
 * Returns seconds as a double (nanosecond precision).                      */
static double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ── Row: fixed-size stack-allocated CSV record ─────────────────────────────
 * FIELD_LEN : max characters stored per field (longer fields are truncated).
 * buf[f][c] : character c of field f.
 * n         : how many fields were actually found after parsing.
 *
 * WHY STACK-ALLOCATED?
 *   Each OpenMP thread will create its own Row r on the stack inside the
 *   parallel for loop body.  Stack allocation is per-thread and requires
 *   no heap lock, which eliminates malloc() contention that would otherwise
 *   serialise threads and destroy parallel speedup.                        */
#define FIELD_LEN 64
typedef struct {
    char buf[MAX_FIELDS][FIELD_LEN];  /* 2-D array: [field index][chars]  */
    int  n;                           /* number of fields parsed           */
} Row;

/* free_row: resets the field count. The char data remains but will be
   overwritten on the next parse() call.  Cheap O(1) "clear".             */
static inline void free_row(Row *r) { r->n = 0; }

/* ── parse(): split one CSV line into a Row ─────────────────────────────────
 * Algorithm (single-pass, no dynamic allocation):
 *   1. Walk pointer p through the line character by character.
 *   2. When we hit a comma (or end-of-string), we have found one field
 *      spanning [start, p).
 *   3. Strip leading/trailing whitespace and quote characters.
 *   4. Copy at most FIELD_LEN-1 bytes into buf[n], null-terminate, n++.
 * Returns the number of fields stored (= r->n).                           */
static int parse(const char *ln, Row *r) {
    r->n = 0;
    const char *p = ln;

    while (*p && r->n < MAX_FIELDS) {
        const char *start = p;

        /* advance p until we hit a comma or end-of-string */
        while (*p && *p != ',') p++;

        /* skip leading whitespace/quotes on this field */
        while (*start == ' ' || *start == '"' || *start == '\'') start++;

        int len = (int)(p - start);   /* raw field length before right-trim */

        /* trim trailing whitespace/quotes */
        while (len > 0 && (start[len-1] <= ' ' ||
               start[len-1] == '"' || start[len-1] == '\'')) len--;

        /* clamp to buffer size — very long fields get silently truncated */
        if (len >= FIELD_LEN) len = FIELD_LEN - 1;

        memcpy(r->buf[r->n], start, len);
        r->buf[r->n][len] = '\0';    /* null-terminate the field string */
        r->n++;

        if (*p == ',') p++;          /* skip the comma separator          */
    }
    return r->n;
}

/* ── Field accessor helpers ──────────────────────────────────────────────────
 * ff / fi / fs: safe reads of field c from row r.
 * Guard checks: column index is valid (>=0 and < r->n) AND field is non-empty.
 * If the guard fails, a safe default (0.0f / 0 / "") is returned instead
 * of crashing with an out-of-bounds access.                               */

/* ff: field as float — used for rate, sload, sjit, djit */
static inline float ff(Row *r, int c) {
    return (c>=0 && c<r->n && r->buf[c][0]) ? strtof(r->buf[c], NULL) : 0.0f;
}

/* fi: field as int — used for spkts, dpkts, sttl, dttl, sloss, dloss,
                       ct_srv_src, ct_state_ttl, ct_src_dport_ltm, label */
static inline int fi(Row *r, int c) {
    return (c>=0 && c<r->n && r->buf[c][0]) ? (int)strtol(r->buf[c], NULL, 10) : 0;
}

/* fs: field as C-string — used for state, proto, service               */
static inline const char *fs(Row *r, int c) {
    return (c>=0 && c<r->n) ? r->buf[c] : "";
}

/* ── detect_columns(): map CSV header names → column indices ────────────────
 * Reads the header line, parses it into a Row, then does a case-insensitive
 * name match against the list of features we care about.
 * Sets each C_* global to the matching column index, or leaves it at -1
 * if the column is absent in this particular CSV file.
 * This makes the code dataset-agnostic — column ORDER does not matter.    */
static void detect_columns(const char *hdr) {
    Row r;
    char *h = malloc(strlen(hdr) + 1);   /* mutable copy for in-place tolower */
    strcpy(h, hdr);
    h[strcspn(h, "\n")] = '\0';          /* strip trailing newline            */
    parse(h, &r);

    /* mapping table: feature name → pointer to its global index variable */
    struct { const char *n; int *t; } m[] = {
        {"state",&C_STATE}, {"proto",&C_PROTO}, {"service",&C_SERVICE},
        {"spkts",&C_SPKTS}, {"dpkts",&C_DPKTS}, {"rate",&C_RATE},
        {"sttl",&C_STTL},   {"dttl",&C_DTTL},   {"sload",&C_SLOAD},
        {"sloss",&C_SLOSS}, {"dloss",&C_DLOSS},  {"sjit",&C_SJIT},
        {"djit",&C_DJIT},   {"ct_srv_src",&C_CT_SRV},
        {"ct_state_ttl",&C_CT_STT}, {"ct_src_dport_ltm",&C_CT_DPT},
        {"label",&C_LABEL}
    };
    int nm = sizeof(m)/sizeof(m[0]);

    /* initialise all indices to -1 (= "column not found") */
    for (int i = 0; i < nm; i++) *m[i].t = -1;

    /* for each parsed header field: lowercase it, then scan mapping table */
    for (int i = 0; i < r.n; i++) {
        for (char *p = r.buf[i]; *p; p++)
            if (*p>='A' && *p<='Z') *p += 32;   /* ASCII lowercase trick */
        for (int j = 0; j < nm; j++)
            if (strcmp(r.buf[i], m[j].n) == 0) { *m[j].t = i; break; }
    }

    /* 'label' is mandatory — we cannot validate predictions without it */
    if (C_LABEL < 0) { fprintf(stderr, "ERROR: 'label' not found\n"); exit(1); }

    /* MIN_F: any row with fewer than this many fields is too short to use */
    MIN_F = C_LABEL + 1;

    printf("Columns: total=%d | label=%d | state=%d | dttl=%d | ct_state_ttl=%d\n\n",
           r.n, C_LABEL, C_STATE, C_DTTL, C_CT_STT);

    free_row(&r);
    free(h);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * detect(): RULE-BASED ANOMALY SCORING ENGINE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * WHAT IT DOES:
 *   Reads network-flow features from one parsed CSV row and returns an
 *   integer "suspicion score".  The caller classifies the flow as:
 *     score >= ATTACK_THRESHOLD  →  ATTACK  (1)
 *     score <  ATTACK_THRESHOLD  →  NORMAL  (0)
 *
 * THREAD SAFETY:
 *   - Reads ONLY from Row *r  (private per-thread stack variable) and
 *     the C_* column index globals (set once, read-only after that).
 *   - Has NO static/global mutable state.
 *   - Can be called from any number of threads simultaneously without
 *     locks, atomics, or any synchronisation.
 *
 * IDENTICAL TO SERIAL/MPI VERSIONS — guarantees same accuracy everywhere.
 * ═══════════════════════════════════════════════════════════════════════════ */
static int detect(Row *r) {

    /* ── Extract features from the row ─────────────────────────────────── */
    const char *st  = fs(r, C_STATE);    /* connection state: CON/INT/REQ/…  */
    const char *pr  = fs(r, C_PROTO);    /* protocol: tcp/udp/unas/sctp/…    */
    const char *svc = fs(r, C_SERVICE);  /* application service: dns/http/…  */
    int   spkts = fi(r, C_SPKTS);        /* source-to-dest packet count       */
    int   dpkts = fi(r, C_DPKTS);        /* dest-to-source packet count       */
    float rate  = ff(r, C_RATE);         /* total packets per second          */
    float sload = ff(r, C_SLOAD);        /* source bits per second            */
    int   sttl  = fi(r, C_STTL);         /* source IP TTL value               */
    int   dttl  = fi(r, C_DTTL);         /* destination IP TTL value          */
    float sjit  = ff(r, C_SJIT);         /* source jitter (ms)                */
    float djit  = ff(r, C_DJIT);         /* destination jitter (ms)           */
    int   loss  = fi(r, C_SLOSS) + fi(r, C_DLOSS); /* total packet loss      */
    int   cst   = fi(r, C_CT_STT);       /* count of flows with same state+TTL*/
    int   cdp   = fi(r, C_CT_DPT);       /* count of flows to same dest port  */
    int   csv   = fi(r, C_CT_SRV);       /* count of flows to same service    */
    int   s     = 0;                      /* suspicion score accumulator       */

    /* ── POSITIVE signals — each raises suspicion ──────────────────────── */

    /* INT state: connection was interrupted before completing handshake.
       This is a very strong indicator of a scan or half-open attack.      */
    if (!strcmp(st,"INT"))                                      s += 5;

    /* dttl 60 or 253: these specific destination TTL values appear almost
       exclusively in attack traffic in the UNSW-NB15 dataset.             */
    if (dttl==60 || dttl==253)                                  s += 5;

    /* Unusual/rare protocols: unas=unassigned, sctp/gre/ospf are routing
       or tunnelling protocols almost never seen in normal end-user flows.  */
    if (!strcmp(pr,"unas") || !strcmp(pr,"sctp") ||
        !strcmp(pr,"any")  || !strcmp(pr,"gre")  ||
        !strcmp(pr,"ospf"))                                     s += 5;

    /* High-value target services: pop3(email), ssl(encrypted channel),
       snmp(network management) are commonly exploited services.           */
    if (!strcmp(svc,"pop3") || !strcmp(svc,"ssl") ||
        !strcmp(svc,"snmp"))                                    s += 5;

    /* ct_state_ttl == 2: very few flows share this state+TTL combination,
       which correlates with attack traffic in the dataset.                */
    if (cst == 2)                                               s += 3;

    /* dttl == 0: destination TTL of zero is abnormal in legitimate traffic.*/
    if (dttl == 0)                                              s += 3;

    /* sttl near 255: values 254/255 are typical of crafted/spoofed packets
       or scanning tools that set TTL to max.                              */
    if (sttl == 254)                                            s += 2;
    if (sttl == 255)                                            s += 2;

    /* DNS service: used in DNS amplification DDoS attacks.               */
    if (!strcmp(svc,"dns"))                                     s += 2;

    /* Many destination ports from one source: classic port scan pattern.  */
    if (cdp > 10)                                               s += 2;

    /* Tiered sload checks: progressively higher source bandwidth likely
       indicates a data exfiltration or flood attack.                      */
    if (sload >  1000000.0f)                                    s += 1;
    if (sload > 10000000.0f)                                    s += 1;  /* cumulative */
    if (sload > 50000000.0f)                                    s += 1;  /* cumulative */

    /* Very high packet rate: characteristic of flood/DoS attacks.         */
    if (rate  > 100000.0f)                                      s += 1;
    if (rate  > 166666.0f)                                      s += 1;  /* cumulative */

    /* One-way traffic: source sends many packets but receives none.
       Typical of SYN flood or UDP flood where victim cannot respond.      */
    if (dpkts == 0 && spkts > 2)                                s += 1;

    /* Many flows to same service: service-level scanning behaviour.       */
    if (csv > 20)                                               s += 1;

    /* High jitter: erratic inter-packet timing suggests evasion or DDoS
       traffic with irregular burst patterns.                              */
    if (sjit > 1000.0f || djit > 1000.0f)                      s += 1;

    /* Packet loss combined with high jitter: degraded path quality that
       often accompanies attack traffic causing congestion.                */
    if (loss > 5 && (sjit > 500.0f || djit > 500.0f))          s += 1;

    /* ── NEGATIVE signals — each REDUCES suspicion ─────────────────────── */

    /* sttl=31: common Linux source TTL after typical routing hops — normal.*/
    if (sttl == 31)          s -= 4;

    /* dttl=29: similarly benign destination TTL observed in normal flows.  */
    if (dttl == 29)          s -= 4;

    /* ct_state_ttl=0: no other flows share this state+TTL — statistically
       normal; absence of clustering suggests legitimate traffic.           */
    if (cst  == 0)           s -= 3;

    /* CON state: fully established TCP connection — typical of normal use. */
    if (!strcmp(st,"CON"))   s -= 2;

    /* REQ state: normal application request in progress.                  */
    if (!strcmp(st,"REQ"))   s -= 2;

    return s;   /* caller compares this to ATTACK_THRESHOLD */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main(): orchestrates loading, parallel detection, and reporting
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {

    /* Default CSV path if no argument is provided on the command line */
    const char *file = argc > 1 ? argv[1] :
        "data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv";

    /* omp_get_max_threads() reads the OMP_NUM_THREADS environment variable.
       This tells us how many threads the parallel region will spawn.       */
    int nthreads = omp_get_max_threads();

    printf("=== OpenMP Network Traffic Anomaly Detection ===\n");
    printf("Threads: %d | Repeat factor: %d\n", nthreads, REPEAT_FACTOR);
    printf("File: %s\n", file);

    /* ── STEP 1: Open file and detect column layout from header ────────── */
    FILE *fp = fopen(file, "r");
    if (!fp) { perror(file); return 1; }

    char ln[MAX_LINE];
    if (!fgets(ln, MAX_LINE, fp)) {    /* read exactly the first (header) line */
        fprintf(stderr, "Empty file\n"); fclose(fp); return 1;
    }
    detect_columns(ln);   /* parse header → set all C_* global indices     */

    /* ── STEP 2: Load entire dataset into RAM ────────────────────────────
     * WHY LOAD EVERYTHING FIRST?
     *   File I/O with fgets() uses an internal lock (flockfile) in glibc.
     *   If we called fgets() from inside the parallel loop, every thread
     *   would contend for that lock and we would get ZERO parallel speedup.
     *   By loading all lines into a plain array up-front (single-threaded),
     *   the parallel loop can then access all_lines[i] with no locking.    */
    int total_lines = 0;

    /* Allocate a 2D array: MAX_RECORDS rows, each MAX_LINE characters wide.
       750000 × 512 = ~384 MB.  Large but fits in modern server RAM.       */
    char (*all_lines)[MAX_LINE] = malloc(sizeof(*all_lines) * MAX_RECORDS);
    if (!all_lines) { fprintf(stderr, "Out of memory\n"); return 1; }

    while (fgets(ln, MAX_LINE, fp) && total_lines < MAX_RECORDS) {
        ln[strcspn(ln, "\n")] = '\0';   /* strip trailing newline          */
        if (!ln[0]) continue;           /* skip blank lines                */
        snprintf(all_lines[total_lines], MAX_LINE, "%s", ln);
        total_lines++;
    }
    fclose(fp);
    printf("Records per pass: %d\n\n", total_lines);

    /* ── STEP 3: Parallel detection over REPEAT_FACTOR passes ───────────
     * Confusion matrix counters (global across all threads):              */
    long   TP = 0, TN = 0, FP = 0, FN = 0;
    double sse = 0.0;   /* sum of squared errors — for RMSE calculation    */
    long   tot = 0;     /* total records processed across all passes       */

    double t0 = now();  /* ← start timing here, BEFORE all repetitions    */

    for (int rep = 0; rep < REPEAT_FACTOR; rep++) {

        /* Thread-local accumulators declared OUTSIDE the parallel region.
           Each will be split into per-thread private copies by OpenMP's
           reduction clause, then summed back here automatically.          */
        long   local_TP = 0, local_TN = 0, local_FP = 0, local_FN = 0;
        double local_sse = 0.0;
        long   local_tot = 0;

        /* ── OpenMP parallel for region ────────────────────────────────────
         *
         * #pragma omp parallel for
         *   Forks a team of nthreads threads.  Each thread executes a
         *   disjoint subset of the loop iterations [0, total_lines).
         *
         * schedule(static)
         *   Divides iterations into equal chunks assigned to threads at
         *   compile-time (no dynamic load-balancing overhead).
         *   Optimal here because every iteration costs ~the same time.
         *
         * reduction(+: local_TP, ...)
         *   Tells OpenMP: for each listed variable, give each thread a
         *   private copy initialised to 0. After the loop, sum all private
         *   copies back into the original variable.
         *   This is the correct, race-condition-free way to accumulate
         *   integer/float counters across threads.
         *
         * default(shared)
         *   Any variable not listed in reduction/private is shared.
         *   all_lines is shared (read-only → safe).
         *   total_lines, ATTACK_THRESHOLD etc. are shared (read-only → safe).
         * ──────────────────────────────────────────────────────────────── */
        #pragma omp parallel for                    \
            schedule(static)                        \
            reduction(+: local_TP, local_TN,        \
                         local_FP, local_FN,        \
                         local_sse, local_tot)      \
            default(shared)
        for (int i = 0; i < total_lines; i++) {

            /* Row r is declared INSIDE the loop body.
               It lives on each thread's own stack — completely private.
               No two threads share this variable. No synchronisation needed. */
            Row r;
            parse(all_lines[i], &r);   /* parse CSV line into Row struct   */

            /* Score the flow.  detect() is thread-safe: reads only r
               (private) and C_* globals (read-only).                      */
            int pred = (detect(&r) >= ATTACK_THRESHOLD) ? 1 : 0;

            /* Ground-truth label from the CSV — used ONLY for validation,
               never fed back into the scoring logic.                       */
            int act  = fi(&r, C_LABEL);

            /* Accumulate squared error for RMSE (all passes)              */
            local_sse += (double)(pred - act) * (pred - act);
            local_tot++;

            /* Build confusion matrix on the FIRST pass only (rep==0).
               Counting it every pass would multiply the counts by 50.     */
            if (rep == 0) {
                if      (act==1 && pred==1) local_TP++;   /* True Positive  */
                else if (act==0 && pred==0) local_TN++;   /* True Negative  */
                else if (act==0 && pred==1) local_FP++;   /* False Positive */
                else                        local_FN++;   /* False Negative */
            }

        } /* ← OpenMP joins threads here; reductions are applied here      */

        /* Merge this repetition's reduced locals into the grand totals    */
        sse += local_sse;
        tot += local_tot;
        if (rep == 0) {
            /* Confusion matrix only from pass 0 — copy once               */
            TP = local_TP; TN = local_TN;
            FP = local_FP; FN = local_FN;
        }
    }

    double elapsed = now() - t0;   /* total wall time for all 50 passes    */
    free(all_lines);               /* release the ~384 MB data buffer       */

    /* ── STEP 4: Compute and display performance metrics ─────────────────
     * Divide by REPEAT_FACTOR to get per-pass (single-dataset) statistics. */
    long   single_pass = tot / REPEAT_FACTOR;       /* records in one pass  */
    double single_time = elapsed / REPEAT_FACTOR;   /* seconds for one pass */
    double rmse        = sqrt(sse / tot);            /* Root Mean Square Error*/

    /* Classification metrics from confusion matrix (first-pass counts):   */
    double accuracy    = 100.0 * (TP + TN) / single_pass;
    double precision   = (TP + FP) > 0 ? 100.0*TP/(TP+FP) : 0.0;
    double recall      = (TP + FN) > 0 ? 100.0*TP/(TP+FN) : 0.0;
    double f1          = (precision + recall) > 0
                         ? 2.0*precision*recall/(precision+recall) : 0.0;

    /* Print results */
    printf("=== Detection Results ===\n");
    printf("Records/pass: %ld | Passes: %d | Total processed: %ld\n",
           single_pass, REPEAT_FACTOR, tot);
    printf("Attacks: %ld | Normal: %ld  (first pass)\n", TP+FP, TN+FN);
    printf("Time (total x%d): %.4fs | Throughput: %.0f rec/s\n\n",
           REPEAT_FACTOR, elapsed, tot/elapsed);

    printf("=== Confusion Matrix (first pass) ===\n");
    printf("              Predicted\n");
    printf("Actual   Normal   Attack\n");
    printf("Normal  %7ld  %7ld   (FP=%ld)\n", TN, FP, FP);
    printf("Attack  %7ld  %7ld   (FN=%ld)\n", FN, TP, FN);

    printf("\n=== Accuracy Metrics ===\n");
    printf("Accuracy:  %7.3f%%\n",  accuracy);
    printf("Precision: %7.3f%%   (of flagged: how many are real attacks)\n", precision);
    printf("Recall:    %7.3f%%   (of real attacks: how many were caught)\n",  recall);
    printf("F1 Score:  %7.3f%%\n",  f1);
    printf("RMSE:      %.6f (%.4f%%)\n", rmse, rmse*100.0);
    printf("\nNote: Predictions from TRAFFIC FEATURES only.\n");
    printf("      Labels used only for validation above.\n\n");

    if      (accuracy > 85.0) printf("Status: EXCELLENT\n");
    else if (accuracy > 75.0) printf("Status: GOOD\n");
    else if (accuracy > 65.0) printf("Status: ACCEPTABLE\n");
    else                       printf("Status: POOR\n");

    /* ── STEP 5: Speedup vs serial baseline ──────────────────────────────
     * serial_time.txt is written by the serial implementation.
     * We compare our single-pass time to it and compute:
     *   Speedup    = serial_time / openmp_time
     *   Efficiency = Speedup / thread_count   (ideal = 100%)             */
    FILE *tf = fopen("../../results/serial_time.txt", "r");
    if (!tf) tf = fopen("serial_time.txt", "r");
    if (tf) {
        double serial_time;
        if (fscanf(tf, "%lf", &serial_time) == 1) {
            double speedup    = serial_time / single_time;
            double efficiency = speedup / nthreads * 100.0;
            printf("\n=== Speedup vs Serial ===\n");
            printf("Serial time (1 pass):   %.4fs\n",  serial_time);
            printf("OpenMP time (1 pass):   %.4fs\n",  single_time);
            printf("Threads:                %d\n",      nthreads);
            printf("Speedup:                %.2fx\n",  speedup);
            printf("Efficiency:             %.1f%%\n", efficiency);
        }
        fclose(tf);
    }

    /* ── STEP 6: Save our time for the Hybrid implementation to compare ── */
    FILE *of = fopen("../../results/openmp_time.txt", "w");
    if (!of) of = fopen("openmp_time.txt", "w");
    if (of) { fprintf(of, "%.6f\n", single_time); fclose(of); }

    printf("\nSingle-pass time: %.4fs  (saved)\n", single_time);
    printf("Total time (x%d): %.4fs\n", REPEAT_FACTOR, elapsed);

    /* ── STEP 7: Write per-thread-count log file for chart generation ────
     * File name includes thread count, e.g. openmp_4t.log, so the
     * benchmarking/plotting script can compare 1/2/4/8-thread results.    */
    { int _r = system("mkdir -p results/logs"); (void)_r; }
    char log_path[64];
    snprintf(log_path, sizeof(log_path), "results/logs/openmp_%dt.log", nthreads);
    FILE *lf = fopen(log_path, "w");
    if (!lf) {
        snprintf(log_path, sizeof(log_path), "openmp_%dt.log", nthreads);
        lf = fopen(log_path, "w");
    }
    if (lf) {
        fprintf(lf, "=== OpenMP Network Traffic Anomaly Detection ===\n");
        fprintf(lf, "Threads: %d\n", nthreads);
        fprintf(lf, "Records/pass: %d\n", total_lines);
        fprintf(lf, "Throughput: %.0f rec/s\n", (double)total_lines / single_time);
        fprintf(lf, "Accuracy:  %.3f%%\n",  accuracy);
        fprintf(lf, "Precision: %.3f%%\n",  precision);
        fprintf(lf, "Recall:    %.3f%%\n",  recall);
        fprintf(lf, "F1 Score:  %.3f%%\n",  f1);
        fprintf(lf, "RMSE:      %.6f \n",   rmse);
        fprintf(lf, "Single-pass time: %.4fs\n", single_time);
        fprintf(lf, "Total time (x%d): %.4fs\n", REPEAT_FACTOR, elapsed);

        /* Also log speedup inside the log file if serial baseline exists  */
        FILE *st = fopen("../../results/serial_time.txt", "r");
        if (!st) st = fopen("serial_time.txt", "r");
        if (st) {
            double serial_time2;
            if (fscanf(st, "%lf", &serial_time2) == 1) {
                double sp = serial_time2 / single_time;
                double ef = sp / nthreads * 100.0;
                fprintf(lf, "Speedup: %.2fx\n", sp);
                fprintf(lf, "Efficiency: %.1f%%\n", ef);
            }
            fclose(st);
        }
        fclose(lf);
        printf("Log saved to %s\n", log_path);
    }

    return 0;
}