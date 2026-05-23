/*
 * HPC Network Traffic Analysis - POSIX Threads Implementation
 * Course: EC7207 - High Performance Computing
 * Authors: EG/2021/4426, EG/2021/4432, EG/2021/4433
 * *
 * ════════════════════════════════════════════════════════════════
 * PARALLELISATION STRATEGY:
 *   All records loaded into memory first (same as OpenMP).
 *   Split into equal chunks — one chunk per thread.
 *   Each thread calls detect() independently (thread-safe: no shared
 *   mutable state inside detect()).
 *   Thread-local counters merged after pthread_join (no mutex needed
 *   in the hot loop — mutex only used once per thread at merge time).
 *
 * COMPILATION:
 *   gcc -Wall -O2 -std=c11 -pthread -lm \
 *       -o results/pthreads src/pthreads/network_analysis_pthread.c
 *
 * RUNNING:
 *   ./results/pthreads data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv 1
 *   ./results/pthreads data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv 2
 *   ./results/pthreads data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv 4
 *   ./results/pthreads data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv 8
 * ════════════════════════════════════════════════════════════════
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define MAX_LINE         512
#define MAX_FIELDS       50
#define ATTACK_THRESHOLD 4
#define REPEAT_FACTOR    50    /* must match serial REPEAT_FACTOR */
#define MAX_RECORDS      750000
#define MAX_THREADS      64

/* ── Column indices (set by detect_columns, read-only after that) ─ */
static int C_STATE, C_PROTO, C_SERVICE, C_SPKTS, C_DPKTS;
static int C_RATE, C_STTL, C_DTTL, C_SLOAD;
static int C_SLOSS, C_DLOSS, C_SJIT, C_DJIT;
static int C_CT_SRV, C_CT_STT, C_CT_DPT, C_LABEL;
static int MIN_F;

/* ── Timing ─────────────────────────────────── */
static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ── Row parsing — zero-malloc stack-based parser ───────────────
 * Identical to serial/OpenMP/MPI: no heap alloc, no lock contention. */
#define FIELD_LEN 64
typedef struct {
    char buf[MAX_FIELDS][FIELD_LEN];
    int  n;
} Row;

static inline void free_row(Row *r) { r->n = 0; }

static int parse_row(const char *ln, Row *r) {
    r->n = 0;
    const char *p = ln;
    while (*p && r->n < MAX_FIELDS) {
        const char *start = p;
        while (*p && *p != ',') p++;
        while (*start == ' ' || *start == '"' || *start == '\'') start++;
        int len = (int)(p - start);
        while (len > 0 && (start[len-1] <= ' ' ||
               start[len-1] == '"' || start[len-1] == '\'')) len--;
        if (len >= FIELD_LEN) len = FIELD_LEN - 1;
        memcpy(r->buf[r->n], start, len);
        r->buf[r->n][len] = '\0';
        r->n++;
        if (*p == ',') p++;
    }
    return r->n;
}

static inline float       ff(Row *r, int c) { return (c>=0&&c<r->n&&r->buf[c][0]) ? (float)atof(r->buf[c]) : 0.0f; }
static inline int         fi(Row *r, int c) { return (c>=0&&c<r->n&&r->buf[c][0]) ? atoi(r->buf[c])        : 0;    }
static inline const char *fs(Row *r, int c) { return (c>=0&&c<r->n)               ? r->buf[c]              : "";   }

/* ── Column detection ── identical to serial/OpenMP/MPI ─────── */
static void detect_columns(const char *hdr) {
    Row r;
    char *h = strdup(hdr);
    h[strcspn(h, "\n")] = '\0';
    parse_row(h, &r);

    struct { const char *n; int *t; } m[] = {
        {"state",&C_STATE}, {"proto",&C_PROTO}, {"service",&C_SERVICE},
        {"spkts",&C_SPKTS}, {"dpkts",&C_DPKTS}, {"rate",&C_RATE},
        {"sttl",&C_STTL},   {"dttl",&C_DTTL},   {"sload",&C_SLOAD},
        {"sloss",&C_SLOSS}, {"dloss",&C_DLOSS},  {"sjit",&C_SJIT},
        {"djit",&C_DJIT},   {"ct_srv_src",&C_CT_SRV},
        {"ct_state_ttl",&C_CT_STT}, {"ct_src_dport_ltm",&C_CT_DPT},
        {"label",&C_LABEL}
    };
    int nm = (int)(sizeof(m)/sizeof(m[0]));
    for (int i = 0; i < nm; i++) *m[i].t = -1;

    for (int i = 0; i < r.n; i++) {
        for (char *q = r.buf[i]; *q; q++)
            if (*q >= 'A' && *q <= 'Z') *q += 32;
        for (int j = 0; j < nm; j++)
            if (strcmp(r.buf[i], m[j].n) == 0) { *m[j].t = i; break; }
    }

    if (C_LABEL < 0) { fprintf(stderr, "ERROR: 'label' not found\n"); exit(1); }
    MIN_F = C_LABEL + 1;

    printf("Columns: total=%d | label=%d | state=%d | dttl=%d | ct_state_ttl=%d\n\n",
           r.n, C_LABEL, C_STATE, C_DTTL, C_CT_STT);
    free_row(&r);
    free(h);
}

/* ═══════════════════════════════════════════════════════════════
 * DETECTION ENGINE — IDENTICAL to serial, OpenMP, MPI versions.
 *
 * This function is THREAD-SAFE:
 *   - Reads only from Row *r (caller's stack) and
 *     global column index constants (set once, then read-only).
 *   - No shared mutable state.  No synchronisation needed here.
 *
 * ═══════════════════════════════════════════════════════════════ */
static int detect(Row *r) {
    const char *st  = fs(r, C_STATE);
    const char *pr  = fs(r, C_PROTO);
    const char *svc = fs(r, C_SERVICE);
    int   spkts = fi(r, C_SPKTS), dpkts = fi(r, C_DPKTS);
    float rate  = ff(r, C_RATE),  sload = ff(r, C_SLOAD);
    int   sttl  = fi(r, C_STTL),  dttl  = fi(r, C_DTTL);
    float sjit  = ff(r, C_SJIT),  djit  = ff(r, C_DJIT);
    int   loss  = fi(r, C_SLOSS) + fi(r, C_DLOSS);
    int   cst   = fi(r, C_CT_STT);
    int   cdp   = fi(r, C_CT_DPT);
    int   csv   = fi(r, C_CT_SRV);
    int   s     = 0;

    /* positive signals */
    if (!strcmp(st,"INT"))                                      s += 5;
    if (dttl==60 || dttl==253)                                  s += 5;
    if (!strcmp(pr,"unas") || !strcmp(pr,"sctp") ||
        !strcmp(pr,"any")  || !strcmp(pr,"gre")  ||
        !strcmp(pr,"ospf"))                                     s += 5;
    if (!strcmp(svc,"pop3") || !strcmp(svc,"ssl") ||
        !strcmp(svc,"snmp"))                                    s += 5;
    if (cst == 2)                                               s += 3;
    if (dttl == 0)                                              s += 3;
    if (sttl == 254)                                            s += 2;
    if (sttl == 255)                                            s += 2;
    if (!strcmp(svc,"dns"))                                     s += 2;
    if (cdp > 10)                                               s += 2;
    if (sload >  1000000.0f)                                    s += 1;
    if (sload > 10000000.0f)                                    s += 1;
    if (sload > 50000000.0f)                                    s += 1;
    if (rate  > 100000.0f)                                      s += 1;
    if (rate  > 166666.0f)                                      s += 1;
    if (dpkts == 0 && spkts > 2)                                s += 1;
    if (csv > 20)                                               s += 1;
    if (sjit > 1000.0f || djit > 1000.0f)                      s += 1;
    if (loss > 5 && (sjit > 500.0f || djit > 500.0f))          s += 1;

    /* negative signals */
    if (sttl == 31)          s -= 4;
    if (dttl == 29)          s -= 4;
    if (cst  == 0)           s -= 3;
    if (!strcmp(st,"CON"))   s -= 2;
    if (!strcmp(st,"REQ"))   s -= 2;

    return s;
}

/* ─── Thread argument and result structs ────────────────────── */
typedef struct {
    /* input */
    char   (*all_lines)[MAX_LINE]; /* pointer into shared read-only array  */
    int     start;                 /* first record index for this thread   */
    int     count;                 /* number of records to process         */
    int     repeat;                /* REPEAT_FACTOR                        */
    /* output — written once after all work done, no mutex needed here */
    long    TP, TN, FP, FN;
    double  sse;
    long    tot;
} ThreadArg;

/* ─── Worker function ─────────────────────────────────────────
 * Each thread processes records [start, start+count) for all
 * repetitions.  Uses only stack variables — no shared writes
 * during computation, so no mutex is needed in the hot loop. */
static void *worker(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;

    long   TP = 0, TN = 0, FP = 0, FN = 0;
    double sse = 0.0;
    long   tot = 0;

    for (int rep = 0; rep < a->repeat; rep++) {
        for (int i = a->start; i < a->start + a->count; i++) {
            char tmp[MAX_LINE];
            /* safe copy — all_lines is shared but read-only */
            memcpy(tmp, a->all_lines[i], MAX_LINE);

            Row r;
            if (parse_row(tmp, &r) < MIN_F) { free_row(&r); continue; }

            int pred = (detect(&r) >= ATTACK_THRESHOLD) ? 1 : 0;
            int act  = fi(&r, C_LABEL);   /* ground truth — validation only */

            sse += (double)(pred - act) * (pred - act);
            tot++;

            /* confusion matrix on first pass only */
            if (rep == 0) {
                if      (act==1 && pred==1) TP++;
                else if (act==0 && pred==0) TN++;
                else if (act==0 && pred==1) FP++;
                else                        FN++;
            }

            free_row(&r);
        }
    }

    /* write results back to struct — done after all work, thread-safe */
    a->TP  = TP;  a->TN  = TN;
    a->FP  = FP;  a->FN  = FN;
    a->sse = sse; a->tot = tot;

    return NULL;
}

/* ─── MAIN ───────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    const char *file     = argc > 1 ? argv[1] :
        "data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv";
    int nthreads = argc > 2 ? atoi(argv[2]) : 4;
    if (nthreads < 1)           nthreads = 1;
    if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;

    printf("=== Pthreads Network Traffic Anomaly Detection ===\n");
    printf("Threads: %d | Repeat factor: %d\n", nthreads, REPEAT_FACTOR);
    printf("File: %s\n", file);

    /* ── Step 1: Read header and detect columns ─── */
    FILE *fp = fopen(file, "r");
    if (!fp) { perror(file); return 1; }

    char ln[MAX_LINE];
    if (!fgets(ln, MAX_LINE, fp)) {
        fprintf(stderr, "Empty file\n"); fclose(fp); return 1;
    }
    detect_columns(ln);

    /* ── Step 2: Load all records into memory ─────
     * Identical strategy to OpenMP — read once, process N times. */
    char (*all_lines)[MAX_LINE] = malloc(sizeof(*all_lines) * MAX_RECORDS);
    if (!all_lines) { fprintf(stderr, "Out of memory\n"); fclose(fp); return 1; }

    int total_lines = 0;
    while (fgets(ln, MAX_LINE, fp) && total_lines < MAX_RECORDS) {
        ln[strcspn(ln, "\n")] = '\0';
        if (!ln[0]) continue;
        memcpy(all_lines[total_lines], ln, MAX_LINE);
        total_lines++;
    }
    fclose(fp);
    printf("Records per pass: %d\n\n", total_lines);

    /* ── Step 3: Divide records across threads ─── */
    pthread_t  threads[MAX_THREADS];
    ThreadArg  args[MAX_THREADS];

    int base = total_lines / nthreads;
    int rem  = total_lines % nthreads;

    double t0 = now_sec();

    for (int t = 0; t < nthreads; t++) {
        args[t].all_lines = all_lines;
        /* give one extra record to the first `rem` threads */
        args[t].count     = base + (t < rem ? 1 : 0);
        args[t].start     = t * base + (t < rem ? t : rem);
        args[t].repeat    = REPEAT_FACTOR;
        pthread_create(&threads[t], NULL, worker, &args[t]);
    }

    /* ── Step 4: Wait for all threads then merge results ── */
    long   TP = 0, TN = 0, FP = 0, FN = 0;
    double sse = 0.0;
    long   tot = 0;

    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
        TP  += args[t].TP;   TN  += args[t].TN;
        FP  += args[t].FP;   FN  += args[t].FN;
        sse += args[t].sse;  tot += args[t].tot;
    }

    double elapsed = now_sec() - t0;
    free(all_lines);

    /* ── Step 5: Compute and print metrics ─────── */
    long   single_pass = tot / REPEAT_FACTOR;
    double single_time = elapsed / REPEAT_FACTOR;
    double rmse        = sqrt(sse / tot);
    double accuracy    = 100.0 * (TP + TN) / (double)single_pass;
    double precision   = (TP + FP) > 0 ? 100.0*TP/(TP+FP) : 0.0;
    double recall      = (TP + FN) > 0 ? 100.0*TP/(TP+FN) : 0.0;
    double f1          = (precision + recall) > 0
                         ? 2.0*precision*recall/(precision+recall) : 0.0;

    printf("=== Detection Results ===\n");
    printf("Records/pass: %ld | Passes: %d | Total processed: %ld\n",
           single_pass, REPEAT_FACTOR, tot);
    printf("Attacks: %ld | Normal: %ld  (first pass)\n", TP+FP, TN+FN);
    printf("Time (total x%d): %.4fs | Throughput: %.0f rec/s\n\n",
           REPEAT_FACTOR, elapsed, (double)tot/elapsed);

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

    /* ── Speedup vs serial ─────────────────────── */
    FILE *tf = fopen("../../results/serial_time.txt", "r");
    if (!tf) tf = fopen("serial_time.txt", "r");
    if (tf) {
        double serial_time;
        if (fscanf(tf, "%lf", &serial_time) == 1) {
            double speedup    = serial_time / single_time;
            double efficiency = speedup / nthreads * 100.0;
            printf("\n=== Speedup vs Serial ===\n");
            printf("Serial time (1 pass):    %.4fs\n", serial_time);
            printf("Pthreads time (1 pass):  %.4fs\n", single_time);
            printf("Threads:                 %d\n",     nthreads);
            printf("Speedup:                 %.2fx\n",  speedup);
            printf("Efficiency:              %.1f%%\n", efficiency);
        }
        fclose(tf);
    }

    /* save pthreads time for hybrid comparison */
    FILE *pf = fopen("../../results/pthreads_time.txt", "w");
    if (!pf) pf = fopen("pthreads_time.txt", "w");
    if (pf) { fprintf(pf, "%.6f\n", single_time); fclose(pf); }

    printf("\nSingle-pass time: %.4fs  (saved)\n", single_time);
    printf("Total time (x%d): %.4fs\n", REPEAT_FACTOR, elapsed);

    /* save log for chart generation */
    { int _r = system("mkdir -p results/logs"); (void)_r; }
    char log_path[128];
    snprintf(log_path, sizeof(log_path),
             "results/logs/pthreads_%dt.log", nthreads);
    FILE *lf = fopen(log_path, "w");
    if (!lf) {
        snprintf(log_path, sizeof(log_path), "pthreads_%dt.log", nthreads);
        lf = fopen(log_path, "w");
    }
    if (lf) {
        fprintf(lf, "=== Pthreads Network Traffic Anomaly Detection ===\n");
        fprintf(lf, "Threads: %d\n", nthreads);
        fprintf(lf, "Records/pass: %ld\n", single_pass);
        fprintf(lf, "Throughput: %.0f rec/s\n", (double)single_pass / single_time);
        fprintf(lf, "Accuracy:  %.3f%%\n",  accuracy);
        fprintf(lf, "Precision: %.3f%%\n",  precision);
        fprintf(lf, "Recall:    %.3f%%\n",  recall);
        fprintf(lf, "F1 Score:  %.3f%%\n",  f1);
        fprintf(lf, "RMSE:      %.6f \n",   rmse);
        fprintf(lf, "Single-pass time: %.4fs\n", single_time);
        fprintf(lf, "Total time (x%d): %.4fs\n", REPEAT_FACTOR, elapsed);
        FILE *st = fopen("../../results/serial_time.txt", "r");
        if (!st) st = fopen("serial_time.txt", "r");
        if (st) {
            double st2;
            if (fscanf(st, "%lf", &st2) == 1) {
                fprintf(lf, "Speedup: %.2fx\n",    st2 / single_time);
                fprintf(lf, "Efficiency: %.1f%%\n", (st2/single_time)/nthreads*100.0);
            }
            fclose(st);
        }
        fclose(lf);
        printf("Log saved to %s\n", log_path);
    }

    return 0;
}