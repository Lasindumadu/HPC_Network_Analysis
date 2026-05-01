/*
 * HPC Network Traffic Analysis - Hybrid MPI + OpenMP Implementation
 * Course: EC7207 - High Performance Computing
 * Authors: EG/2021/4426, EG/2021/4432, EG/2021/4433
 *
 * ════════════════════════════════════════════════════════════════
 * PARALLELISATION STRATEGY:
 *   Two-level parallelism:
 *     Level 1 — MPI (distributed memory):
 *       Rank 0 reads CSV and distributes rows evenly across all
 *       MPI processes (identical to MPI-only version).
 *     Level 2 — OpenMP (shared memory within each MPI rank):
 *       Each rank's processing loop is parallelised with OpenMP,
 *       using OMP_NUM_THREADS threads per rank.
 *
 *   Total parallelism = nprocs × OMP_NUM_THREADS
 *   Example: 4 MPI ranks × 4 threads = 16-way parallelism.
 *
 * CORRECTNESS GUARANTEE:
 *   Identical detect() function to serial/OpenMP/MPI/Pthreads.
 *   Confusion matrix must match baseline exactly:
 *     TP=32552  FP=8712  FN=12780  TN=28288
 *
 * COMPILATION:
 *   mpicc -Wall -O2 -std=c11 -fopenmp -lm \
 *         -o results/hybrid src/hybrid/network_analysis_hybrid.c
 *
 * RUNNING (e.g. 4 MPI ranks × 4 OpenMP threads = 16-way):
 *   OMP_NUM_THREADS=4 mpirun -np 4 ./results/hybrid \
 *       data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
 *
 *   OMP_NUM_THREADS=2 mpirun -np 2 ./results/hybrid data/...
 *   OMP_NUM_THREADS=8 mpirun -np 1 ./results/hybrid data/...
 * ════════════════════════════════════════════════════════════════
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define MAX_LINE         512
#define MAX_FIELDS       50
#define ATTACK_THRESHOLD 4
#define REPEAT_FACTOR    50
#define MAX_RECORDS      750000

/* ── Column indices ─────────────────────────── */
static int C_STATE, C_PROTO, C_SERVICE, C_SPKTS, C_DPKTS;
static int C_RATE, C_STTL, C_DTTL, C_SLOAD;
static int C_SLOSS, C_DLOSS, C_SJIT, C_DJIT;
static int C_CT_SRV, C_CT_STT, C_CT_DPT, C_LABEL;
static int MIN_F, TOTAL_COLS;

/* ── Timing ─────────────────────────────────── */
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ── Row parsing — zero-malloc stack-based ── */
#define FIELD_LEN 64
typedef struct {
    char buf[MAX_FIELDS][FIELD_LEN];
    int  n;
} Row;

static inline void free_row(Row *r) { r->n = 0; }

static int parse(const char *ln, Row *r) {
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

static inline float       ff(Row *r, int c) { return (c>=0&&c<r->n&&r->buf[c][0])?(float)atof(r->buf[c]):0.0f; }
static inline int         fi(Row *r, int c) { return (c>=0&&c<r->n&&r->buf[c][0])?     atoi(r->buf[c]):0;     }
static inline const char *fs(Row *r, int c) { return (c>=0&&c<r->n)?r->buf[c]:"";                              }

/* ── Column detection ── */
static void detect_columns(const char *hdr, int rank) {
    Row r; char *h = strdup(hdr);
    h[strcspn(h, "\n")] = '\0';
    parse(h, &r);
    TOTAL_COLS = r.n;

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
            if (*q>='A'&&*q<='Z') *q += 32;
        for (int j = 0; j < nm; j++)
            if (strcmp(r.buf[i], m[j].n)==0) { *m[j].t = i; break; }
    }

    if (C_LABEL < 0) {
        fprintf(stderr,"[rank %d] ERROR: 'label' not found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MIN_F = C_LABEL + 1;

    if (rank == 0)
        printf("Columns: total=%d | label=%d | state=%d | dttl=%d | ct_state_ttl=%d\n\n",
               r.n, C_LABEL, C_STATE, C_DTTL, C_CT_STT);

    free_row(&r); free(h);
}

/* ═══════════════════════════════════════════════
 * DETECTION ENGINE — identical to all other versions.
 * THREAD-SAFE: reads only Row *r and read-only globals.
 * Called from both MPI ranks and OpenMP threads safely.
 * ═══════════════════════════════════════════════ */
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

    if (sttl == 31)          s -= 4;
    if (dttl == 29)          s -= 4;
    if (cst  == 0)           s -= 3;
    if (!strcmp(st,"CON"))   s -= 2;
    if (!strcmp(st,"REQ"))   s -= 2;

    return s;
}

/* ════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    /* MPI_THREAD_FUNNELED: main thread makes all MPI calls;
       OpenMP threads only call detect() which has no MPI calls.   */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr,"WARNING: MPI does not support MPI_THREAD_FUNNELED "
                "(got %d). Continuing — no MPI calls inside threads.\n",
                provided);
    }

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nthreads = omp_get_max_threads();   /* read OMP_NUM_THREADS */

    const char *file = argc > 1 ? argv[1] :
        "data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv";

    if (rank == 0) {
        printf("=== Hybrid MPI+OpenMP Network Traffic Anomaly Detection ===\n");
        printf("MPI processes: %d | OpenMP threads/rank: %d | Total: %d\n",
               nprocs, nthreads, nprocs * nthreads);
        printf("Repeat factor: %d\n", REPEAT_FACTOR);
        printf("File: %s\n", file);
    }

    /* ── Step 1: Rank 0 reads all records ─────────────────────── */
    char header[MAX_LINE] = {0};
    int  total_lines = 0;
    char (*all_lines)[MAX_LINE] = NULL;

    if (rank == 0) {
        FILE *fp = fopen(file, "r");
        if (!fp) { perror(file); MPI_Abort(MPI_COMM_WORLD, 1); }

        if (!fgets(header, MAX_LINE, fp)) {
            fprintf(stderr,"Empty file\n");
            fclose(fp); MPI_Abort(MPI_COMM_WORLD, 1);
        }

        all_lines = (char (*)[MAX_LINE])malloc(sizeof(*all_lines) * MAX_RECORDS);
        if (!all_lines) {
            fprintf(stderr,"Out of memory\n");
            fclose(fp); MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char ln[MAX_LINE];
        while (fgets(ln, MAX_LINE, fp) && total_lines < MAX_RECORDS) {
            ln[strcspn(ln,"\n")] = '\0';
            if (!ln[0]) continue;
            memcpy(all_lines[total_lines], ln, MAX_LINE);
            total_lines++;
        }
        fclose(fp);
        printf("Records per pass: %d\n\n", total_lines);
    }

    /* ── Step 2: Broadcast header + record count ────────────── */
    MPI_Bcast(header,       MAX_LINE, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_lines, 1,        MPI_INT,  0, MPI_COMM_WORLD);

    detect_columns(header, rank);

    /* ── Step 3: Calculate this rank's slice ─────────────────── */
    int base     = total_lines / nprocs;
    int rem      = total_lines % nprocs;
    int my_count = base + (rank < rem ? 1 : 0);
    int my_start = rank * base + (rank < rem ? rank : rem);

    /* ── Step 4: Distribute lines to each rank ──────────────── */
    char *my_lines = (char *)malloc((size_t)my_count * MAX_LINE);
    if (!my_lines) {
        fprintf(stderr,"[rank %d] Out of memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        for (int dest = 1; dest < nprocs; dest++) {
            int d_count = base + (dest < rem ? 1 : 0);
            int d_start = dest * base + (dest < rem ? dest : rem);
            MPI_Send(all_lines[d_start], d_count * MAX_LINE,
                     MPI_CHAR, dest, 0, MPI_COMM_WORLD);
        }
        memcpy(my_lines, all_lines[my_start], (size_t)my_count * MAX_LINE);
        free(all_lines);
        all_lines = NULL;
    } else {
        MPI_Recv(my_lines, my_count * MAX_LINE,
                 MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* ── Step 5: Hybrid processing — MPI rank × OpenMP threads ──
     *
     * Each MPI rank runs an OpenMP parallel for over its slice.
     * detect() is thread-safe — no shared mutable state.
     * OpenMP reduction merges thread-local counters.
     * MPI_Reduce merges rank-local counters to rank 0.           */
    long   local_TP = 0, local_TN = 0, local_FP = 0, local_FN = 0;
    double local_sse = 0.0;
    long   local_tot = 0;

    MPI_Barrier(MPI_COMM_WORLD);   /* sync all ranks before timing */
    double t0 = now();

    for (int rep = 0; rep < REPEAT_FACTOR; rep++) {

        long   rep_TP = 0, rep_TN = 0, rep_FP = 0, rep_FN = 0;
        double rep_sse = 0.0;
        long   rep_tot = 0;

        /* OpenMP parallel for within this MPI rank's slice.
         * schedule(static): deterministic, equal-sized chunks.
         * reduction: each thread has private counters, merged here. */
        #pragma omp parallel for                        \
            schedule(static)                            \
            reduction(+: rep_TP, rep_TN, rep_FP, rep_FN, rep_sse, rep_tot) \
            default(shared)
        for (int i = 0; i < my_count; i++) {
            char tmp[MAX_LINE];
            memcpy(tmp, my_lines + (size_t)i * MAX_LINE, MAX_LINE);

            Row r;
            if (parse(tmp, &r) < MIN_F) { free_row(&r); continue; }

            int pred = (detect(&r) >= ATTACK_THRESHOLD) ? 1 : 0;
            int act  = fi(&r, C_LABEL);

            rep_sse += (double)(pred - act) * (pred - act);
            rep_tot++;

            if (rep == 0) {
                if      (act==1 && pred==1) rep_TP++;
                else if (act==0 && pred==0) rep_TN++;
                else if (act==0 && pred==1) rep_FP++;
                else                        rep_FN++;
            }

            free_row(&r);
        }

        local_sse += rep_sse;
        local_tot += rep_tot;
        if (rep == 0) {
            local_TP = rep_TP; local_TN = rep_TN;
            local_FP = rep_FP; local_FN = rep_FN;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);   /* sync before stopping timer */
    double elapsed = now() - t0;

    free(my_lines);

    /* ── Step 6: MPI reduce partial results to rank 0 ───────── */
    long   global_TP = 0, global_TN = 0, global_FP = 0, global_FN = 0;
    double global_sse = 0.0;
    long   global_tot = 0;
    double max_elapsed;

    MPI_Reduce(&local_TP,  &global_TP,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_TN,  &global_TN,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_FP,  &global_FP,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_FN,  &global_FN,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_tot, &global_tot, 1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed,   &max_elapsed,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ── Step 7: Rank 0 prints results ──────────────────────── */
    if (rank == 0) {
        long   single_pass = global_tot / REPEAT_FACTOR;
        double single_time = max_elapsed / REPEAT_FACTOR;
        double rmse        = sqrt(global_sse / global_tot);
        double accuracy    = 100.0 * (global_TP + global_TN) / (double)single_pass;
        double precision   = (global_TP+global_FP)>0
                             ? 100.0*global_TP/(global_TP+global_FP) : 0.0;
        double recall      = (global_TP+global_FN)>0
                             ? 100.0*global_TP/(global_TP+global_FN) : 0.0;
        double f1          = (precision+recall)>0
                             ? 2.0*precision*recall/(precision+recall) : 0.0;

        printf("=== Detection Results ===\n");
        printf("Records/pass: %ld | Passes: %d | Total processed: %ld\n",
               single_pass, REPEAT_FACTOR, global_tot);
        printf("Attacks: %ld | Normal: %ld  (first pass)\n",
               global_TP+global_FP, global_TN+global_FN);
        printf("Time (total x%d): %.4fs | Throughput: %.0f rec/s\n\n",
               REPEAT_FACTOR, max_elapsed, (double)global_tot/max_elapsed);

        printf("=== Confusion Matrix (first pass) ===\n");
        printf("              Predicted\n");
        printf("Actual   Normal   Attack\n");
        printf("Normal  %7ld  %7ld   (FP=%ld)\n", global_TN, global_FP, global_FP);
        printf("Attack  %7ld  %7ld   (FN=%ld)\n", global_FN, global_TP, global_FN);

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

        /* ── Speedup vs serial ───────────────── */
        FILE *tf = fopen("../../results/serial_time.txt", "r");
        if (!tf) tf = fopen("serial_time.txt", "r");
        if (tf) {
            double serial_time;
            if (fscanf(tf, "%lf", &serial_time) == 1) {
                double speedup    = serial_time / single_time;
                int    total_par  = nprocs * nthreads;
                double efficiency = speedup / total_par * 100.0;
                printf("\n=== Speedup vs Serial ===\n");
                printf("Serial time (1 pass):  %.4fs\n",  serial_time);
                printf("Hybrid time (1 pass):  %.4fs\n",  single_time);
                printf("MPI ranks:             %d\n",      nprocs);
                printf("Threads/rank:          %d\n",      nthreads);
                printf("Total parallelism:     %d\n",      total_par);
                printf("Speedup:               %.2fx\n",   speedup);
                printf("Efficiency:            %.1f%%\n",  efficiency);
            }
            fclose(tf);
        }

        /* save hybrid time */
        FILE *hf = fopen("../../results/hybrid_time.txt", "w");
        if (!hf) hf = fopen("hybrid_time.txt", "w");
        if (hf) { fprintf(hf, "%.6f\n", single_time); fclose(hf); }

        printf("\nSingle-pass time: %.4fs  (saved)\n", single_time);
        printf("Total time (x%d): %.4fs\n", REPEAT_FACTOR, max_elapsed);

        /* save log */
        { int _r = system("mkdir -p results/logs"); (void)_r; }
        char log_path[128];
        snprintf(log_path, sizeof(log_path),
                 "results/logs/hybrid_%dp_%dt.log", nprocs, nthreads);
        FILE *lf = fopen(log_path, "w");
        if (!lf) {
            snprintf(log_path, sizeof(log_path),
                     "hybrid_%dp_%dt.log", nprocs, nthreads);
            lf = fopen(log_path, "w");
        }
        if (lf) {
            fprintf(lf, "=== Hybrid MPI+OpenMP Network Traffic Anomaly Detection ===\n");
            fprintf(lf, "MPI processes: %d | Threads/rank: %d | Total: %d\n",
                    nprocs, nthreads, nprocs*nthreads);
            fprintf(lf, "Records/pass: %ld\n", single_pass);
            fprintf(lf, "Throughput: %.0f rec/s\n",
                    (double)global_tot/max_elapsed);
            fprintf(lf, "Accuracy:  %.3f%%\n",  accuracy);
            fprintf(lf, "Precision: %.3f%%\n",  precision);
            fprintf(lf, "Recall:    %.3f%%\n",  recall);
            fprintf(lf, "F1 Score:  %.3f%%\n",  f1);
            fprintf(lf, "RMSE:      %.6f \n",   rmse);
            fprintf(lf, "Single-pass time: %.4fs\n", single_time);
            fprintf(lf, "Total time (x%d): %.4fs\n", REPEAT_FACTOR, max_elapsed);
            FILE *st = fopen("../../results/serial_time.txt", "r");
            if (!st) st = fopen("serial_time.txt", "r");
            if (st) {
                double st2;
                if (fscanf(st, "%lf", &st2) == 1) {
                    int tp = nprocs * nthreads;
                    fprintf(lf, "Speedup: %.2fx\n",    st2/single_time);
                    fprintf(lf, "Efficiency: %.1f%%\n",(st2/single_time)/tp*100.0);
                }
                fclose(st);
            }
            fclose(lf);
            printf("Log saved to %s\n", log_path);
        }
    }

    MPI_Finalize();
    return 0;
}