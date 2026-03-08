/*
 * HPC Network Traffic Analysis - MPI Implementation
 * Course: EC7207 - High Performance Computing
 * Authors: EG/2021/4426, EG/2021/4432, EG/2021/4433
 *
 * ════════════════════════════════════════════════════════════════
 * PARALLELISATION STRATEGY:
 *   Rank 0 reads the CSV and distributes rows evenly across all
 *   MPI processes. Each process independently scores its chunk.
 *   MPI_Reduce gathers TP/TN/FP/FN/SSE totals to rank 0.
 *   Rank 0 prints final metrics — identical to serial output.
 *
 * CORRECTNESS GUARANTEE:
 *   Since each record is scored independently (no shared state),
 *   the confusion matrix must match the serial baseline exactly:
 *     TP=32552  FP=8712  FN=12780  TN=28288
 *
 * COMPILATION:
 *   mpicc -Wall -O2 -std=c11 -lm -o results/mpi src/mpi/network_analysis_mpi.c
 *
 * RUNNING:
 *   mpirun -np 1 ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
 *   mpirun -np 2 ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
 *   mpirun -np 4 ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
 *   mpirun -np 8 ./results/mpi data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
 * ════════════════════════════════════════════════════════════════
 */

/* _POSIX_C_SOURCE MUST be first — before ANY #include —
   so that strdup, clock_gettime, CLOCK_MONOTONIC are visible
   even after mpi.h is included.                              */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define MAX_LINE         512
#define MAX_FIELDS       50
#define ATTACK_THRESHOLD 4
#define REPEAT_FACTOR    50    /* must match serial REPEAT_FACTOR */
#define MAX_RECORDS      750000

/* ── Column indices ─────────────────────────── */
static int C_STATE, C_PROTO, C_SERVICE, C_SPKTS, C_DPKTS;
static int C_RATE, C_STTL, C_DTTL, C_SLOAD;
static int C_SLOSS, C_DLOSS, C_SJIT, C_DJIT;
static int C_CT_SRV, C_CT_STT, C_CT_DPT, C_LABEL;
static int MIN_F, TOTAL_COLS;

/* ── Timing ─────────────────────────────────── */
static double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ── Row parsing — zero-malloc stack-based parser ───
 * Identical to serial and OpenMP implementations.
 * No heap allocation — no malloc lock contention.   */
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

static inline float       ff(Row *r, int c) { return (c>=0&&c<r->n&&r->buf[c][0]) ? atof(r->buf[c]) : 0.0f; }
static inline int         fi(Row *r, int c) { return (c>=0&&c<r->n&&r->buf[c][0]) ? atoi(r->buf[c]) : 0;    }
static inline const char *fs(Row *r, int c) { return (c>=0&&c<r->n)               ? r->buf[c]       : "";   }

/* ── Column detection (all ranks call this on the same header) ── */
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
    int nm = sizeof(m)/sizeof(m[0]);
    for (int i = 0; i < nm; i++) *m[i].t = -1;

    for (int i = 0; i < r.n; i++) {
        for (char *p = r.buf[i]; *p; p++) if (*p>='A'&&*p<='Z') *p += 32;
        for (int j = 0; j < nm; j++)
            if (strcmp(r.buf[i], m[j].n) == 0) { *m[j].t = i; break; }
    }

    if (C_LABEL < 0) {
        fprintf(stderr, "[rank %d] ERROR: 'label' column not found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);   /* terminate all ranks immediately — label column is required */
    }
    MIN_F = C_LABEL + 1;

    if (rank == 0)
        printf("Columns: total=%d | label=%d | state=%d | dttl=%d | ct_state_ttl=%d\n\n",
               r.n, C_LABEL, C_STATE, C_DTTL, C_CT_STT);

    free_row(&r); free(h);
}

/* ═══════════════════════════════════════════════
 * DETECTION ENGINE — identical to serial version.
 * No labels used. Score >= ATTACK_THRESHOLD = attack.
 *
 * Weights from analyze_features.py distributions:
 *  POSITIVE signals:
 *   state=INT       +5  (86.9% attack)
 *   dttl=60/253     +5  (≥99.6% attack)
 *   proto=unas..    +5  (≥94% attack)
 *   service=pop3..  +5  (100% attack)
 *   ct_state_ttl=2  +3  (87.0% attack)
 *   dttl=0          +3  (82.7% attack)
 *   sttl=254/255    +2  (72-100% attack)
 *   service=dns     +2  (85.6% attack)
 *   ct_dport>10     +2  (scan signal)
 *   sload tiers     +1 each (3 levels)
 *   rate tiers      +1 each (2 levels)
 *   dpkts=0,spkts>2 +1  (one-way traffic)
 *   ct_srv>20       +1
 *   jitter>1000     +1
 *   loss+jitter     +1  (backdoor signal)
 *  NEGATIVE signals:
 *   sttl=31         -4  (0.0% attack)
 *   dttl=29         -4  (0.0% attack)
 *   ct_state_ttl=0  -3  (0.9% attack)
 *   state=CON       -2  (5.0% attack)
 *   state=REQ       -2  (7.3% attack)
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

/* ═══════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);                /* initialise MPI environment — must be called before any other MPI function */

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  /* get this process's unique rank (0 to nprocs-1) */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* get total number of processes in the communicator */

    const char *file = argc > 1 ? argv[1] :
        "data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv";

    if (rank == 0) {
        printf("=== MPI Network Traffic Anomaly Detection ===\n");
        printf("Processes: %d | Repeat factor: %d\n", nprocs, REPEAT_FACTOR);
        printf("File: %s\n", file);
    }

    /* ── Step 1: Rank 0 reads all records into memory ────────── */
    char header[MAX_LINE] = {0};
    int  total_lines = 0;
    char (*all_lines)[MAX_LINE] = NULL;

    if (rank == 0) {
        FILE *fp = fopen(file, "r");
        if (!fp) { perror(file); MPI_Abort(MPI_COMM_WORLD, 1); }  /* abort all ranks if rank 0 cannot open the dataset file */

        if (!fgets(header, MAX_LINE, fp)) {
            fprintf(stderr, "Empty file\n");
            fclose(fp); MPI_Abort(MPI_COMM_WORLD, 1);  /* abort all ranks — cannot proceed with empty file */
        }

        all_lines = malloc(sizeof(*all_lines) * MAX_RECORDS);
        if (!all_lines) {
            fprintf(stderr, "Out of memory\n");
            fclose(fp); MPI_Abort(MPI_COMM_WORLD, 1);  /* abort all ranks — not enough memory to load dataset */
        }

        char ln[MAX_LINE];
        while (fgets(ln, MAX_LINE, fp) && total_lines < MAX_RECORDS) {
            ln[strcspn(ln, "\n")] = '\0';
            if (!ln[0]) continue;
            snprintf(all_lines[total_lines], MAX_LINE, "%s", ln);
            total_lines++;
        }
        fclose(fp);
        printf("Records per pass: %d\n\n", total_lines);
    }

    /* ── Step 2: Broadcast header + record count to all ranks ── */
    MPI_Bcast(header,       MAX_LINE, MPI_CHAR, 0, MPI_COMM_WORLD);  /* broadcast CSV header from rank 0 to all ranks so every rank can detect column indices */
    MPI_Bcast(&total_lines, 1,        MPI_INT,  0, MPI_COMM_WORLD);  /* broadcast total record count so all ranks can compute their slice boundaries */

    /* All ranks detect column indices from the same header */
    detect_columns(header, rank);

    /* ── Step 3: Calculate this rank's slice boundaries ─────── */
    int base     = total_lines / nprocs;
    int rem      = total_lines % nprocs;
    int my_count = base + (rank < rem ? 1 : 0);
    int my_start = rank * base + (rank < rem ? rank : rem);

    /* ── Step 4: Send each rank its slice of lines ───────────── */
    char *my_lines = malloc((size_t)my_count * MAX_LINE);
    if (!my_lines) {
        fprintf(stderr, "[rank %d] Out of memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);   /* abort all ranks — worker could not allocate its receive buffer */
    }

    if (rank == 0) {
        /* Send slices to all other ranks */
        for (int dest = 1; dest < nprocs; dest++) {
            int d_count = base + (dest < rem ? 1 : 0);
            int d_start = dest * base + (dest < rem ? dest : rem);
            MPI_Send(all_lines[d_start], d_count * MAX_LINE,
                     MPI_CHAR, dest, 0, MPI_COMM_WORLD);  /* send this rank's slice of raw CSV lines to destination rank */
        }
        /* Rank 0 keeps its own slice */
        memcpy(my_lines, all_lines[my_start], (size_t)my_count * MAX_LINE);
        free(all_lines);
        all_lines = NULL;
    } else {
        MPI_Recv(my_lines, my_count * MAX_LINE,
                 MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  /* worker rank receives its assigned slice of CSV lines from rank 0 */
    }

    /* ── Step 5: Each rank processes its chunk REPEAT_FACTOR times ─ */
    long   local_TP  = 0, local_TN  = 0, local_FP  = 0, local_FN  = 0;
    double local_sse = 0.0;
    long   local_tot = 0;

    MPI_Barrier(MPI_COMM_WORLD);   /* sync all ranks before timing — ensures no rank starts early */
    double t0 = now();

    for (int rep = 0; rep < REPEAT_FACTOR; rep++) {
        for (int i = 0; i < my_count; i++) {
            char ln[MAX_LINE];
            snprintf(ln, MAX_LINE, "%s", my_lines + (size_t)i * MAX_LINE);
            if (!ln[0]) continue;

            Row r;
            if (parse(ln, &r) < MIN_F) { free_row(&r); continue; }

            int pred = (detect(&r) >= ATTACK_THRESHOLD) ? 1 : 0;
            int act  = fi(&r, C_LABEL);   /* ground truth — validation only */

            local_sse += (double)(pred - act) * (pred - act);
            local_tot++;

            /* confusion matrix counted on first pass only */
            if (rep == 0) {
                if      (act==1 && pred==1) local_TP++;
                else if (act==0 && pred==0) local_TN++;
                else if (act==0 && pred==1) local_FP++;
                else                        local_FN++;
            }

            free_row(&r);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);   /* sync all ranks after processing — elapsed time measured from the slowest rank */
    double elapsed = now() - t0;

    free(my_lines);

    /* ── Step 6: Reduce all partial results to rank 0 ───────── */
    long   global_TP  = 0, global_TN  = 0, global_FP  = 0, global_FN  = 0;
    double global_sse = 0.0;
    long   global_tot = 0;
    double max_elapsed;

    MPI_Reduce(&local_TP,  &global_TP,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);  /* sum all ranks' true positive counts into rank 0 */
    MPI_Reduce(&local_TN,  &global_TN,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);  /* sum all ranks' true negative counts into rank 0 */
    MPI_Reduce(&local_FP,  &global_FP,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);  /* sum all ranks' false positive counts into rank 0 */
    MPI_Reduce(&local_FN,  &global_FN,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);  /* sum all ranks' false negative counts into rank 0 */
    MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);  /* sum all ranks' squared errors for global RMSE calculation */
    MPI_Reduce(&local_tot, &global_tot, 1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);  /* sum all ranks' processed record counts for throughput */
    MPI_Reduce(&elapsed,   &max_elapsed,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);  /* take the slowest rank's time as the true wall-clock elapsed time */

    /* ── Step 7: Rank 0 computes and prints all results ─────── */
    if (rank == 0) {
        long   single_pass = global_tot / REPEAT_FACTOR;
        double single_time = max_elapsed / REPEAT_FACTOR;
        double rmse        = sqrt(global_sse / global_tot);
        double accuracy    = 100.0 * (global_TP + global_TN) / single_pass;
        double precision   = (global_TP + global_FP) > 0
                             ? 100.0 * global_TP / (global_TP + global_FP) : 0.0;
        double recall      = (global_TP + global_FN) > 0
                             ? 100.0 * global_TP / (global_TP + global_FN) : 0.0;
        double f1          = (precision + recall) > 0
                             ? 2.0 * precision * recall / (precision + recall) : 0.0;

        printf("=== Detection Results ===\n");
        printf("Records/pass: %ld | Passes: %d | Total processed: %ld\n",
               single_pass, REPEAT_FACTOR, global_tot);
        printf("Attacks: %ld | Normal: %ld  (first pass)\n",
               global_TP + global_FP, global_TN + global_FN);
        printf("Time (total x%d): %.4fs | Throughput: %.0f rec/s\n\n",
               REPEAT_FACTOR, max_elapsed, global_tot / max_elapsed);

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
        printf("RMSE:      %.6f (%.4f%%)\n", rmse, rmse * 100.0);
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
                double efficiency = speedup / nprocs * 100.0;
                printf("\n=== Speedup vs Serial ===\n");
                printf("Serial time (1 pass):  %.4fs\n",  serial_time);
                printf("MPI    time (1 pass):  %.4fs\n",  single_time);
                printf("Processes:             %d\n",      nprocs);
                printf("Speedup:               %.2fx\n",  speedup);
                printf("Efficiency:            %.1f%%\n", efficiency);
            }
            fclose(tf);
        }

        /* save MPI time for hybrid comparison */
        FILE *mf = fopen("../../results/mpi_time.txt", "w");
        if (!mf) mf = fopen("mpi_time.txt", "w");
        if (mf) { fprintf(mf, "%.6f\n", single_time); fclose(mf); }

        printf("\nSingle-pass time: %.4fs  (saved)\n", single_time);
        printf("Total time (x%d): %.4fs\n", REPEAT_FACTOR, max_elapsed);

        /* save log for chart generation (rank 0 only) */
        { int _r = system("mkdir -p results/logs"); (void)_r; }
        char log_path[64];
        snprintf(log_path, sizeof(log_path), "results/logs/mpi_%dp.log", nprocs);
        FILE *lf = fopen(log_path, "w");
        if (!lf) { snprintf(log_path, sizeof(log_path), "mpi_%dp.log", nprocs); lf = fopen(log_path, "w"); }
        if (lf) {
            fprintf(lf, "=== MPI Network Traffic Anomaly Detection ===\n");
            fprintf(lf, "Processes: %d\n", nprocs);
            fprintf(lf, "Records/pass: %ld\n", global_tot / REPEAT_FACTOR);
            fprintf(lf, "Throughput: %.0f rec/s\n", global_tot / max_elapsed);
            double g_acc  = 100.0*(global_TP+global_TN)/(double)(global_TP+global_TN+global_FP+global_FN);
            double g_prec = (global_TP+global_FP)>0 ? 100.0*global_TP/(global_TP+global_FP) : 0.0;
            double g_rec  = (global_TP+global_FN)>0 ? 100.0*global_TP/(global_TP+global_FN) : 0.0;
            double g_f1   = (g_prec+g_rec)>0 ? 2.0*g_prec*g_rec/(g_prec+g_rec) : 0.0;
            double g_rmse = sqrt(global_sse/(global_TP+global_TN+global_FP+global_FN));
            fprintf(lf, "Accuracy:  %.3f%%\n",  g_acc);
            fprintf(lf, "Precision: %.3f%%\n",  g_prec);
            fprintf(lf, "Recall:    %.3f%%\n",  g_rec);
            fprintf(lf, "F1 Score:  %.3f%%\n",  g_f1);
            fprintf(lf, "RMSE:      %.6f \n",   g_rmse);
            fprintf(lf, "Single-pass time: %.4fs\n", single_time);
            fprintf(lf, "Total time (x%d): %.4fs\n", REPEAT_FACTOR, max_elapsed);
            /* re-read serial time to save speedup/efficiency in log */
            FILE *st = fopen("../../results/serial_time.txt", "r");
            if (!st) st = fopen("serial_time.txt", "r");
            if (st) {
                double serial_time2;
                if (fscanf(st, "%lf", &serial_time2) == 1) {
                    double sp = serial_time2 / single_time;
                    double ef = sp / nprocs * 100.0;
                    fprintf(lf, "Speedup: %.2fx\n", sp);
                    fprintf(lf, "Efficiency: %.1f%%\n", ef);
                }
                fclose(st);
            }
            fclose(lf);
            printf("Log saved to %s\n", log_path);
        }
    }

    MPI_Finalize();   /* shut down MPI environment — must be the last MPI call */
    return 0;
}