/*
 * HPC Network Traffic Analysis - MPI Implementation
 * Course: EC7207 - High Performance Computing
 * Authors: EG/2021/4426, EG/2021/4432, EG/2021/4433
 *
 * WHAT THIS PROGRAM DOES:
 * - Reads a CSV file containing network traffic records
 * - Distributes records across multiple CPU processes (MPI)
 * - Each process scores its chunk using a rule-based detection engine
 * - Results are combined at Rank 0 to produce final metrics
 */

/* _POSIX_C_SOURCE must be defined FIRST before any #include
   This unlocks POSIX functions like strdup(), clock_gettime()
   that are not available in standard C by default              */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>    /* printf, fopen, fgets, fclose          */
#include <stdlib.h>   /* malloc, free, atof, atoi              */
#include <string.h>   /* strcmp, strcpy, memcpy, strdup        */
#include <math.h>     /* sqrt() — used for RMSE calculation    */
#include <time.h>     /* clock_gettime() — for timing          */
#include <mpi.h>      /* All MPI functions (core of this file) */

/* ── Constants ─────────────────────────────────────────────── */
#define MAX_LINE         512      /* max characters per CSV line          */
#define MAX_FIELDS       50       /* max columns in the CSV               */
#define ATTACK_THRESHOLD 4        /* score >= 4 means "attack detected"   */
#define REPEAT_FACTOR    50       /* process dataset 50x for timing tests */
#define MAX_RECORDS      750000   /* max rows we can load into memory     */

/* ── Column index variables ─────────────────────────────────── 
   These store WHICH column number each feature is in the CSV.
   Set dynamically by detect_columns() from the header row.    */
static int C_STATE, C_PROTO, C_SERVICE, C_SPKTS, C_DPKTS;
static int C_RATE, C_STTL, C_DTTL, C_SLOAD;
static int C_SLOSS, C_DLOSS, C_SJIT, C_DJIT;
static int C_CT_SRV, C_CT_STT, C_CT_DPT, C_LABEL;
static int MIN_F;       /* minimum fields needed per row (= label col + 1) */
static int TOTAL_COLS;  /* total number of columns detected in CSV          */

/* ══════════════════════════════════════════════════════════════
 * TIMING FUNCTION
 * Returns current wall-clock time in seconds (with nanosecond
 * precision). Used to measure how long processing takes.
 * ══════════════════════════════════════════════════════════════ */
static double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);  /* monotonic = never goes backward */
    return t.tv_sec + t.tv_nsec * 1e-9; /* convert to seconds as a double  */
}

/* ══════════════════════════════════════════════════════════════
 * ROW STRUCTURE
 * Instead of using malloc() for every CSV row (which would cause
 * lock contention in parallel code), we use a fixed-size struct
 * on the stack. Much faster in multi-process environments.
 * ══════════════════════════════════════════════════════════════ */
#define FIELD_LEN 64   /* max characters in a single CSV field */
typedef struct {
    char buf[MAX_FIELDS][FIELD_LEN];  /* 2D array: [column][characters] */
    int  n;                           /* how many columns were parsed    */
} Row;

/* Resets a Row struct for reuse (just sets count to 0) */
static inline void free_row(Row *r) { r->n = 0; }

/* ══════════════════════════════════════════════════════════════
 * CSV LINE PARSER
 * Splits a comma-separated line into individual fields.
 * Also strips leading/trailing spaces and quote characters.
 * 
 * Example: '  "hello" , world , 42 '
 *   → buf[0]="hello"  buf[1]="world"  buf[2]="42"   n=3
 * ══════════════════════════════════════════════════════════════ */
static int parse(const char *ln, Row *r) {
    r->n = 0;
    const char *p = ln;

    while (*p && r->n < MAX_FIELDS) {
        const char *start = p;

        /* advance p to next comma or end of string */
        while (*p && *p != ',') p++;

        /* strip leading whitespace and quote characters */
        while (*start == ' ' || *start == '"' || *start == '\'') start++;

        int len = (int)(p - start);

        /* strip trailing whitespace and quote characters */
        while (len > 0 && (start[len-1] <= ' ' ||
               start[len-1] == '"' || start[len-1] == '\'')) len--;

        /* truncate if field is too long for our buffer */
        if (len >= FIELD_LEN) len = FIELD_LEN - 1;

        /* copy field into the Row buffer and null-terminate */
        memcpy(r->buf[r->n], start, len);
        r->buf[r->n][len] = '\0';
        r->n++;

        if (*p == ',') p++;  /* skip the comma before next field */
    }
    return r->n;
}

/* ── Helper functions to read a field as float, int, or string ── */
static inline float       ff(Row *r, int c) {
    /* return field c as float; 0.0 if column missing or empty */
    return (c>=0 && c<r->n && r->buf[c][0]) ? atof(r->buf[c]) : 0.0f;
}
static inline int         fi(Row *r, int c) {
    /* return field c as int; 0 if column missing or empty */
    return (c>=0 && c<r->n && r->buf[c][0]) ? atoi(r->buf[c]) : 0;
}
static inline const char *fs(Row *r, int c) {
    /* return field c as string; "" if column missing */
    return (c>=0 && c<r->n) ? r->buf[c] : "";
}

/* ══════════════════════════════════════════════════════════════
 * COLUMN DETECTION
 * Reads the CSV header row and finds which column number
 * corresponds to each feature we need (state, proto, label, etc.)
 * All MPI ranks call this on the same header string.
 * ══════════════════════════════════════════════════════════════ */
static void detect_columns(const char *hdr, int rank) {
    Row r;
    char *h = strdup(hdr);              /* make a writable copy of header */
    h[strcspn(h, "\n")] = '\0';         /* strip trailing newline          */
    parse(h, &r);
    TOTAL_COLS = r.n;

    /* map of feature name → pointer to its column index variable */
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

    /* initialise all column indices to -1 (not found) */
    for (int i = 0; i < nm; i++) *m[i].t = -1;

    /* scan each header column, lowercase it, match against feature names */
    for (int i = 0; i < r.n; i++) {
        /* convert to lowercase so matching is case-insensitive */
        for (char *p = r.buf[i]; *p; p++)
            if (*p>='A' && *p<='Z') *p += 32;

        for (int j = 0; j < nm; j++)
            if (strcmp(r.buf[i], m[j].n) == 0) { *m[j].t = i; break; }
    }

    /* 'label' column is mandatory — abort all ranks if missing */
    if (C_LABEL < 0) {
        fprintf(stderr, "[rank %d] ERROR: 'label' column not found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* minimum fields required: at least up to and including label column */
    MIN_F = C_LABEL + 1;

    /* only rank 0 prints — avoids duplicate output from all processes */
    if (rank == 0)
        printf("Columns: total=%d | label=%d | state=%d | dttl=%d | ct_state_ttl=%d\n\n",
               r.n, C_LABEL, C_STATE, C_DTTL, C_CT_STT);

    free_row(&r);
    free(h);
}

/* ══════════════════════════════════════════════════════════════
 * DETECTION ENGINE
 * Rule-based scoring system. Examines traffic features and
 * assigns a score. Score >= ATTACK_THRESHOLD (4) = attack.
 *
 * Each rule is based on statistical analysis of the dataset:
 *   e.g. state=INT appears in 86.9% of attack records → +5
 *        sttl=31 appears in 0% of attacks → -4 (clearly normal)
 *
 * NO labels are used here — purely feature-based detection.
 * ══════════════════════════════════════════════════════════════ */
static int detect(Row *r) {
    /* read all relevant features from the parsed row */
    const char *st  = fs(r, C_STATE);    /* connection state (INT/CON/REQ) */
    const char *pr  = fs(r, C_PROTO);    /* protocol (tcp/udp/unas/sctp...) */
    const char *svc = fs(r, C_SERVICE);  /* service type (dns/pop3/ssl...)   */
    int   spkts = fi(r, C_SPKTS);        /* source packets sent              */
    int   dpkts = fi(r, C_DPKTS);        /* destination packets sent         */
    float rate  = ff(r, C_RATE);         /* overall packet rate              */
    float sload = ff(r, C_SLOAD);        /* source bits per second           */
    int   sttl  = fi(r, C_STTL);         /* source time-to-live value        */
    int   dttl  = fi(r, C_DTTL);         /* destination time-to-live value   */
    float sjit  = ff(r, C_SJIT);         /* source jitter (ms)               */
    float djit  = ff(r, C_DJIT);         /* destination jitter (ms)          */
    int   loss  = fi(r, C_SLOSS) + fi(r, C_DLOSS); /* total packet loss      */
    int   cst   = fi(r, C_CT_STT);       /* connection state TTL counter     */
    int   cdp   = fi(r, C_CT_DPT);       /* dst port connection count        */
    int   csv   = fi(r, C_CT_SRV);       /* service connection count         */
    int   s     = 0;                     /* cumulative score (starts at 0)   */

    /* ── POSITIVE SIGNALS (push score toward "attack") ──────── */

    if (!strcmp(st,"INT"))                   s += 5; /* INT state = 86.9% attack */
    if (dttl==60 || dttl==253)               s += 5; /* these dttl = >99.6% attack */
    if (!strcmp(pr,"unas") || !strcmp(pr,"sctp") ||
        !strcmp(pr,"any")  || !strcmp(pr,"gre")  ||
        !strcmp(pr,"ospf"))                  s += 5; /* unusual protocols = >94% attack */
    if (!strcmp(svc,"pop3") || !strcmp(svc,"ssl") ||
        !strcmp(svc,"snmp"))                 s += 5; /* these services = 100% attack */
    if (cst == 2)                            s += 3; /* ct_state_ttl=2 = 87% attack */
    if (dttl == 0)                           s += 3; /* dttl=0 = 82.7% attack */
    if (sttl == 254)                         s += 2; /* sttl=254 = 72-100% attack */
    if (sttl == 255)                         s += 2; /* sttl=255 = 72-100% attack */
    if (!strcmp(svc,"dns"))                  s += 2; /* dns service = 85.6% attack */
    if (cdp > 10)                            s += 2; /* many dst port conns = scan */
    if (sload >  1000000.0f)                 s += 1; /* high source load tier 1 */
    if (sload > 10000000.0f)                 s += 1; /* high source load tier 2 */
    if (sload > 50000000.0f)                 s += 1; /* high source load tier 3 */
    if (rate  > 100000.0f)                   s += 1; /* high packet rate tier 1 */
    if (rate  > 166666.0f)                   s += 1; /* high packet rate tier 2 */
    if (dpkts == 0 && spkts > 2)             s += 1; /* one-way traffic = suspicious */
    if (csv > 20)                            s += 1; /* many connections to same service */
    if (sjit > 1000.0f || djit > 1000.0f)   s += 1; /* high jitter = unusual */
    if (loss > 5 && (sjit>500.0f || djit>500.0f)) s += 1; /* loss + jitter = backdoor signal */

    /* ── NEGATIVE SIGNALS (push score toward "normal") ─────── */

    if (sttl == 31)           s -= 4; /* sttl=31 = 0.0% attack — definitely normal */
    if (dttl == 29)           s -= 4; /* dttl=29 = 0.0% attack — definitely normal */
    if (cst  == 0)            s -= 3; /* ct_state_ttl=0 = 0.9% attack */
    if (!strcmp(st,"CON"))    s -= 2; /* CON state = 5.0% attack — mostly normal */
    if (!strcmp(st,"REQ"))    s -= 2; /* REQ state = 7.3% attack — mostly normal */

    return s;  /* caller compares: s >= ATTACK_THRESHOLD → attack */
}

/* ══════════════════════════════════════════════════════════════
 * MAIN — MPI EXECUTION FLOW
 *
 * Step 1: Rank 0 reads entire CSV into memory
 * Step 2: Broadcast header + record count to all ranks
 * Step 3: Each rank calculates its slice boundaries
 * Step 4: Rank 0 sends each rank its slice of rows
 * Step 5: Each rank independently scores its chunk
 * Step 6: MPI_Reduce gathers all results to rank 0
 * Step 7: Rank 0 computes and prints final metrics
 * ══════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {

    /* Initialise MPI — must be the very first MPI call */
    MPI_Init(&argc, &argv);

    int rank;    /* this process's ID: 0, 1, 2, ... nprocs-1 */
    int nprocs;  /* total number of processes launched        */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const char *file = argc > 1 ? argv[1] :
        "data/UNSW-NB15_1.csv/UNSW-NB15_1_with_header.csv";

    /* Only rank 0 prints the header — prevents N duplicate lines */
    if (rank == 0) {
        printf("=== MPI Network Traffic Anomaly Detection ===\n");
        printf("Processes: %d | Repeat factor: %d\n", nprocs, REPEAT_FACTOR);
        printf("File: %s\n", file);
    }

    /* ── STEP 1: Rank 0 reads all CSV records into memory ───── */
    char header[MAX_LINE] = {0};
    int  total_lines = 0;
    char (*all_lines)[MAX_LINE] = NULL;  /* 2D array: [row][characters] */

    if (rank == 0) {
        FILE *fp = fopen(file, "r");
        if (!fp) {
            perror(file);
            MPI_Abort(MPI_COMM_WORLD, 1);  /* kill all processes if file not found */
        }

        /* read and store the header line separately */
        if (!fgets(header, MAX_LINE, fp)) {
            fprintf(stderr, "Empty file\n");
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* allocate memory for all rows at once */
        all_lines = malloc(sizeof(*all_lines) * MAX_RECORDS);
        if (!all_lines) {
            fprintf(stderr, "Out of memory\n");
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* read every data row into all_lines array */
        char ln[MAX_LINE];
        while (fgets(ln, MAX_LINE, fp) && total_lines < MAX_RECORDS) {
            ln[strcspn(ln, "\n")] = '\0';  /* strip newline */
            if (!ln[0]) continue;           /* skip blank lines */
            snprintf(all_lines[total_lines], MAX_LINE, "%s", ln);
            total_lines++;
        }
        fclose(fp);
        printf("Records per pass: %d\n\n", total_lines);
    }

    /* ── STEP 2: Broadcast header and record count to all ranks ─
       Without this, ranks 1..N-1 don't know the column layout
       or how many records exist.                                 */
    MPI_Bcast(header,       MAX_LINE, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_lines, 1,        MPI_INT,  0, MPI_COMM_WORLD);

    /* Every rank now parses the same header to find column indices */
    detect_columns(header, rank);

    /* ── STEP 3: Calculate this rank's slice boundaries ────────
       Example: 82332 records, 4 processes
         base=20583, rem=0
         rank 0: rows   0..20582  (20583 rows)
         rank 1: rows 20583..41165
         rank 2: rows 41166..61748
         rank 3: rows 61749..82331
       If records don't divide evenly, first `rem` ranks get +1 row */
    int base     = total_lines / nprocs;
    int rem      = total_lines % nprocs;
    int my_count = base + (rank < rem ? 1 : 0);  /* rows this rank processes */
    int my_start = rank * base + (rank < rem ? rank : rem);  /* starting row index */

    /* ── STEP 4: Distribute rows — rank 0 sends, others receive ─ */
    char *my_lines = malloc((size_t)my_count * MAX_LINE);
    if (!my_lines) {
        fprintf(stderr, "[rank %d] Out of memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        /* Send each other rank its slice using point-to-point MPI_Send */
        for (int dest = 1; dest < nprocs; dest++) {
            int d_count = base + (dest < rem ? 1 : 0);
            int d_start = dest * base + (dest < rem ? dest : rem);
            MPI_Send(all_lines[d_start], d_count * MAX_LINE,
                     MPI_CHAR, dest, 0, MPI_COMM_WORLD);
        }
        /* Rank 0 copies its own slice locally (no network needed) */
        memcpy(my_lines, all_lines[my_start], (size_t)my_count * MAX_LINE);
        free(all_lines);   /* free the full dataset — no longer needed */
        all_lines = NULL;
    } else {
        /* Worker ranks wait to receive their slice from rank 0 */
        MPI_Recv(my_lines, my_count * MAX_LINE,
                 MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* ── STEP 5: Each rank independently scores its chunk ───────
       REPEAT_FACTOR=50 means we process the data 50 times to get
       a stable timing measurement (avoids noise from a single pass).
       Confusion matrix is only counted on the FIRST pass (rep==0). */
    long   local_TP  = 0, local_TN  = 0;
    long   local_FP  = 0, local_FN  = 0;
    double local_sse = 0.0;   /* sum of squared errors for RMSE */
    long   local_tot = 0;     /* total records processed         */

    /* Sync all ranks before starting the timer */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = now();

    for (int rep = 0; rep < REPEAT_FACTOR; rep++) {
        for (int i = 0; i < my_count; i++) {
            char ln[MAX_LINE];
            snprintf(ln, MAX_LINE, "%s", my_lines + (size_t)i * MAX_LINE);
            if (!ln[0]) continue;

            Row r;
            /* skip rows that don't have enough fields */
            if (parse(ln, &r) < MIN_F) { free_row(&r); continue; }

            /* run detection engine — returns a score */
            int pred = (detect(&r) >= ATTACK_THRESHOLD) ? 1 : 0;
            int act  = fi(&r, C_LABEL);  /* ground truth (1=attack, 0=normal) */

            /* squared error: (predicted - actual)^2 */
            local_sse += (double)(pred - act) * (pred - act);
            local_tot++;

            /* confusion matrix only counted on first pass to avoid x50 inflation */
            if (rep == 0) {
                if      (act==1 && pred==1) local_TP++;  /* correctly caught attack  */
                else if (act==0 && pred==0) local_TN++;  /* correctly marked normal  */
                else if (act==0 && pred==1) local_FP++;  /* false alarm              */
                else                        local_FN++;  /* missed attack            */
            }

            free_row(&r);
        }
    }

    /* Sync all ranks after processing — wall time = slowest rank */
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = now() - t0;

    free(my_lines);

    /* ── STEP 6: Reduce all partial results to rank 0 ──────────
       MPI_Reduce combines values from all ranks using an operation
       (MPI_SUM, MPI_MAX) and delivers the result to rank 0.      */
    long   global_TP=0, global_TN=0, global_FP=0, global_FN=0;
    double global_sse=0.0;
    long   global_tot=0;
    double max_elapsed;

    MPI_Reduce(&local_TP,  &global_TP,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_TN,  &global_TN,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_FP,  &global_FP,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_FN,  &global_FN,  1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_tot, &global_tot, 1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
    /* MPI_MAX ensures we use the SLOWEST rank's time (true wall clock) */
    MPI_Reduce(&elapsed,   &max_elapsed,1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ── STEP 7: Rank 0 computes and prints all metrics ─────── */
    if (rank == 0) {
        long   single_pass = global_tot / REPEAT_FACTOR;
        double single_time = max_elapsed / REPEAT_FACTOR;
        double rmse        = sqrt(global_sse / global_tot);

        /* standard classification metrics */
        double accuracy  = 100.0 * (global_TP + global_TN) / single_pass;
        double precision = (global_TP + global_FP) > 0
                           ? 100.0 * global_TP / (global_TP + global_FP) : 0.0;
        double recall    = (global_TP + global_FN) > 0
                           ? 100.0 * global_TP / (global_TP + global_FN) : 0.0;
        double f1        = (precision + recall) > 0
                           ? 2.0 * precision * recall / (precision + recall) : 0.0;

        printf("=== Detection Results ===\n");
        printf("Records/pass: %ld | Passes: %d | Total processed: %ld\n",
               single_pass, REPEAT_FACTOR, global_tot);
        printf("Time (total x%d): %.4fs | Throughput: %.0f rec/s\n\n",
               REPEAT_FACTOR, max_elapsed, global_tot / max_elapsed);

        printf("=== Confusion Matrix (first pass) ===\n");
        printf("              Predicted\n");
        printf("Actual   Normal   Attack\n");
        printf("Normal  %7ld  %7ld   (FP=%ld)\n", global_TN, global_FP, global_FP);
        printf("Attack  %7ld  %7ld   (FN=%ld)\n", global_FN, global_TP, global_FN);

        printf("\n=== Accuracy Metrics ===\n");
        printf("Accuracy:  %7.3f%%\n",  accuracy);
        printf("Precision: %7.3f%%\n",  precision);
        printf("Recall:    %7.3f%%\n",  recall);
        printf("F1 Score:  %7.3f%%\n",  f1);
        printf("RMSE:      %.6f\n",     rmse);

        /* grade the result */
        if      (accuracy > 85.0) printf("Status: EXCELLENT\n");
        else if (accuracy > 75.0) printf("Status: GOOD\n");
        else if (accuracy > 65.0) printf("Status: ACCEPTABLE\n");
        else                       printf("Status: POOR\n");

        /* ── Compare against serial baseline if available ───── */
        FILE *tf = fopen("../../results/serial_time.txt", "r");
        if (!tf) tf = fopen("serial_time.txt", "r");
        if (tf) {
            double serial_time;
            if (fscanf(tf, "%lf", &serial_time) == 1) {
                double speedup    = serial_time / single_time;
                double efficiency = speedup / nprocs * 100.0;
                printf("\n=== Speedup vs Serial ===\n");
                printf("Serial time:  %.4fs\n", serial_time);
                printf("MPI time:     %.4fs\n", single_time);
                printf("Processes:    %d\n",    nprocs);
                printf("Speedup:      %.2fx\n", speedup);
                printf("Efficiency:   %.1f%%\n",efficiency);
            }
            fclose(tf);
        }

        /* save MPI single-pass time for hybrid (MPI+OpenMP) comparison */
        FILE *mf = fopen("../../results/mpi_time.txt", "w");
        if (!mf) mf = fopen("mpi_time.txt", "w");
        if (mf) { fprintf(mf, "%.6f\n", single_time); fclose(mf); }

        printf("\nSingle-pass time: %.4fs\n", single_time);
        printf("Total time (x%d): %.4fs\n", REPEAT_FACTOR, max_elapsed);

        /* save structured log for chart generation scripts */
        { int _r = system("mkdir -p results/logs"); (void)_r; }
        char log_path[64];
        snprintf(log_path, sizeof(log_path), "results/logs/mpi_%dp.log", nprocs);
        FILE *lf = fopen(log_path, "w");
        if (!lf) {
            snprintf(log_path, sizeof(log_path), "mpi_%dp.log", nprocs);
            lf = fopen(log_path, "w");
        }
        if (lf) {
            fprintf(lf, "=== MPI Network Traffic Anomaly Detection ===\n");
            fprintf(lf, "Processes: %d\n",       nprocs);
            fprintf(lf, "Records/pass: %ld\n",   global_tot / REPEAT_FACTOR);
            fprintf(lf, "Throughput: %.0f rec/s\n", global_tot / max_elapsed);
            double g_acc  = 100.0*(global_TP+global_TN)/(double)(global_TP+global_TN+global_FP+global_FN);
            double g_prec = (global_TP+global_FP)>0 ? 100.0*global_TP/(global_TP+global_FP) : 0.0;
            double g_rec  = (global_TP+global_FN)>0 ? 100.0*global_TP/(global_TP+global_FN) : 0.0;
            double g_f1   = (g_prec+g_rec)>0 ? 2.0*g_prec*g_rec/(g_prec+g_rec) : 0.0;
            double g_rmse = sqrt(global_sse / global_tot);
            fprintf(lf, "Accuracy:  %.3f%%\n",  g_acc);
            fprintf(lf, "Precision: %.3f%%\n",  g_prec);
            fprintf(lf, "Recall:    %.3f%%\n",  g_rec);
            fprintf(lf, "F1 Score:  %.3f%%\n",  g_f1);
            fprintf(lf, "RMSE:      %.6f\n",    g_rmse);
            fprintf(lf, "Single-pass time: %.4fs\n", single_time);
            fprintf(lf, "Total time (x%d): %.4fs\n", REPEAT_FACTOR, max_elapsed);

            /* re-read serial time to include speedup in the log file */
            FILE *st = fopen("../../results/serial_time.txt", "r");
            if (!st) st = fopen("serial_time.txt", "r");
            if (st) {
                double serial_time2;
                if (fscanf(st, "%lf", &serial_time2) == 1) {
                    fprintf(lf, "Speedup: %.2fx\n",    serial_time2 / single_time);
                    fprintf(lf, "Efficiency: %.1f%%\n",(serial_time2/single_time)/nprocs*100.0);
                }
                fclose(st);
            }
            fclose(lf);
            printf("Log saved to %s\n", log_path);
        }
    }

    /* Shut down MPI — must be the very last MPI call */
    MPI_Finalize();
    return 0;
}