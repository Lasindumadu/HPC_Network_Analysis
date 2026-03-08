/*
 * HPC Network Traffic Analysis - Serial Baseline
 * Course: EC7207 - High Performance Computing
 * Authors: EG/2021/4426, EG/2021/4432, EG/2021/4433
 *
 * REPEAT_FACTOR: process dataset N times to get meaningful
 * execution time for HPC speedup comparison.
 * Set to 1 for single-pass, 50 for ~7s baseline.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_LINE         512
#define MAX_FIELDS       50
#define HASH_SIZE        1024
#define ATTACK_THRESHOLD 4
#define REPEAT_FACTOR    50    /* ← tune for meaningful timing */

/* ── Column indices (set by detect_columns) ─── */
static int C_STATE, C_PROTO, C_SERVICE, C_SPKTS, C_DPKTS;
static int C_RATE, C_STTL, C_DTTL, C_SLOAD;
static int C_SLOSS, C_DLOSS, C_SJIT, C_DJIT;
static int C_CT_SRV, C_CT_STT, C_CT_DPT, C_LABEL;
static int MIN_F;

/* ── Timing ──────────────────────────────────── */
static double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);        /* read high-resolution monotonic clock — unaffected by system time changes */
    return t.tv_sec + t.tv_nsec * 1e-9;       /* convert to fractional seconds (e.g. 1.426743s) */
}

/* ── Row parsing — zero-malloc stack-based parser ───
 * Each field is stored in buf[field][64] on the stack.
 * No heap allocation → no malloc lock contention.
 * Identical logic to original; results unchanged.    */
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

/* ── Column detection ────────────────────────── */
static void detect_columns(const char *hdr) {
    Row r; char *h = strdup(hdr);              /* duplicate header string so we can modify it without altering the original */
    h[strcspn(h, "\n")] = '\0';               /* find first newline character and replace it with null terminator */
    parse(h, &r);

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
        for (char *p = r.buf[i]; *p; p++)
            if (*p>='A' && *p<='Z') *p += 32;
        for (int j = 0; j < nm; j++)
            if (strcmp(r.buf[i], m[j].n) == 0) { *m[j].t = i; break; }
    }

    if (C_LABEL < 0) { fprintf(stderr, "ERROR: 'label' not found\n"); exit(1); }  /* terminate immediately — cannot compute accuracy without label column */
    MIN_F = C_LABEL + 1;

    printf("Columns: total=%d | label=%d | state=%d | dttl=%d | ct_state_ttl=%d\n\n",
           r.n, C_LABEL, C_STATE, C_DTTL, C_CT_STT);
    free_row(&r);
    free(h);
}

/* ═══════════════════════════════════════════════
 * DETECTION ENGINE  (no labels used)
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
 *
 *  NEGATIVE signals (suppress normals):
 *   sttl=31         -4  (0.0% attack, 16,702 normals)
 *   dttl=29         -4  (0.0% attack, 16,668 normals)
 *   ct_state_ttl=0  -3  (0.9% attack, 16,897 normals)
 *   state=CON       -2  (5.0% attack,  6,633 normals)
 *   state=REQ       -2  (7.3% attack,  1,707 normals)
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
    int   cst   = fi(r, C_CT_STT), cdp  = fi(r, C_CT_DPT);
    int   csv   = fi(r, C_CT_SRV);
    int   s     = 0;

    /* positive signals */
    if (!strcmp(st,"INT"))                                       s += 5;
    if (dttl==60 || dttl==253)                                   s += 5;
    if (!strcmp(pr,"unas") || !strcmp(pr,"sctp") ||
        !strcmp(pr,"any")  || !strcmp(pr,"gre")  ||
        !strcmp(pr,"ospf"))                                      s += 5;
    if (!strcmp(svc,"pop3") || !strcmp(svc,"ssl") ||
        !strcmp(svc,"snmp"))                                     s += 5;
    if (cst == 2)                                                s += 3;
    if (dttl == 0)                                               s += 3;
    if (sttl == 254)                                             s += 2;
    if (sttl == 255)                                             s += 2;
    if (!strcmp(svc,"dns"))                                      s += 2;
    if (cdp > 10)                                                s += 2;
    if (sload >  1000000.0f)                                     s += 1;
    if (sload > 10000000.0f)                                     s += 1;
    if (sload > 50000000.0f)                                     s += 1;
    if (rate  > 100000.0f)                                       s += 1;
    if (rate  > 166666.0f)                                       s += 1;
    if (dpkts == 0 && spkts > 2)                                 s += 1;
    if (csv > 20)                                                s += 1;
    if (sjit > 1000.0f || djit > 1000.0f)                       s += 1;
    if (loss > 5 && (sjit > 500.0f || djit > 500.0f))           s += 1;

    /* negative signals */
    if (sttl == 31)          s -= 4;
    if (dttl == 29)          s -= 4;
    if (cst  == 0)           s -= 3;
    if (!strcmp(st,"CON"))   s -= 2;
    if (!strcmp(st,"REQ"))   s -= 2;

    return s;
}

/* ── Proto stats table ───────────────────────── */
typedef struct { char name[32]; long att, norm; } ProtoStat;
static ProtoStat ptable[HASH_SIZE];

static ProtoStat *proto_slot(const char *p) {
    unsigned h = 5381;                                              /* DJB2 hash seed — chosen for low collision rate on short strings */
    for (const char *c = p; *c; c++) h = h*33 + (unsigned char)*c; /* DJB2 hash: multiply-add loop over each character of the protocol name */
    h %= HASH_SIZE;                                                 /* map hash to table index within [0, HASH_SIZE-1] */
    if (!ptable[h].name[0]) strncpy(ptable[h].name, p, 31);        /* first visit to this slot: store protocol name, capped at 31 chars */
    return &ptable[h];
}

/* ═══════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    const char *file = argc > 1 ? argv[1] :
        "data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv";

    printf("=== Serial Network Traffic Anomaly Detection ===\n");
    printf("File: %s\n", file);
    printf("Repeat factor: %d  (total records: ~%d)\n\n",
           REPEAT_FACTOR, 82332 * REPEAT_FACTOR);

    /* detect columns from header */
    FILE *fp = fopen(file, "r");                /* open dataset CSV file in read mode */
    if (!fp) { perror(file); return 1; }        /* perror prints OS-level error message (e.g. "No such file") then exit */
    char ln[MAX_LINE];
    if (!fgets(ln, MAX_LINE, fp)) {   /* FIX: check return value */
        fprintf(stderr, "Empty file\n"); fclose(fp); return 1;
    }
    detect_columns(ln);

    /* count records */
    int nrec = 0;
    while (fgets(ln, MAX_LINE, fp)) if (ln[0] != '\n') nrec++;  /* count non-empty lines to know records per pass before main loop */
    fclose(fp);                                                  /* close file after counting — will reopen once per repeat in main loop */
    printf("Records per pass: %d\n\n", nrec);

    /* main detection loop */
    long tot = 0, TP = 0, TN = 0, FP = 0, FN = 0;
    double sse = 0.0;

    double t0 = now();                          /* record start time — measured after file pre-scan so I/O setup is excluded */

    for (int rep = 0; rep < REPEAT_FACTOR; rep++) {
        fp = fopen(file, "r");                  /* reopen file at start of each repeat — serial reads from disk every pass */
        if (!fp) { perror(file); return 1; }

        if (!fgets(ln, MAX_LINE, fp)) {   /* FIX: check return value */
            fclose(fp); break;
        }

        while (fgets(ln, MAX_LINE, fp)) {
            ln[strcspn(ln, "\n")] = '\0';       /* strip trailing newline before parsing — prevents it being included in field values */
            if (!ln[0]) continue;

            Row r;
            if (parse(ln, &r) < MIN_F) { free_row(&r); continue; }

            int pred = (detect(&r) >= ATTACK_THRESHOLD) ? 1 : 0;
            int act  = fi(&r, C_LABEL);   /* ground truth — RMSE only */

            sse += (double)(pred - act) * (pred - act);
            tot++;

            ProtoStat *ps = proto_slot(fs(&r, C_PROTO));  /* look up or create hash table entry for this record's protocol */
            if (pred) ps->att++;  else ps->norm++;

            /* confusion matrix — first pass only */
            if (rep == 0) {
                if      (act==1 && pred==1) TP++;
                else if (act==0 && pred==0) TN++;
                else if (act==0 && pred==1) FP++;
                else                        FN++;
            }

            free_row(&r);
        }
        fclose(fp);   /* close file at end of each repeat pass before reopening next iteration */
    }

    double elapsed     = now() - t0;              /* total wall-clock time for all REPEAT_FACTOR passes */
    double single_time = elapsed / REPEAT_FACTOR;
    long   single_pass = tot / REPEAT_FACTOR;

    double rmse      = sqrt(sse / tot);           /* root mean squared error: sqrt(sum((pred-actual)²) / total) */
    double accuracy  = 100.0 * (TP + TN) / single_pass;
    double precision = (TP+FP) > 0 ? 100.0*TP/(TP+FP) : 0.0;
    double recall    = (TP+FN) > 0 ? 100.0*TP/(TP+FN) : 0.0;
    double f1        = (precision+recall) > 0
                       ? 2.0*precision*recall/(precision+recall) : 0.0;

    /* ── results ─── */
    printf("=== Detection Results ===\n");
    printf("Records/pass: %ld | Passes: %d | Total processed: %ld\n",
           single_pass, REPEAT_FACTOR, tot);
    printf("Attacks: %ld | Normal: %ld  (first pass)\n", TP+FP, TN+FN);
    printf("Time (total x%d): %.4fs | Throughput: %.0f rec/s\n\n",
           REPEAT_FACTOR, elapsed, tot/elapsed);

    printf("=== Protocol Stats (first pass, >= 50 records) ===\n");
    printf("%-15s %8s %8s\n", "Proto", "Att", "Norm");
    printf("%-15s %8s %8s\n", "-----", "---", "----");
    for (int i = 0; i < HASH_SIZE; i++) {
        long total = ptable[i].att + ptable[i].norm;
        if (ptable[i].name[0] && total >= (long)50*REPEAT_FACTOR)
            printf("%-15s %8ld %8ld\n",
                   ptable[i].name,
                   ptable[i].att  / REPEAT_FACTOR,
                   ptable[i].norm / REPEAT_FACTOR);
    }

    printf("\n=== Confusion Matrix (first pass) ===\n");
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

    /* save single-pass time for speedup comparison */
    FILE *tf = fopen("../../results/serial_time.txt", "w");  /* try project-relative path first */
    if (!tf) tf = fopen("serial_time.txt", "w");             /* fallback to current directory if run from different location */
    if (tf) {
        fprintf(tf, "%.6f\n", single_time);                  /* write single-pass time so OpenMP/MPI can read it for speedup calculation */
        fclose(tf);
    }
    printf("\nSingle-pass time: %.4fs  (saved for speedup comparison)\n",
           single_time);
    printf("Total time (x%d): %.4fs\n", REPEAT_FACTOR, elapsed);

    /* save log for chart generation */
    { int _r = system("mkdir -p results/logs"); (void)_r; }
    FILE *lf = fopen("results/logs/serial.log", "w");
    if (!lf) lf = fopen("serial.log", "w");
    if (lf) {
        fprintf(lf, "=== Serial Network Traffic Anomaly Detection ===\n");
        fprintf(lf, "Records/pass: %ld\n", (long)nrec);
        fprintf(lf, "Throughput: %.0f rec/s\n", (double)nrec / single_time);
        fprintf(lf, "Accuracy:  %.3f%%\n",  accuracy);
        fprintf(lf, "Precision: %.3f%%\n",  precision);
        fprintf(lf, "Recall:    %.3f%%\n",  recall);
        fprintf(lf, "F1 Score:  %.3f%%\n",  f1);
        fprintf(lf, "RMSE:      %.6f \n",   rmse);
        fprintf(lf, "Single-pass time: %.4fs\n", single_time);
        fprintf(lf, "Total time (x%d): %.4fs\n", REPEAT_FACTOR, elapsed);
        fclose(lf);
        printf("Log saved to results/logs/serial.log\n");
    }

    return 0;
}