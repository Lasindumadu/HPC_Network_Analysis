/*

 *
 * PARALLELISATION STRATEGY:
 *   CPU: Parse CSV, encode string fields (state/proto/service) as ints,
 *        pack into RecordGPU structs, copy to GPU once.
 *   GPU: One CUDA thread per record — massively parallel scoring.
 *        Uses integer codes instead of strcmp (GPU has no libc).
 *   CPU: Copy results back, reduce for confusion matrix + RMSE.
 *
 * CORRECTNESS GUARANTEE:
 *   Identical weights to serial/OpenMP/MPI/Pthreads.
 *   TP=32552  FP=8712  FN=12780  TN=28288
 *
 * COMPILATION:
 *   nvcc -O2 -std=c++11 -o results/cuda src/cuda/network_analysis_cuda.cu
 *
 * RUNNING:
 *   ./results/cuda data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv
 *   ./results/cuda data/... 256    # custom block size
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_LINE         512
#define MAX_FIELDS       50
#define ATTACK_THRESHOLD 4
#define REPEAT_FACTOR    50
#define MAX_RECORDS      750000
#define FIELD_LEN        64

/* CUDA error check macro */
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(_e));      \
            exit(1);                                                  \
        }                                                             \
    } while (0)

/* ── Integer encoding of string fields ─────────────────────────
 * Strings can't be compared efficiently on GPU.
 * Encode on CPU, pass integer codes to GPU kernel.            */

/* state codes */
#define ST_INT   1
#define ST_CON   2
#define ST_REQ   3
#define ST_OTHER 0

/* proto codes */
#define PR_UNAS  1
#define PR_SCTP  2
#define PR_ANY   3
#define PR_GRE   4
#define PR_OSPF  5
#define PR_OTHER 0

/* service codes */
#define SVC_POP3  1
#define SVC_SSL   2
#define SVC_SNMP  3
#define SVC_DNS   4
#define SVC_OTHER 0

static int encode_state(const char *s) {
    if (!strcmp(s,"INT")) return ST_INT;
    if (!strcmp(s,"CON")) return ST_CON;
    if (!strcmp(s,"REQ")) return ST_REQ;
    return ST_OTHER;
}
static int encode_proto(const char *s) {
    if (!strcmp(s,"unas")) return PR_UNAS;
    if (!strcmp(s,"sctp")) return PR_SCTP;
    if (!strcmp(s,"any"))  return PR_ANY;
    if (!strcmp(s,"gre"))  return PR_GRE;
    if (!strcmp(s,"ospf")) return PR_OSPF;
    return PR_OTHER;
}
static int encode_service(const char *s) {
    if (!strcmp(s,"pop3")) return SVC_POP3;
    if (!strcmp(s,"ssl"))  return SVC_SSL;
    if (!strcmp(s,"snmp")) return SVC_SNMP;
    if (!strcmp(s,"dns"))  return SVC_DNS;
    return SVC_OTHER;
}

/* ── Packed record sent to GPU ─────────────────────────────────
 * 14 ints + 4 floats = 72 bytes — fits in L1 cache.          */
typedef struct __align__(16) {
    int   state;
    int   proto;
    int   service;
    int   spkts, dpkts;
    float rate, sload;
    int   sttl, dttl;
    float sjit, djit;
    int   loss;
    int   cst, cdp, csv;
    int   label;          /* ground truth — validation only */
} RecordGPU;

/* ── CPU-side CSV row parser ─────────────────────────────────── */
typedef struct {
    char buf[MAX_FIELDS][FIELD_LEN];
    int  n;
} Row;

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

static inline float       ff_r(Row *r, int c) { return (c>=0&&c<r->n&&r->buf[c][0])?(float)atof(r->buf[c]):0.0f; }
static inline int         fi_r(Row *r, int c) { return (c>=0&&c<r->n&&r->buf[c][0])?atoi(r->buf[c]):0;           }
static inline const char *fs_r(Row *r, int c) { return (c>=0&&c<r->n)?r->buf[c]:"";                              }

/* ── Column indices (CPU only) ───────────────────────────────── */
static int C_STATE, C_PROTO, C_SERVICE, C_SPKTS, C_DPKTS;
static int C_RATE,  C_STTL,  C_DTTL,   C_SLOAD;
static int C_SLOSS, C_DLOSS, C_SJIT,   C_DJIT;
static int C_CT_SRV, C_CT_STT, C_CT_DPT, C_LABEL;
static int MIN_F;

static void detect_columns(const char *hdr) {
    Row r; char *h = strdup(hdr);
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
            if (*q>='A' && *q<='Z') *q += 32;
        for (int j = 0; j < nm; j++)
            if (strcmp(r.buf[i], m[j].n)==0) { *m[j].t = i; break; }
    }

    if (C_LABEL < 0) { fprintf(stderr,"ERROR: 'label' not found\n"); exit(1); }
    MIN_F = C_LABEL + 1;

    printf("Columns: total=%d | label=%d | state=%d | dttl=%d | ct_state_ttl=%d\n\n",
           r.n, C_LABEL, C_STATE, C_DTTL, C_CT_STT);
    free(h);
}

/* ════════════════════════════════════════════════════════════════
 * CUDA KERNEL — one thread per record
 *
 * Uses integer-encoded fields (no strcmp on GPU).
 * Identical scoring weights to serial baseline.
 * ════════════════════════════════════════════════════════════════ */
__global__ void detect_kernel(const RecordGPU *recs, int n,
                               int *d_pred, int *d_act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const RecordGPU *rec = &recs[i];
    int s = 0;

    /* positive signals */
    if (rec->state   == ST_INT)                                 s += 5;
    if (rec->dttl    == 60 || rec->dttl == 253)                 s += 5;
    if (rec->proto   == PR_UNAS || rec->proto == PR_SCTP  ||
        rec->proto   == PR_ANY  || rec->proto == PR_GRE   ||
        rec->proto   == PR_OSPF)                                s += 5;
    if (rec->service == SVC_POP3 || rec->service == SVC_SSL  ||
        rec->service == SVC_SNMP)                               s += 5;
    if (rec->cst     == 2)                                      s += 3;
    if (rec->dttl    == 0)                                      s += 3;
    if (rec->sttl    == 254)                                    s += 2;
    if (rec->sttl    == 255)                                    s += 2;
    if (rec->service == SVC_DNS)                                s += 2;
    if (rec->cdp     > 10)                                      s += 2;
    if (rec->sload   >  1000000.0f)                             s += 1;
    if (rec->sload   > 10000000.0f)                             s += 1;
    if (rec->sload   > 50000000.0f)                             s += 1;
    if (rec->rate    > 100000.0f)                               s += 1;
    if (rec->rate    > 166666.0f)                               s += 1;
    if (rec->dpkts   == 0 && rec->spkts > 2)                   s += 1;
    if (rec->csv     > 20)                                      s += 1;
    if (rec->sjit    > 1000.0f || rec->djit > 1000.0f)         s += 1;
    if (rec->loss    > 5 &&
        (rec->sjit   > 500.0f  || rec->djit > 500.0f))         s += 1;

    /* negative signals */
    if (rec->sttl  == 31)     s -= 4;
    if (rec->dttl  == 29)     s -= 4;
    if (rec->cst   == 0)      s -= 3;
    if (rec->state == ST_CON) s -= 2;
    if (rec->state == ST_REQ) s -= 2;

    d_pred[i] = (s >= ATTACK_THRESHOLD) ? 1 : 0;
    d_act[i]  = rec->label;
}

/* ── CPU wall clock ──────────────────────────────────────────── */
static double cpu_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    const char *file = argc > 1 ? argv[1] :
        "data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv";
    int block_size = argc > 2 ? atoi(argv[2]) : 256;
    if (block_size < 32)   block_size = 32;
    if (block_size > 1024) block_size = 1024;

    /* ── Print GPU info ─────────────────────── */
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) { fprintf(stderr,"No CUDA GPU found.\n"); return 1; }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== CUDA Network Traffic Anomaly Detection ===\n");
    printf("GPU: %s | SMs: %d | Max threads/block: %d\n",
           prop.name, prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("Block size: %d | Repeat factor: %d\n", block_size, REPEAT_FACTOR);
    printf("File: %s\n\n", file);

    /* ── Step 1: Parse CSV on CPU ────────────── */
    FILE *fp = fopen(file, "r");
    if (!fp) { perror(file); return 1; }

    char ln[MAX_LINE];
    if (!fgets(ln, MAX_LINE, fp)) {
        fprintf(stderr,"Empty file\n"); fclose(fp); return 1;
    }
    detect_columns(ln);

    RecordGPU *h_recs = (RecordGPU *)malloc(sizeof(RecordGPU) * MAX_RECORDS);
    if (!h_recs) { fprintf(stderr,"Out of memory\n"); return 1; }

    int total = 0;
    while (fgets(ln, MAX_LINE, fp) && total < MAX_RECORDS) {
        ln[strcspn(ln,"\n")] = '\0';
        if (!ln[0]) continue;

        Row r;
        if (parse_row(ln, &r) < MIN_F) continue;

        RecordGPU *g = &h_recs[total];
        g->state   = encode_state  (fs_r(&r, C_STATE));
        g->proto   = encode_proto  (fs_r(&r, C_PROTO));
        g->service = encode_service(fs_r(&r, C_SERVICE));
        g->spkts   = fi_r(&r, C_SPKTS);
        g->dpkts   = fi_r(&r, C_DPKTS);
        g->rate    = ff_r(&r, C_RATE);
        g->sload   = ff_r(&r, C_SLOAD);
        g->sttl    = fi_r(&r, C_STTL);
        g->dttl    = fi_r(&r, C_DTTL);
        g->sjit    = ff_r(&r, C_SJIT);
        g->djit    = ff_r(&r, C_DJIT);
        g->loss    = fi_r(&r, C_SLOSS) + fi_r(&r, C_DLOSS);
        g->cst     = fi_r(&r, C_CT_STT);
        g->cdp     = fi_r(&r, C_CT_DPT);
        g->csv     = fi_r(&r, C_CT_SRV);
        g->label   = fi_r(&r, C_LABEL);
        total++;
    }
    fclose(fp);
    printf("Records per pass: %d\n\n", total);

    /* ── Step 2: Copy data to GPU (once — reused every pass) ── */
    RecordGPU *d_recs = NULL;
    int       *d_pred = NULL;
    int       *d_act  = NULL;

    CUDA_CHECK(cudaMalloc(&d_recs, sizeof(RecordGPU) * total));
    CUDA_CHECK(cudaMalloc(&d_pred, sizeof(int) * total));
    CUDA_CHECK(cudaMalloc(&d_act,  sizeof(int) * total));
    CUDA_CHECK(cudaMemcpy(d_recs, h_recs,
                          sizeof(RecordGPU) * total,
                          cudaMemcpyHostToDevice));

    int grid_size = (total + block_size - 1) / block_size;

    /* ── Step 3: Run kernel REPEAT_FACTOR times ─────────────── */
    int *h_pred = (int *)malloc(sizeof(int) * total);
    int *h_act  = (int *)malloc(sizeof(int) * total);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double cpu_t0 = cpu_now();
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int rep = 0; rep < REPEAT_FACTOR; rep++) {
        detect_kernel<<<grid_size, block_size>>>(d_recs, total, d_pred, d_act);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    double cpu_elapsed = cpu_now() - cpu_t0;

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));

    /* ── Step 4: Copy results back, reduce on CPU ───────────── */
    CUDA_CHECK(cudaMemcpy(h_pred, d_pred, sizeof(int)*total, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_act,  d_act,  sizeof(int)*total, cudaMemcpyDeviceToHost));

    long   TP = 0, TN = 0, FP = 0, FN = 0;
    double sse = 0.0;
    for (int i = 0; i < total; i++) {
        int p = h_pred[i], a = h_act[i];
        sse += (double)(p - a) * (p - a);
        if      (a==1 && p==1) TP++;
        else if (a==0 && p==0) TN++;
        else if (a==0 && p==1) FP++;
        else                   FN++;
    }
    double total_sse  = sse * REPEAT_FACTOR;
    long   total_recs = (long)total * REPEAT_FACTOR;

    /* ── Step 5: Metrics ─────────────────────── */
    double single_time = (double)gpu_ms / 1000.0 / REPEAT_FACTOR;
    double rmse        = sqrt(total_sse / total_recs);
    double accuracy    = 100.0 * (TP + TN) / (double)total;
    double precision   = (TP+FP)>0 ? 100.0*TP/(TP+FP) : 0.0;
    double recall      = (TP+FN)>0 ? 100.0*TP/(TP+FN) : 0.0;
    double f1          = (precision+recall)>0
                         ? 2.0*precision*recall/(precision+recall) : 0.0;

    printf("=== Detection Results ===\n");
    printf("Records/pass: %d | Passes: %d | Total processed: %ld\n",
           total, REPEAT_FACTOR, total_recs);
    printf("Attacks: %ld | Normal: %ld\n", TP+FP, TN+FN);
    printf("Total time (x%d): %.4fs\n",
           REPEAT_FACTOR, (double)gpu_ms/1000.0);
    printf("CPU wall time (total): %.4fs\n", cpu_elapsed);
    printf("Throughput: %.0f rec/s\n\n",
           (double)total_recs/((double)gpu_ms/1000.0));

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

    /* ── Single-pass time (parsed by web app) ── */
    printf("Single-pass time: %.4fs\n", single_time);
    printf("Grid: %d blocks x %d threads = %d total threads\n",
           grid_size, block_size, grid_size*block_size);

    /* ── Speedup vs serial ─────────────────────── */
    FILE *tf = fopen("results/serial_time.txt", "r");
    if (!tf) tf = fopen("../../results/serial_time.txt", "r");
    if (!tf) tf = fopen("serial_time.txt", "r");
    if (tf) {
        double serial_t;
        if (fscanf(tf, "%lf", &serial_t) == 1) {
            double speedup = serial_t / single_time;
            printf("\n=== Speedup vs Serial ===\n");
            printf("Serial time (1 pass):  %.4fs\n", serial_t);
            printf("CUDA   time (1 pass):  %.4fs\n", single_time);
            printf("Speedup: %.2fx\n", speedup);
            printf("(Note: excludes one-time H2D transfer of %.2f MB)\n",
                   (double)sizeof(RecordGPU)*total/1e6);
        }
        fclose(tf);
    }

    /* ── Save CUDA time ─────────────────────────── */
    FILE *cf = fopen("results/cuda_time.txt", "w");
    if (!cf) cf = fopen("../../results/cuda_time.txt", "w");
    if (!cf) cf = fopen("cuda_time.txt", "w");
    if (cf) { fprintf(cf, "%.6f\n", single_time); fclose(cf); }

    /* ── Save log ───────────────────────────────── */
    { int _r = system("mkdir -p results/logs"); (void)_r; }
    FILE *lf = fopen("results/logs/cuda.log", "w");
    if (!lf) lf = fopen("cuda.log", "w");
    if (lf) {
        fprintf(lf, "=== CUDA Network Traffic Anomaly Detection ===\n");
        fprintf(lf, "GPU: %s\n", prop.name);
        fprintf(lf, "Block size: %d | Grid size: %d\n", block_size, grid_size);
        fprintf(lf, "Records/pass: %d\n", total);
        fprintf(lf, "Throughput: %.0f rec/s\n",
                (double)total_recs/((double)gpu_ms/1000.0));
        fprintf(lf, "Accuracy:  %.3f%%\n",  accuracy);
        fprintf(lf, "Precision: %.3f%%\n",  precision);
        fprintf(lf, "Recall:    %.3f%%\n",  recall);
        fprintf(lf, "F1 Score:  %.3f%%\n",  f1);
        fprintf(lf, "RMSE:      %.6f \n",   rmse);
        fprintf(lf, "Single-pass time: %.4fs\n", single_time);
        fprintf(lf, "Total time (x%d): %.4fs\n",
                REPEAT_FACTOR, (double)gpu_ms/1000.0);
        FILE *st = fopen("results/serial_time.txt", "r");
        if (!st) st = fopen("../../results/serial_time.txt", "r");
        if (!st) st = fopen("serial_time.txt", "r");
        if (st) {
            double st2;
            if (fscanf(st, "%lf", &st2) == 1)
                fprintf(lf, "Speedup: %.2fx\n", st2/single_time);
            fclose(st);
        }
        fclose(lf);
        printf("Log saved to results/logs/cuda.log\n");
    }

    /* cleanup */
    cudaFree(d_recs); cudaFree(d_pred); cudaFree(d_act);
    free(h_recs); free(h_pred); free(h_act);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);

    return 0;
}