/**
 * EC7207 HPC Project - Group 12
 * pthreads_analysis.c  –  POSIX Threads manual parallel implementation
 *
 * Manually partitions rows across threads. Each thread independently
 * processes its slice, counts anomalies, then the main thread collects
 * and merges results.
 *
 * Compile:  gcc -O2 -pthread -o pthreads pthreads_analysis.c -lm
 * Run:      ./pthreads ../data/train_data.bin 8
 *           (second arg = number of threads, default 4)
 */

#include "common.h"
#include <pthread.h>

/* ── Thread argument struct ──────────────────────────────────────────────── */
typedef struct {
    /* Input (read-only shared) */
    const float  *X;
    const double *means;
    const double *stds;
    int           n_feat;
    double        threshold;

    /* Row range assigned to this thread */
    int row_start;
    int row_end;

    /* Output (written by this thread only — no race condition) */
    int *flags;          /* points into the shared flags array at row_start */
    int  anomaly_count;

    /* Thread ID for diagnostics */
    int tid;
} ThreadArgs;

/* ── Worker function ─────────────────────────────────────────────────────── */
static void *anomaly_worker(void *arg) {
    ThreadArgs *a = (ThreadArgs *)arg;
    a->anomaly_count = 0;

    for (int i = a->row_start; i < a->row_end; i++) {
        a->flags[i] = 0;
        for (int f = 0; f < a->n_feat; f++) {
            if (a->stds[f] < 1e-9) continue;
            double z = fabs(((double)a->X[(size_t)i * a->n_feat + f]
                             - a->means[f]) / a->stds[f]);
            if (z > a->threshold) { a->flags[i] = 1; break; }
        }
        a->anomaly_count += a->flags[i];
    }
    return NULL;
}

/* ── Stats worker: compute partial sum for a feature range ───────────────── */
typedef struct {
    const float *X;
    double      *means;
    double      *stds;
    int          n_rows;
    int          n_feat;
    int          feat_start;
    int          feat_end;
} StatsArgs;

static void *stats_worker(void *arg) {
    StatsArgs *a = (StatsArgs *)arg;
    for (int f = a->feat_start; f < a->feat_end; f++) {
        double s = 0.0;
        for (int i = 0; i < a->n_rows; i++)
            s += a->X[(size_t)i * a->n_feat + f];
        a->means[f] = s / a->n_rows;
    }
    for (int f = a->feat_start; f < a->feat_end; f++) {
        double s = 0.0;
        for (int i = 0; i < a->n_rows; i++) {
            double d = a->X[(size_t)i * a->n_feat + f] - a->means[f];
            s += d * d;
        }
        a->stds[f] = sqrt(s / a->n_rows);
    }
    return NULL;
}

/* ── Accuracy metrics ────────────────────────────────────────────────────── */
static void print_metrics(const int *pred, const int *truth, int n) {
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < n; i++) {
        if      (pred[i] == 1 && truth[i] == 1) tp++;
        else if (pred[i] == 0 && truth[i] == 0) tn++;
        else if (pred[i] == 1 && truth[i] == 0) fp++;
        else                                      fn++;
    }
    double accuracy  = (double)(tp + tn) / n;
    double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
    double recall    = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
    double f1        = (precision + recall) > 0
                       ? 2.0 * precision * recall / (precision + recall) : 0.0;
    printf("  TP=%-7d FP=%-7d TN=%-7d FN=%-7d\n", tp, fp, tn, fn);
    printf("  Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f\n",
           accuracy, precision, recall, f1);
}

/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    const char *data_path = (argc > 1) ? argv[1] : "../data/train_data.bin";
    int NTHREADS           = (argc > 2) ? atoi(argv[2]) : 4;
    if (NTHREADS < 1) NTHREADS = 1;

    printf("=== POSIX Threads (%d threads) ===\n", NTHREADS);
    Dataset ds = load_bin(data_path);

    double *means = (double *)calloc(ds.n_feat, sizeof(double));
    double *stds  = (double *)calloc(ds.n_feat, sizeof(double));
    int    *flags = (int    *)malloc(ds.n_rows * sizeof(int));

    pthread_t  *threads = (pthread_t *)malloc(NTHREADS * sizeof(pthread_t));

    /* ── Phase 1: Parallel stats (split over features) ─────────────────── */
    double t_start = wall_time();

    {
        StatsArgs *sargs = (StatsArgs *)malloc(NTHREADS * sizeof(StatsArgs));
        int feat_chunk = ds.n_feat / NTHREADS;
        int feat_extra = ds.n_feat % NTHREADS;
        int f_offset = 0;

        for (int t = 0; t < NTHREADS; t++) {
            sargs[t].X          = ds.X;
            sargs[t].means      = means;
            sargs[t].stds       = stds;
            sargs[t].n_rows     = ds.n_rows;
            sargs[t].n_feat     = ds.n_feat;
            sargs[t].feat_start = f_offset;
            f_offset += feat_chunk + (t < feat_extra ? 1 : 0);
            sargs[t].feat_end   = f_offset;
            pthread_create(&threads[t], NULL, stats_worker, &sargs[t]);
        }
        for (int t = 0; t < NTHREADS; t++)
            pthread_join(threads[t], NULL);
        free(sargs);
    }

    /* ── Phase 2: Parallel anomaly detection (split over rows) ─────────── */
    {
        ThreadArgs *args = (ThreadArgs *)malloc(NTHREADS * sizeof(ThreadArgs));
        int base  = ds.n_rows / NTHREADS;
        int extra = ds.n_rows % NTHREADS;
        int offset = 0;

        for (int t = 0; t < NTHREADS; t++) {
            args[t].X           = ds.X;
            args[t].means       = means;
            args[t].stds        = stds;
            args[t].flags       = flags;
            args[t].n_feat      = ds.n_feat;
            args[t].threshold   = ZSCORE_THRESHOLD;
            args[t].row_start   = offset;
            int rows = base + (t < extra ? 1 : 0);
            offset += rows;
            args[t].row_end     = offset;
            args[t].tid         = t;
            pthread_create(&threads[t], NULL, anomaly_worker, &args[t]);
        }

        int total_anomalies = 0;
        for (int t = 0; t < NTHREADS; t++) {
            pthread_join(threads[t], NULL);
            total_anomalies += args[t].anomaly_count;
        }

        double elapsed = wall_time() - t_start;

        printf("\n--- Results ---\n");
        printf("  Threads        : %d\n",     NTHREADS);
        printf("  Rows processed : %d\n",     ds.n_rows);
        printf("  Anomalies found: %d (%.2f%%)\n",
               total_anomalies, 100.0 * total_anomalies / ds.n_rows);
        printf("  Time           : %.6f s\n", elapsed);

        double rmse = compute_rmse(flags, ds.y, ds.n_rows);
        printf("  RMSE vs label  : %.8f\n", rmse);

        printf("\n--- Detection Metrics ---\n");
        print_metrics(flags, ds.y, ds.n_rows);

        /* Correctness check vs serial */
        FILE *sf = fopen("../data/serial_flags.bin", "rb");
        if (sf) {
            int *serial_flags = (int *)malloc(ds.n_rows * sizeof(int));
            fread(serial_flags, sizeof(int), ds.n_rows, sf);
            fclose(sf);
            double diff = compute_rmse(flags, serial_flags, ds.n_rows);
            printf("  RMSE vs serial : %.8f  (%s)\n", diff,
                   diff < 1e-6 ? "PASS" : "MISMATCH");
            free(serial_flags);
        }

        save_flags(flags, ds.n_rows, "../data/pthreads_flags.bin");

        /* Per-thread breakdown */
        printf("\n--- Thread Breakdown ---\n");
        for (int t = 0; t < NTHREADS; t++)
            printf("  Thread %2d: rows [%7d – %7d]  anomalies=%d\n",
                   t, args[t].row_start, args[t].row_end - 1,
                   args[t].anomaly_count);

        free(args);
    }

    free(threads);
    free(means); free(stds); free(flags);
    free_dataset(&ds);
    return 0;
}
