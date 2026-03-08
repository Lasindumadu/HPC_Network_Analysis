/**
 * EC7207 HPC Project - Group 12
 * openmp_analysis.c  –  OpenMP shared-memory parallel implementation
 *
 * Parallelises both the statistics computation (over features) and
 * the anomaly detection loop (over rows) using OpenMP pragmas.
 *
 * Compile:  gcc -O2 -fopenmp -o omp openmp_analysis.c -lm
 * Run:      ./omp ../data/train_data.bin 8
 *           (second arg = number of threads, default 4)
 */

#include "common.h"
#include <omp.h>

/* ── OpenMP parallel stats (parallelise over features) ───────────────────── */
static void compute_stats_omp(const float *X, int n_rows, int n_feat,
                               double *means, double *stds) {
    /* Pass 1: mean — each thread owns its feature range */
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < n_feat; f++) {
        double sum = 0.0;
        for (int i = 0; i < n_rows; i++)
            sum += X[(size_t)i * n_feat + f];
        means[f] = sum / n_rows;
    }

    /* Pass 2: std — same partitioning, means[] is read-only now */
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < n_feat; f++) {
        double sum = 0.0;
        for (int i = 0; i < n_rows; i++) {
            double diff = X[(size_t)i * n_feat + f] - means[f];
            sum += diff * diff;
        }
        stds[f] = sqrt(sum / n_rows);
    }
}

/* ── OpenMP parallel anomaly detection (parallelise over rows) ───────────── */
static int detect_anomalies_omp(const float *X, int n_rows, int n_feat,
                                  const double *means, const double *stds,
                                  double threshold, int *flags) {
    int total = 0;

    /*
     * schedule(dynamic, 512):
     *   rows vary in cost (some exit early, some scan all features),
     *   so dynamic scheduling balances load better than static.
     * reduction(+:total):
     *   each thread maintains its own partial count; combined at the end.
     */
    #pragma omp parallel for schedule(dynamic, 512) reduction(+:total)
    for (int i = 0; i < n_rows; i++) {
        flags[i] = 0;
        for (int f = 0; f < n_feat; f++) {
            if (stds[f] < 1e-9) continue;
            double z = fabs(((double)X[(size_t)i * n_feat + f] - means[f]) / stds[f]);
            if (z > threshold) { flags[i] = 1; break; }
        }
        total += flags[i];
    }
    return total;
}

/* ── OpenMP parallel traffic-volume spike detection ──────────────────────── */
static int detect_spikes_omp(const float *X, int n_rows, int n_feat,
                               int bytes_col, double global_avg_bytes,
                               double spike_factor) {
    int spike_count = 0;
    #pragma omp parallel for schedule(static) reduction(+:spike_count)
    for (int i = 0; i < n_rows; i++) {
        if (bytes_col < 0 || bytes_col >= n_feat) continue;
        double val = X[(size_t)i * n_feat + bytes_col];
        if (val > spike_factor * global_avg_bytes)
            spike_count++;
    }
    return spike_count;
}

/* ── Accuracy metrics (serial, fast) ─────────────────────────────────────── */
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
    int nthreads           = (argc > 2) ? atoi(argv[2]) : 4;

    omp_set_num_threads(nthreads);
    printf("=== OpenMP (%d threads) ===\n", omp_get_max_threads());

    Dataset ds = load_bin(data_path);

    double *means = (double *)malloc(ds.n_feat * sizeof(double));
    double *stds  = (double *)malloc(ds.n_feat * sizeof(double));
    int    *flags = (int    *)malloc(ds.n_rows * sizeof(int));

    /* ── Timed region ──────────────────────────────────────────────────── */
    double t_start = omp_get_wtime();

    compute_stats_omp(ds.X, ds.n_rows, ds.n_feat, means, stds);
    int anomalies = detect_anomalies_omp(ds.X, ds.n_rows, ds.n_feat,
                                          means, stds, ZSCORE_THRESHOLD, flags);

    double elapsed = omp_get_wtime() - t_start;

    /* ── Results ───────────────────────────────────────────────────────── */
    printf("\n--- Results ---\n");
    printf("  Threads        : %d\n",     nthreads);
    printf("  Rows processed : %d\n",     ds.n_rows);
    printf("  Anomalies found: %d (%.2f%%)\n",
           anomalies, 100.0 * anomalies / ds.n_rows);
    printf("  Time           : %.6f s\n", elapsed);

    double rmse = compute_rmse(flags, ds.y, ds.n_rows);
    printf("  RMSE vs label  : %.8f\n", rmse);

    printf("\n--- Detection Metrics ---\n");
    print_metrics(flags, ds.y, ds.n_rows);

    /* ── Compare against serial flags for correctness ──────────────────── */
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

    save_flags(flags, ds.n_rows, "../data/omp_flags.bin");

    /* ── Thread scaling info ────────────────────────────────────────────── */
    printf("\n--- OpenMP Info ---\n");
    printf("  Max threads available: %d\n", omp_get_max_threads());
    #pragma omp parallel
    {
        #pragma omp single
        printf("  Threads in parallel region: %d\n", omp_get_num_threads());
    }

    free(means); free(stds); free(flags);
    free_dataset(&ds);
    return 0;
}
