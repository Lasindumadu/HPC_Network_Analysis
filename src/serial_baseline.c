/**
 * EC7207 HPC Project - Group 12
 * serial_baseline.c  –  Serial reference implementation
 *
 * Computes per-feature statistics (mean, std) and detects anomalies
 * using Z-score thresholding. Saves timing and flag output for RMSE
 * comparison against all parallel implementations.
 *
 * Compile:  gcc -O2 -o serial serial_baseline.c -lm
 * Run:      ./serial ../data/train_data.bin
 */

#include "common.h"

/* ── Compute per-feature mean and std ────────────────────────────────────── */
static void compute_stats(const float *X, int n_rows, int n_feat,
                           double *means, double *stds) {
    /* Pass 1: mean */
    for (int f = 0; f < n_feat; f++) {
        double sum = 0.0;
        for (int i = 0; i < n_rows; i++)
            sum += X[(size_t)i * n_feat + f];
        means[f] = sum / n_rows;
    }
    /* Pass 2: standard deviation */
    for (int f = 0; f < n_feat; f++) {
        double sum = 0.0;
        for (int i = 0; i < n_rows; i++) {
            double diff = X[(size_t)i * n_feat + f] - means[f];
            sum += diff * diff;
        }
        stds[f] = sqrt(sum / n_rows);
    }
}

/* ── Z-score anomaly detection ───────────────────────────────────────────── */
static int detect_anomalies(const float *X, int n_rows, int n_feat,
                              const double *means, const double *stds,
                              double threshold, int *flags) {
    int anomaly_count = 0;
    for (int i = 0; i < n_rows; i++) {
        flags[i] = 0;
        for (int f = 0; f < n_feat; f++) {
            if (stds[f] < 1e-9) continue;   /* skip zero-variance features */
            double z = fabs(((double)X[(size_t)i * n_feat + f] - means[f]) / stds[f]);
            if (z > threshold) {
                flags[i] = 1;
                break;   /* one suspicious feature is enough */
            }
        }
        anomaly_count += flags[i];
    }
    return anomaly_count;
}

/* ── Traffic-volume spike detection (per 1000-row window) ────────────────── */
static void detect_volume_spikes(const float *X, int n_rows, int n_feat,
                                   int bytes_col, int pkts_col,
                                   int window, int *spike_windows) {
    int n_windows = (n_rows + window - 1) / window;
    double total_bytes = 0.0, total_pkts = 0.0;
    int valid = 0;

    /* Compute global averages */
    for (int i = 0; i < n_rows; i++) {
        if (bytes_col >= 0) total_bytes += X[(size_t)i * n_feat + bytes_col];
        if (pkts_col  >= 0) total_pkts  += X[(size_t)i * n_feat + pkts_col];
        valid++;
    }
    double avg_bytes = (valid > 0 && bytes_col >= 0) ? total_bytes / valid : 0;
    double avg_pkts  = (valid > 0 && pkts_col  >= 0) ? total_pkts  / valid : 0;

    /* Flag windows where average exceeds 3x global average */
    for (int w = 0; w < n_windows; w++) {
        int start = w * window;
        int end   = (start + window < n_rows) ? start + window : n_rows;
        double wb = 0.0, wp = 0.0;
        for (int i = start; i < end; i++) {
            if (bytes_col >= 0) wb += X[(size_t)i * n_feat + bytes_col];
            if (pkts_col  >= 0) wp += X[(size_t)i * n_feat + pkts_col];
        }
        int cnt = end - start;
        spike_windows[w] = ((avg_bytes > 0 && (wb / cnt) > 3.0 * avg_bytes) ||
                            (avg_pkts  > 0 && (wp / cnt) > 3.0 * avg_pkts)) ? 1 : 0;
    }
}

/* ── Accuracy, precision, recall ─────────────────────────────────────────── */
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

    printf("=== Serial Baseline ===\n");
    Dataset ds = load_bin(data_path);

    double *means = (double *)malloc(ds.n_feat * sizeof(double));
    double *stds  = (double *)malloc(ds.n_feat * sizeof(double));
    int    *flags = (int    *)malloc(ds.n_rows * sizeof(int));
    if (!means || !stds || !flags) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* ── Time the core computation ─────────────────────────────────────── */
    double t_start = wall_time();

    compute_stats(ds.X, ds.n_rows, ds.n_feat, means, stds);
    int anomalies = detect_anomalies(ds.X, ds.n_rows, ds.n_feat,
                                     means, stds, ZSCORE_THRESHOLD, flags);

    double elapsed = wall_time() - t_start;

    /* ── Results ───────────────────────────────────────────────────────── */
    printf("\n--- Results ---\n");
    printf("  Rows processed : %d\n",     ds.n_rows);
    printf("  Features       : %d\n",     ds.n_feat);
    printf("  Anomalies found: %d (%.2f%%)\n",
           anomalies, 100.0 * anomalies / ds.n_rows);
    printf("  Time           : %.6f s\n", elapsed);

    double rmse = compute_rmse(flags, ds.y, ds.n_rows);
    printf("  RMSE vs label  : %.8f\n", rmse);

    printf("\n--- Detection Metrics ---\n");
    print_metrics(flags, ds.y, ds.n_rows);

    /* ── Save flags as ground truth ────────────────────────────────────── */
    save_flags(flags, ds.n_rows, "../data/serial_flags.bin");

    /* ── Print feature stats summary (first 5 features) ───────────────── */
    printf("\n--- Feature Stats (first 5) ---\n");
    int show = ds.n_feat < 5 ? ds.n_feat : 5;
    for (int f = 0; f < show; f++)
        printf("  feat[%d]: mean=%.4f  std=%.4f\n", f, means[f], stds[f]);

    free(means); free(stds); free(flags);
    free_dataset(&ds);
    return 0;
}
