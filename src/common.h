/**
 * EC7207 HPC Project - Group 12
 * common.h  –  Shared data structures and utilities for all C implementations
 *
 * Binary file format (produced by prepare_data.py):
 *   int32  n_rows
 *   int32  n_feat
 *   float32[n_rows * n_feat]   X  (row-major)
 *   int32[n_rows]              y  (labels: 0=Benign, 1=Attack)
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/* ── Dataset struct ──────────────────────────────────────────────────────── */
typedef struct {
    int    n_rows;
    int    n_feat;
    float *X;      /* [n_rows * n_feat] row-major */
    int   *y;      /* [n_rows] ground-truth labels */
} Dataset;

/* ── Load binary dataset from file ──────────────────────────────────────── */
static inline Dataset load_bin(const char *path) {
    Dataset d = {0, 0, NULL, NULL};
    FILE *f = fopen(path, "rb");
    if (!f) { perror("load_bin: fopen"); exit(EXIT_FAILURE); }

    if (fread(&d.n_rows, sizeof(int), 1, f) != 1 ||
        fread(&d.n_feat, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "load_bin: failed to read header\n"); exit(1);
    }

    d.X = (float *)malloc((size_t)d.n_rows * d.n_feat * sizeof(float));
    d.y = (int   *)malloc((size_t)d.n_rows           * sizeof(int));
    if (!d.X || !d.y) { fprintf(stderr, "load_bin: malloc failed\n"); exit(1); }

    fread(d.X, sizeof(float), (size_t)d.n_rows * d.n_feat, f);
    fread(d.y, sizeof(int),   (size_t)d.n_rows,             f);
    fclose(f);

    printf("[load_bin] %s -> rows=%d  feat=%d\n", path, d.n_rows, d.n_feat);
    return d;
}

/* ── Free dataset ────────────────────────────────────────────────────────── */
static inline void free_dataset(Dataset *d) {
    free(d->X); d->X = NULL;
    free(d->y); d->y = NULL;
    d->n_rows = d->n_feat = 0;
}

/* ── Compute per-feature mean and std (serial, used for stats init) ──────── */
static inline void compute_stats_serial(const float *X, int n_rows, int n_feat,
                                         double *means, double *stds) {
    memset(means, 0, n_feat * sizeof(double));
    memset(stds,  0, n_feat * sizeof(double));

    for (int f = 0; f < n_feat; f++) {
        double s = 0.0;
        for (int i = 0; i < n_rows; i++)
            s += X[(size_t)i * n_feat + f];
        means[f] = s / n_rows;
    }
    for (int f = 0; f < n_feat; f++) {
        double s = 0.0;
        for (int i = 0; i < n_rows; i++) {
            double d = X[(size_t)i * n_feat + f] - means[f];
            s += d * d;
        }
        stds[f] = sqrt(s / n_rows);
    }
}

/* ── RMSE between predicted flags and true labels ────────────────────────── */
static inline double compute_rmse(const int *pred, const int *truth, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)pred[i] - (double)truth[i];
        s += d * d;
    }
    return sqrt(s / n);
}

/* ── Save flag array to binary file ─────────────────────────────────────── */
static inline void save_flags(const int *flags, int n, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("save_flags: fopen"); return; }
    fwrite(flags, sizeof(int), n, f);
    fclose(f);
    printf("[save_flags] %d flags -> %s\n", n, path);
}

/* ── Wall-clock timer (seconds) ─────────────────────────────────────────── */
static inline double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Anomaly threshold ───────────────────────────────────────────────────── */
#define ZSCORE_THRESHOLD 3.0

#endif /* COMMON_H */
