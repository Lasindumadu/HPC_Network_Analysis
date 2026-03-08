/**
 * EC7207 HPC Project - Group 12
 * cuda_analysis.cu  –  CUDA GPU parallel implementation
 *
 * Each CUDA thread processes one row independently.
 * Uses shared memory for feature stats to reduce global memory access.
 * cudaEvent used for accurate kernel timing.
 *
 * Compile:  nvcc -O2 -arch=sm_75 -o cuda cuda_analysis.cu -lm
 *           (change sm_75 to match your GPU; T4=sm_75, A100=sm_80, V100=sm_70)
 * Run:      ./cuda ../data/train_data.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "common.h"

/* ── CUDA error checking macro ───────────────────────────────────────────── */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

/* ── Kernel: one thread per row ──────────────────────────────────────────── */
/*
 * d_means and d_stds are loaded into shared memory once per block,
 * then all threads in the block reuse the cached values — much faster
 * than every thread reading from global memory independently.
 *
 * MAX_FEAT must be >= n_feat. Increase if you have more features.
 */
#define MAX_FEAT 64

__global__ void anomaly_kernel(const float  *__restrict__ d_X,
                                const double *__restrict__ d_means,
                                const double *__restrict__ d_stds,
                                int          *__restrict__ d_flags,
                                int n_rows, int n_feat, double threshold) {
    /* Cache stats in shared memory */
    __shared__ double s_means[MAX_FEAT];
    __shared__ double s_stds[MAX_FEAT];

    /* First n_feat threads in the block load shared memory */
    if (threadIdx.x < n_feat) {
        s_means[threadIdx.x] = d_means[threadIdx.x];
        s_stds[threadIdx.x]  = d_stds[threadIdx.x];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;

    d_flags[i] = 0;
    for (int f = 0; f < n_feat; f++) {
        if (s_stds[f] < 1e-9) continue;
        double val = (double)d_X[(size_t)i * n_feat + f];
        double z   = fabs((val - s_means[f]) / s_stds[f]);
        if (z > threshold) { d_flags[i] = 1; break; }
    }
}

/* ── Kernel: parallel per-feature mean reduction ─────────────────────────── */
__global__ void mean_kernel(const float *__restrict__ d_X,
                             double      *__restrict__ d_means,
                             int n_rows, int n_feat) {
    /* Each block handles one feature; threads reduce over rows */
    int f = blockIdx.x;
    if (f >= n_feat) return;

    extern __shared__ double sdata[];
    int tid  = threadIdx.x;
    int step = blockDim.x;

    double sum = 0.0;
    for (int i = tid; i < n_rows; i += step)
        sum += (double)d_X[(size_t)i * n_feat + f];
    sdata[tid] = sum;
    __syncthreads();

    /* Parallel reduction within block */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_means[f] = sdata[0] / n_rows;
}

/* ── Kernel: parallel per-feature std reduction ──────────────────────────── */
__global__ void std_kernel(const float  *__restrict__ d_X,
                            const double *__restrict__ d_means,
                            double       *__restrict__ d_stds,
                            int n_rows, int n_feat) {
    int f = blockIdx.x;
    if (f >= n_feat) return;

    extern __shared__ double sdata[];
    int tid  = threadIdx.x;
    int step = blockDim.x;

    double sum = 0.0;
    double m   = d_means[f];
    for (int i = tid; i < n_rows; i += step) {
        double d = (double)d_X[(size_t)i * n_feat + f] - m;
        sum += d * d;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_stds[f] = sqrt(sdata[0] / n_rows);
}

/* ── Accuracy metrics (CPU) ──────────────────────────────────────────────── */
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

    /* GPU info */
    int dev = 0;
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("=== CUDA GPU ===\n");
    printf("  Device: %s\n", prop.name);
    printf("  SM count: %d  |  Warp size: %d  |  Global mem: %.1f GB\n",
           prop.multiProcessorCount, prop.warpSize,
           (double)prop.totalGlobalMem / 1e9);

    Dataset ds = load_bin(data_path);

    if (ds.n_feat > MAX_FEAT) {
        fprintf(stderr, "ERROR: n_feat=%d exceeds MAX_FEAT=%d. "
                "Increase MAX_FEAT in cuda_analysis.cu\n", ds.n_feat, MAX_FEAT);
        exit(1);
    }

    size_t data_sz  = (size_t)ds.n_rows * ds.n_feat * sizeof(float);
    size_t feat_sz  = (size_t)ds.n_feat * sizeof(double);
    size_t flags_sz = (size_t)ds.n_rows * sizeof(int);

    /* ── Allocate GPU memory ───────────────────────────────────────────── */
    float  *d_X;
    double *d_means, *d_stds;
    int    *d_flags;
    CUDA_CHECK(cudaMalloc(&d_X,     data_sz));
    CUDA_CHECK(cudaMalloc(&d_means, feat_sz));
    CUDA_CHECK(cudaMalloc(&d_stds,  feat_sz));
    CUDA_CHECK(cudaMalloc(&d_flags, flags_sz));

    /* ── Host→Device transfer ──────────────────────────────────────────── */
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(d_X, ds.X, data_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float h2d_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_start, ev_stop));

    /* ── Phase 1: GPU stats kernels ────────────────────────────────────── */
    int STATS_THREADS = 256;
    size_t smem = STATS_THREADS * sizeof(double);

    CUDA_CHECK(cudaEventRecord(ev_start));
    mean_kernel<<<ds.n_feat, STATS_THREADS, smem>>>(d_X, d_means, ds.n_rows, ds.n_feat);
    CUDA_CHECK(cudaGetLastError());
    std_kernel <<<ds.n_feat, STATS_THREADS, smem>>>(d_X, d_means, d_stds, ds.n_rows, ds.n_feat);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float stats_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&stats_ms, ev_start, ev_stop));

    /* ── Phase 2: Anomaly detection kernel ─────────────────────────────── */
    int BLOCK = 256;
    int GRID  = (ds.n_rows + BLOCK - 1) / BLOCK;

    CUDA_CHECK(cudaEventRecord(ev_start));
    anomaly_kernel<<<GRID, BLOCK>>>(d_X, d_means, d_stds, d_flags,
                                    ds.n_rows, ds.n_feat, ZSCORE_THRESHOLD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));

    /* ── Device→Host transfer ──────────────────────────────────────────── */
    int *flags = (int *)malloc(flags_sz);
    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(flags, d_flags, flags_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float d2h_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_start, ev_stop));

    /* ── Copy back stats for display ────────────────────────────────────── */
    double *means = (double *)malloc(feat_sz);
    double *stds  = (double *)malloc(feat_sz);
    CUDA_CHECK(cudaMemcpy(means, d_means, feat_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(stds,  d_stds,  feat_sz, cudaMemcpyDeviceToHost));

    /* ── Count anomalies ────────────────────────────────────────────────── */
    int count = 0;
    for (int i = 0; i < ds.n_rows; i++) count += flags[i];

    float total_ms = h2d_ms + stats_ms + kernel_ms + d2h_ms;

    printf("\n--- Results ---\n");
    printf("  Rows processed    : %d\n",     ds.n_rows);
    printf("  Anomalies found   : %d (%.2f%%)\n",
           count, 100.0 * count / ds.n_rows);
    printf("  H2D transfer      : %.2f ms\n", h2d_ms);
    printf("  Stats kernels     : %.2f ms\n", stats_ms);
    printf("  Anomaly kernel    : %.2f ms\n", kernel_ms);
    printf("  D2H transfer      : %.2f ms\n", d2h_ms);
    printf("  Total GPU time    : %.2f ms  (%.4f s)\n", total_ms, total_ms / 1000.0);
    printf("  Grid: %d blocks x %d threads = %d threads\n",
           GRID, BLOCK, GRID * BLOCK);

    double rmse = compute_rmse(flags, ds.y, ds.n_rows);
    printf("  RMSE vs label     : %.8f\n", rmse);

    printf("\n--- Detection Metrics ---\n");
    print_metrics(flags, ds.y, ds.n_rows);

    FILE *sf = fopen("../data/serial_flags.bin", "rb");
    if (sf) {
        int *serial_flags = (int *)malloc(ds.n_rows * sizeof(int));
        fread(serial_flags, sizeof(int), ds.n_rows, sf);
        fclose(sf);
        double diff = compute_rmse(flags, serial_flags, ds.n_rows);
        printf("  RMSE vs serial    : %.8f  (%s)\n", diff,
               diff < 1e-6 ? "PASS" : "MISMATCH");
        free(serial_flags);
    }

    save_flags(flags, ds.n_rows, "../data/cuda_flags.bin");

    /* ── Cleanup ────────────────────────────────────────────────────────── */
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_means));
    CUDA_CHECK(cudaFree(d_stds));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    free(flags); free(means); free(stds);
    free_dataset(&ds);
    return 0;
}
