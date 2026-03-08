/**
 * EC7207 HPC Project - Group 12
 * hybrid_mpi_omp.c  –  Hybrid MPI + OpenMP implementation
 *
 * MPI distributes row chunks across processes (nodes / ranks).
 * OpenMP parallelises computation within each MPI process using CPU threads.
 * This is the most practical HPC approach for multi-node clusters.
 *
 * Usage: each MPI rank uses OMP_NUM_THREADS threads.
 *        Total parallelism = n_ranks × n_threads_per_rank
 *
 * Compile:  mpicc -O2 -fopenmp -o hybrid hybrid_mpi_omp.c -lm
 * Run:      OMP_NUM_THREADS=4 mpirun -np 2 ./hybrid ../data/train_data.bin
 */

#include "common.h"
#include <mpi.h>
#include <omp.h>

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
    /*
     * MPI_THREAD_FUNNELED: only the main thread makes MPI calls.
     * Use MPI_THREAD_MULTIPLE if OpenMP threads need to call MPI.
     */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "WARNING: MPI does not support MPI_THREAD_FUNNELED\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nthreads = omp_get_max_threads();

    int n_rows = 0, n_feat = 0;
    float  *X_all   = NULL;
    int    *y_all   = NULL;
    double *means   = NULL;
    double *stds    = NULL;

    /* ── Rank 0: load data ──────────────────────────────────────────────── */
    if (rank == 0) {
        const char *path = (argc > 1) ? argv[1] : "../data/train_data.bin";
        printf("=== Hybrid MPI+OpenMP (%d ranks x %d threads = %d total) ===\n",
               size, nthreads, size * nthreads);
        Dataset ds = load_bin(path);
        n_rows = ds.n_rows;
        n_feat = ds.n_feat;
        X_all  = ds.X;
        y_all  = ds.y;

        means = (double *)malloc(n_feat * sizeof(double));
        stds  = (double *)malloc(n_feat * sizeof(double));

        /* Compute stats in parallel on rank 0 using OpenMP */
        #pragma omp parallel for schedule(static)
        for (int f = 0; f < n_feat; f++) {
            double s = 0.0;
            for (int i = 0; i < n_rows; i++)
                s += X_all[(size_t)i * n_feat + f];
            means[f] = s / n_rows;
        }
        #pragma omp parallel for schedule(static)
        for (int f = 0; f < n_feat; f++) {
            double s = 0.0;
            for (int i = 0; i < n_rows; i++) {
                double d = X_all[(size_t)i * n_feat + f] - means[f];
                s += d * d;
            }
            stds[f] = sqrt(s / n_rows);
        }
        printf("[rank 0] Stats computed with %d OpenMP threads.\n", nthreads);
    }

    /* ── Broadcast dimensions and stats ────────────────────────────────── */
    MPI_Bcast(&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_feat, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        means = (double *)malloc(n_feat * sizeof(double));
        stds  = (double *)malloc(n_feat * sizeof(double));
    }
    MPI_Bcast(means, n_feat, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(stds,  n_feat, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── Build scatter/gather arrays ────────────────────────────────────── */
    int base  = n_rows / size;
    int extra = n_rows % size;
    int local_n = base + (rank < extra ? 1 : 0);

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    int *flag_counts = (int *)malloc(size * sizeof(int));
    int *flag_displs = (int *)malloc(size * sizeof(int));

    int offset = 0, f_off = 0;
    for (int r = 0; r < size; r++) {
        int rows_r      = base + (r < extra ? 1 : 0);
        sendcounts[r]   = rows_r * n_feat;
        displs[r]       = offset;
        flag_counts[r]  = rows_r;
        flag_displs[r]  = f_off;
        offset         += sendcounts[r];
        f_off          += rows_r;
    }

    /* ── Scatter X rows ─────────────────────────────────────────────────── */
    float *X_local = (float *)malloc((size_t)local_n * n_feat * sizeof(float));
    MPI_Scatterv(X_all, sendcounts, displs, MPI_FLOAT,
                 X_local, local_n * n_feat, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    /* ── Hybrid: OpenMP inside MPI – anomaly detection ───────────────────
     *
     * Each MPI rank uses all available OpenMP threads to process its
     * assigned rows. This is the key hybrid parallelism pattern:
     *   - Inter-node: MPI
     *   - Intra-node: OpenMP
     */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int *flags_local = (int *)calloc(local_n, sizeof(int));
    int  local_count = 0;

    #pragma omp parallel for schedule(dynamic, 256) reduction(+:local_count)
    for (int i = 0; i < local_n; i++) {
        for (int f = 0; f < n_feat; f++) {
            if (stds[f] < 1e-9) continue;
            double z = fabs(((double)X_local[(size_t)i * n_feat + f]
                             - means[f]) / stds[f]);
            if (z > ZSCORE_THRESHOLD) { flags_local[i] = 1; break; }
        }
        local_count += flags_local[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t_start;

    /* ── Reduce total and gather flags ──────────────────────────────────── */
    int total_anomalies = 0;
    MPI_Reduce(&local_count, &total_anomalies, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int *flags_all = NULL;
    if (rank == 0) flags_all = (int *)malloc(n_rows * sizeof(int));
    MPI_Gatherv(flags_local, local_n, MPI_INT,
                flags_all, flag_counts, flag_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Max time across all ranks */
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ── Rank 0: results ────────────────────────────────────────────────── */
    if (rank == 0) {
        printf("\n--- Results ---\n");
        printf("  MPI ranks      : %d\n",     size);
        printf("  Threads/rank   : %d\n",     nthreads);
        printf("  Total parallelism: %d\n",   size * nthreads);
        printf("  Rows processed : %d\n",     n_rows);
        printf("  Anomalies found: %d (%.2f%%)\n",
               total_anomalies, 100.0 * total_anomalies / n_rows);
        printf("  Time (max rank): %.6f s\n", max_time);

        double rmse = compute_rmse(flags_all, y_all, n_rows);
        printf("  RMSE vs label  : %.8f\n", rmse);

        printf("\n--- Detection Metrics ---\n");
        print_metrics(flags_all, y_all, n_rows);

        FILE *sf = fopen("../data/serial_flags.bin", "rb");
        if (sf) {
            int *serial_flags = (int *)malloc(n_rows * sizeof(int));
            fread(serial_flags, sizeof(int), n_rows, sf);
            fclose(sf);
            double diff = compute_rmse(flags_all, serial_flags, n_rows);
            printf("  RMSE vs serial : %.8f  (%s)\n", diff,
                   diff < 1e-6 ? "PASS" : "MISMATCH");
            free(serial_flags);
        }

        /* Per-rank breakdown */
        printf("\n--- Rank Distribution ---\n");
        for (int r = 0; r < size; r++)
            printf("  Rank %2d: %d rows\n", r, flag_counts[r]);

        save_flags(flags_all, n_rows, "../data/hybrid_flags.bin");
        free(flags_all);
        free(X_all); free(y_all);
    }

    free(X_local); free(flags_local);
    free(means); free(stds);
    free(sendcounts); free(displs);
    free(flag_counts); free(flag_displs);
    MPI_Finalize();
    return 0;
}
