/**
 * EC7207 HPC Project - Group 12
 * mpi_analysis.c  –  MPI distributed-memory parallel implementation
 *
 * Distributes row chunks across MPI processes using MPI_Scatterv.
 * Each rank independently detects anomalies in its chunk.
 * Results are gathered back to rank 0 with MPI_Gatherv.
 *
 * Compile:  mpicc -O2 -o mpi mpi_analysis.c -lm
 * Run:      mpirun -np 4 ./mpi ../data/train_data.bin
 */

#include "common.h"
#include <mpi.h>

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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ── Variables declared on all ranks ───────────────────────────────── */
    int n_rows = 0, n_feat = 0;
    float  *X_all   = NULL;   /* only rank 0 fills this */
    int    *y_all   = NULL;   /* only rank 0 fills this */
    double *means   = NULL;
    double *stds    = NULL;

    /* ── Rank 0: load data and compute global stats ──────────────────────*/
    if (rank == 0) {
        const char *path = (argc > 1) ? argv[1] : "../data/train_data.bin";
        printf("=== MPI (%d ranks) ===\n", size);
        Dataset ds = load_bin(path);
        n_rows = ds.n_rows;
        n_feat = ds.n_feat;
        X_all  = ds.X;
        y_all  = ds.y;

        means = (double *)malloc(n_feat * sizeof(double));
        stds  = (double *)malloc(n_feat * sizeof(double));
        compute_stats_serial(X_all, n_rows, n_feat, means, stds);
        printf("[rank 0] Stats computed.\n");
    }

    /* ── Broadcast dimensions ───────────────────────────────────────────── */
    MPI_Bcast(&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_feat, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ── Broadcast stats to all ranks ──────────────────────────────────── */
    if (rank != 0) {
        means = (double *)malloc(n_feat * sizeof(double));
        stds  = (double *)malloc(n_feat * sizeof(double));
    }
    MPI_Bcast(means, n_feat, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(stds,  n_feat, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── Calculate per-rank chunk sizes (handles uneven division) ────────*/
    int base  = n_rows / size;
    int extra = n_rows % size;
    int local_n = base + (rank < extra ? 1 : 0);

    /* Build sendcounts / displacements arrays (elements = rows * n_feat) */
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    int offset = 0;
    for (int r = 0; r < size; r++) {
        int rows_r     = base + (r < extra ? 1 : 0);
        sendcounts[r]  = rows_r * n_feat;
        displs[r]      = offset;
        offset        += sendcounts[r];
    }

    /* Flag gatherv arrays (per-row, not per-element) */
    int *flag_counts  = (int *)malloc(size * sizeof(int));
    int *flag_displs  = (int *)malloc(size * sizeof(int));
    int f_off = 0;
    for (int r = 0; r < size; r++) {
        flag_counts[r]  = base + (r < extra ? 1 : 0);
        flag_displs[r]  = f_off;
        f_off          += flag_counts[r];
    }

    /* ── Scatter X rows to each rank ─────────────────────────────────────*/
    float *X_local = (float *)malloc((size_t)local_n * n_feat * sizeof(float));
    MPI_Scatterv(X_all, sendcounts, displs, MPI_FLOAT,
                 X_local, local_n * n_feat, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    /* ── Each rank detects anomalies on its local chunk ──────────────────*/
    MPI_Barrier(MPI_COMM_WORLD);   /* sync before timing */
    double t_start = MPI_Wtime();

    int *flags_local = (int *)malloc(local_n * sizeof(int));
    int  local_count = 0;
    for (int i = 0; i < local_n; i++) {
        flags_local[i] = 0;
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

    /* ── Gather per-rank anomaly counts ─────────────────────────────────── */
    int total_anomalies = 0;
    MPI_Reduce(&local_count, &total_anomalies, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    /* ── Gather all flag arrays back to rank 0 ───────────────────────────*/
    int *flags_all = NULL;
    if (rank == 0)
        flags_all = (int *)malloc(n_rows * sizeof(int));

    MPI_Gatherv(flags_local, local_n, MPI_INT,
                flags_all, flag_counts, flag_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    /* ── Rank 0: print results ───────────────────────────────────────────*/
    if (rank == 0) {
        printf("\n--- Results ---\n");
        printf("  MPI ranks      : %d\n",     size);
        printf("  Rows processed : %d\n",     n_rows);
        printf("  Anomalies found: %d (%.2f%%)\n",
               total_anomalies, 100.0 * total_anomalies / n_rows);
        printf("  Time           : %.6f s\n", elapsed);

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

        /* Per-rank row distribution */
        printf("\n--- Rank Distribution ---\n");
        for (int r = 0; r < size; r++)
            printf("  Rank %2d: %d rows\n", r, flag_counts[r]);

        save_flags(flags_all, n_rows, "../data/mpi_flags.bin");
        free(flags_all);
        free(X_all); free(y_all);
    }

    /* ── Max time across all ranks (for accurate reporting) ────────────── */
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("  Max rank time  : %.6f s\n", max_time);

    free(X_local); free(flags_local);
    free(means); free(stds);
    free(sendcounts); free(displs);
    free(flag_counts); free(flag_displs);
    MPI_Finalize();
    return 0;
}
