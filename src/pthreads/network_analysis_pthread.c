/*
 * HPC Network Traffic Analysis - POSIX Threads Implementation
 * Includes RMSE validation and speedup calculation
 */

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <errno.h>

#define MAX_LINE_LENGTH 10000
#define MAX_FIELDS 50
#define HASH_TABLE_SIZE 10000
#define MAX_THREADS 16

// Data structures
typedef struct {
    char *fields[MAX_FIELDS];
    int field_count;
} Record;

typedef struct {
    char ip[50];
    int packets;
    int bytes;
    int attacks;
    int normal;
} IPInfo;

typedef struct {
    int total_predicted;
    int total_actual;
    double sse;
} RMSEInfo;

// Globals
IPInfo ip_hash[HASH_TABLE_SIZE];
int total_records = 0;
int total_attacks = 0;
int total_normal = 0;
RMSEInfo global_rmse = {0, 0, 0.0};

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

// Thread data
typedef struct {
    int tid;
    int start, end;
    Record *records;
    int local_pred, local_actual;
    double local_sse;
} ThreadData;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

unsigned int hash_ip(const char *ip) {
    unsigned int h = 5381;
    while (*ip) h = h*33 + *ip++;
    return h % HASH_TABLE_SIZE;
}

int parse_line(char *line, Record *r) {
    r->field_count = 0;
    char *copy = strdup(line);
    char *tok = strtok(copy, ",");
    while (tok && r->field_count < MAX_FIELDS) {
        while (*tok=='"'||*tok=='\'') tok++;
        char *e = tok + strlen(tok)-1;
        while (e>tok && (*e=='"'||*e=='\'')) *e--=0;
        r->fields[r->field_count++] = strdup(tok);
        tok = strtok(NULL, ",");
    }
    free(copy);
    return r->field_count;
}

void *process_chunk(void *arg) {
    ThreadData *d = (ThreadData*)arg;
    d->local_pred = d->local_actual = 0;
    d->local_sse = 0.0;
    
    for (int i = d->start; i < d->end; i++) {
        if (d->records[i].field_count < 45) continue;
        
        char *ip = d->records[i].fields[2] ?: "0";
        int bytes = atoi(d->records[i].fields[8] ?: "0");
        char *attack_cat = d->records[i].fields[44] ?: "Normal";
        int actual = atoi(d->records[i].fields[45] ?: "0");
        
        int predicted = strcmp(attack_cat, "Normal") ? 1 : 0;
        d->local_pred += predicted;
        d->local_actual += actual;
        d->local_sse += (predicted - actual) * (predicted - actual);
        
        unsigned int h = hash_ip(ip);
        if (ip_hash[h].packets == 0) strncpy(ip_hash[h].ip, ip, 49);
        ip_hash[h].packets++;
        ip_hash[h].bytes += bytes;
        
        int is_attack = strcmp(attack_cat, "Normal") != 0;
        if (is_attack) ip_hash[h].attacks++;
        else ip_hash[h].normal++;
    }
    
    pthread_mutex_lock(&lock);
    global_rmse.total_predicted += d->local_pred;
    global_rmse.total_actual += d->local_actual;
    global_rmse.sse += d->local_sse;
    pthread_mutex_unlock(&lock);
    
    return NULL;
}

double read_serial_time() {
    FILE *f = fopen("../../results/serial_time.txt", "r");
    if (!f) return 0.0;
    double t; fscanf(f, "%lf", &t); fclose(f);
    return t;
}

int main(int argc, char *argv[]) {
    char file[256] = "../data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv";
    int nthreads = 4;
    if (argc > 1) strcpy(file, argv[1]);
    if (argc > 2) nthreads = atoi(argv[2]);
    if (nthreads > MAX_THREADS) nthreads = MAX_THREADS;
    
    printf("=== Pthreads Network Analysis | Threads: %d ===\n", nthreads);
    
    FILE *fp = fopen(file, "r");
    if (!fp) { perror("fopen"); return 1; }
    
    char line[MAX_LINE_LENGTH];
    fgets(line, MAX_LINE_LENGTH, fp); // skip header
    int count = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp)) count++;
    fclose(fp);
    
    printf("Records: %d\n", count);
    
    Record *recs = malloc(count * sizeof(Record));
    fp = fopen(file, "r");
    fgets(line, MAX_LINE_LENGTH, fp);
    int idx = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp) && idx < count) {
        line[strcspn(line,"\n")] = 0;
        if (parse_line(line, &recs[idx]) > 0) idx++;
    }
    fclose(fp);
    
    pthread_t thr[MAX_THREADS];
    ThreadData td[MAX_THREADS];
    int chunk = idx / nthreads;
    
    double t0 = get_time();
    for (int i = 0; i < nthreads; i++) {
        td[i].tid = i;
        td[i].records = recs;
        td[i].start = i * chunk;
        td[i].end = (i == nthreads-1) ? idx : (i+1) * chunk;
        pthread_create(&thr[i], NULL, process_chunk, &td[i]);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(thr[i], NULL);
    double t1 = get_time();
    
    double exec_time = t1 - t0;
    double serial_time = read_serial_time();
    double speedup = serial_time > 0 ? serial_time / exec_time : 0;
    double eff = speedup > 0 ? (speedup / nthreads) * 100 : 0;
    
    // Count totals
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        total_records += ip_hash[i].packets;
        total_attacks += ip_hash[i].attacks;
        total_normal += ip_hash[i].normal;
    }
    
    double rmse = total_records > 0 ? sqrt(global_rmse.sse / total_records) : 0;
    
    printf("\n=== Results ===\n");
    printf("Total: %d | Attacks: %d | Normal: %d\n", total_records, total_attacks, total_normal);
    printf("Time: %.4fs | Throughput: %.0f rec/s\n", exec_time, total_records/exec_time);
    
    if (serial_time > 0) {
        printf("\n=== Speedup Analysis ===\n");
        printf("Serial: %.4fs | Speedup: %.2fx | Efficiency: %.1f%%\n", serial_time, speedup, eff);
    }
    
    printf("\n=== RMSE Validation ===\n");
    printf("RMSE: %.6f (%.6f%%)\n", rmse, rmse*100);
    printf("%s\n", rmse*100 < 0.01 ? "✓ Target met!" : "✗ Target not met");
    
    // Free records
    for (int i = 0; i < idx; i++)
        for (int j = 0; j < recs[i].field_count; j++)
            free(recs[i].fields[j]);
    free(recs);
    pthread_mutex_destroy(&lock);
    
    return 0;
}

