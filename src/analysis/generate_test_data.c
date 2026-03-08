/*
 * Test Data Generator for UNSW-NB15 Format
 * Generates sample network traffic data for testing HPC implementations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_RECORDS 100000
#define NUM_IPS 1000
#define NUM_SERVICES 10
#define NUM_ATTACK_TYPES 10

const char *services[] = {"http", "https", "ftp", "ssh", "dns", "smtp", "pop3", "imap", "telnet", "none"};
const char *attacks[] = {"Normal", "Fuzzers", "Analysis", "Backdoors", "DoS", "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"};
const char *protocols[] = {"tcp", "udp", "icmp", "arp"};

char ips[NUM_IPS][50];

void generate_ips() {
    for (int i = 0; i < NUM_IPS; i++) {
        sprintf(ips[i], "192.168.%d.%d", (i / 256) % 256, i % 256);
    }
}

void generate_record(FILE *fp, int idx) {
    const char *service = services[rand() % NUM_SERVICES];
    const char *attack = attacks[rand() % NUM_ATTACK_TYPES];
    const char *protocol = protocols[rand() % 4];
    
    int srcip_idx = rand() % NUM_IPS;
    int dstip_idx = rand() % NUM_IPS;
    int srcport = (rand() % 65535);
    int dstport = (rand() % 65535);
    int duration = rand() % 1000;
    int srcbytes = rand() % 100000;
    int dstbytes = rand() % 100000;
    int packets = rand() % 1000;
    int rate = rand() % 10000;
    int sttl = rand() % 255;
    int dttl = rand() % 255;
    int syn = rand() % 2;
    int ack = rand() % 2;
    int psh = rand() % 2;
    int rst = rand() % 2;
    int fin = rand() % 2;
    
    fprintf(fp, "%d,%s,%s,%d,%d,%s,0,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,0,0,0,0,0,0,0,0,0,0,0,0,%s,0,0,0,0,0,0,%s\n",
            idx + 1,
            ips[srcip_idx],
            ips[dstip_idx],
            srcport,
            dstport,
            protocol,
            duration,
            srcbytes,
            dstbytes,
            packets,
            rate,
            sttl,
            dttl,
            syn,
            ack,
            psh,
            rst,
            fin,
            service,
            attack
    );
}

int main() {
    srand(time(NULL));
    generate_ips();
    
    FILE *fp = fopen("../../data/UNSW_NB15_training-set.csv", "w");
    if (!fp) {
        printf("Error creating file\n");
        return 1;
    }
    
    fprintf(fp, "sid,src_ip,dst_ip,src_port,dst_port,protocol,flags,duration,src_bytes,dst_bytes,packets,rate,sTTL,dTTL,syn,ack,psh,rst,fin\n");
    
    printf("Generating %d records...\n", NUM_RECORDS);
    
    for (int i = 0; i < NUM_RECORDS; i++) {
        generate_record(fp, i);
        if ((i + 1) % 10000 == 0) {
            printf("Generated %d records...\n", i + 1);
        }
    }
    
    fclose(fp);
    printf("Done! Generated %d records in data/UNSW_NB15_training-set.csv\n", NUM_RECORDS);
    
    return 0;
}

