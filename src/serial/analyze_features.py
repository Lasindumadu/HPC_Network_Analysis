#!/usr/bin/env python3
"""
UNSW-NB15 Feature Analysis Script
Run this to find exactly which feature values separate attacks from normal.
Output guides threshold tuning in the C detector.

Usage:
  python3 analyze_features.py /path/to/UNSW_NB15_training-set.csv
"""
import sys, csv, math
from collections import defaultdict

FILE = sys.argv[1] if len(sys.argv) > 1 else \
    "data/UNSW_NB15_training-set.csv/UNSW_NB15_training-set.csv"

def safe_float(v):
    try: return float(v)
    except: return 0.0

def safe_int(v):
    try: return int(float(v))
    except: return 0

print(f"Analyzing: {FILE}\n")

rows = []
with open(FILE, newline='', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

total   = len(rows)
attacks = [r for r in rows if safe_int(r.get('label','0')) == 1]
normals = [r for r in rows if safe_int(r.get('label','0')) == 0]

print(f"Total: {total}  |  Attacks: {len(attacks)}  |  Normal: {len(normals)}\n")
print("="*65)

# ── 1. State distribution ────────────────────────────────────────
print("\n[1] STATE distribution")
print(f"{'State':<10} {'Att':>8} {'Norm':>8} {'Att%':>8} {'Norm%':>8}  Verdict")
print("-"*65)
state_att  = defaultdict(int)
state_norm = defaultdict(int)
for r in attacks: state_att[r.get('state','?')] += 1
for r in normals: state_norm[r.get('state','?')] += 1
all_states = sorted(set(list(state_att)+list(state_norm)),
                    key=lambda s: -(state_att[s]+state_norm[s]))
for s in all_states[:12]:
    a = state_att[s]; n = state_norm[s]; tot = a+n
    ap = 100*a/tot if tot else 0
    np_ = 100*n/tot if tot else 0
    verdict = "ATTACK" if ap > 75 else ("NORMAL" if np_ > 75 else "mixed")
    print(f"{s:<10} {a:>8} {n:>8} {ap:>7.1f}% {np_:>7.1f}%  {verdict}")

# ── 2. sttl (source TTL) distribution ───────────────────────────
print("\n[2] STTL (source TTL) distribution")
print(f"{'sttl':<8} {'Att':>8} {'Norm':>8} {'Att%':>8}  Verdict")
print("-"*50)
sttl_att  = defaultdict(int)
sttl_norm = defaultdict(int)
for r in attacks: sttl_att[safe_int(r.get('sttl',0))] += 1
for r in normals: sttl_norm[safe_int(r.get('sttl',0))] += 1
all_sttl = sorted(set(list(sttl_att)+list(sttl_norm)),
                  key=lambda v: -(sttl_att[v]+sttl_norm[v]))
for v in all_sttl[:15]:
    a = sttl_att[v]; n = sttl_norm[v]; tot = a+n
    ap = 100*a/tot if tot else 0
    verdict = "ATTACK" if ap>80 else ("NORMAL" if ap<20 else "mixed")
    print(f"{v:<8} {a:>8} {n:>8} {ap:>7.1f}%  {verdict}")

# ── 3. dttl (dest TTL) distribution ─────────────────────────────
print("\n[3] DTTL (dest TTL) distribution")
print(f"{'dttl':<8} {'Att':>8} {'Norm':>8} {'Att%':>8}  Verdict")
print("-"*50)
dttl_att  = defaultdict(int)
dttl_norm = defaultdict(int)
for r in attacks: dttl_att[safe_int(r.get('dttl',0))] += 1
for r in normals: dttl_norm[safe_int(r.get('dttl',0))] += 1
for v in sorted(set(list(dttl_att)+list(dttl_norm)),
                key=lambda v: -(dttl_att[v]+dttl_norm[v]))[:12]:
    a = dttl_att[v]; n = dttl_norm[v]; tot = a+n
    ap = 100*a/tot if tot else 0
    verdict = "ATTACK" if ap>80 else ("NORMAL" if ap<20 else "mixed")
    print(f"{v:<8} {a:>8} {n:>8} {ap:>7.1f}%  {verdict}")

# ── 4. ct_state_ttl distribution ────────────────────────────────
print("\n[4] CT_STATE_TTL distribution")
print(f"{'val':<6} {'Att':>8} {'Norm':>8} {'Att%':>8}  Verdict")
print("-"*50)
cs_att  = defaultdict(int)
cs_norm = defaultdict(int)
for r in attacks: cs_att[safe_int(r.get('ct_state_ttl',0))] += 1
for r in normals: cs_norm[safe_int(r.get('ct_state_ttl',0))] += 1
for v in sorted(set(list(cs_att)+list(cs_norm))):
    a = cs_att[v]; n = cs_norm[v]; tot = a+n
    ap = 100*a/tot if tot else 0
    verdict = "ATTACK" if ap>80 else ("NORMAL" if ap<20 else "mixed")
    print(f"{v:<6} {a:>8} {n:>8} {ap:>7.1f}%  {verdict}")

# ── 5. dpkts=0 analysis (one-way traffic) ───────────────────────
print("\n[5] DPKTS=0 (one-way / no response)")
dp0_att  = sum(1 for r in attacks if safe_int(r.get('dpkts',0)) == 0)
dp0_norm = sum(1 for r in normals if safe_int(r.get('dpkts',0)) == 0)
print(f"  dpkts=0 in attacks: {dp0_att} ({100*dp0_att/len(attacks):.1f}%)")
print(f"  dpkts=0 in normals: {dp0_norm} ({100*dp0_norm/len(normals):.1f}%)")

# ── 6. rate percentile analysis ─────────────────────────────────
print("\n[6] RATE percentiles")
att_rates  = sorted([safe_float(r.get('rate',0)) for r in attacks])
norm_rates = sorted([safe_float(r.get('rate',0)) for r in normals])
for pct in [50, 75, 90, 95, 99]:
    ai = int(pct/100 * len(att_rates))
    ni = int(pct/100 * len(norm_rates))
    print(f"  p{pct:>2}: attacks={att_rates[ai]:>12.1f}  normals={norm_rates[ni]:>12.1f}")

# ── 7. spkts percentile analysis ────────────────────────────────
print("\n[7] SPKTS percentiles")
att_sp  = sorted([safe_int(r.get('spkts',0)) for r in attacks])
norm_sp = sorted([safe_int(r.get('spkts',0)) for r in normals])
for pct in [50, 75, 90, 95, 99]:
    ai = int(pct/100 * len(att_sp))
    ni = int(pct/100 * len(norm_sp))
    print(f"  p{pct:>2}: attacks={att_sp[ai]:>8}  normals={norm_sp[ni]:>8}")

# ── 8. ct_srv_src percentiles ────────────────────────────────────
print("\n[8] CT_SRV_SRC percentiles")
att_cs  = sorted([safe_int(r.get('ct_srv_src',0)) for r in attacks])
norm_cs = sorted([safe_int(r.get('ct_srv_src',0)) for r in normals])
for pct in [50, 75, 90, 95, 99]:
    ai = int(pct/100 * len(att_cs))
    ni = int(pct/100 * len(norm_cs))
    print(f"  p{pct:>2}: attacks={att_cs[ai]:>8}  normals={norm_cs[ni]:>8}")

# ── 9. ct_src_dport_ltm percentiles ─────────────────────────────
print("\n[9] CT_SRC_DPORT_LTM percentiles")
att_dp  = sorted([safe_int(r.get('ct_src_dport_ltm',0)) for r in attacks])
norm_dp = sorted([safe_int(r.get('ct_src_dport_ltm',0)) for r in normals])
for pct in [50, 75, 90, 95, 99]:
    ai = int(pct/100 * len(att_dp))
    ni = int(pct/100 * len(norm_dp))
    print(f"  p{pct:>2}: attacks={att_dp[ai]:>8}  normals={norm_dp[ni]:>8}")

# ── 10. sload percentiles ────────────────────────────────────────
print("\n[10] SLOAD percentiles")
att_sl  = sorted([safe_float(r.get('sload',0)) for r in attacks])
norm_sl = sorted([safe_float(r.get('sload',0)) for r in normals])
for pct in [50, 75, 90, 95, 99]:
    ai = int(pct/100 * len(att_sl))
    ni = int(pct/100 * len(norm_sl))
    print(f"  p{pct:>2}: attacks={att_sl[ai]:>14.1f}  normals={norm_sl[ni]:>14.1f}")

# ── 11. proto distribution ───────────────────────────────────────
print("\n[11] PROTO attack rate (top 8)")
print(f"{'proto':<12} {'Att':>8} {'Norm':>8} {'Att%':>8}")
print("-"*45)
proto_att  = defaultdict(int)
proto_norm = defaultdict(int)
for r in attacks: proto_att[r.get('proto','?')] += 1
for r in normals: proto_norm[r.get('proto','?')] += 1
all_proto = sorted(set(list(proto_att)+list(proto_norm)),
                   key=lambda p:-(proto_att[p]+proto_norm[p]))
for p in all_proto[:8]:
    a = proto_att[p]; n = proto_norm[p]; tot = a+n
    print(f"{p:<12} {a:>8} {n:>8} {100*a/tot if tot else 0:>7.1f}%")

# ── 12. service distribution ─────────────────────────────────────
print("\n[12] SERVICE attack rate (top 10)")
print(f"{'service':<15} {'Att':>8} {'Norm':>8} {'Att%':>8}  Verdict")
print("-"*55)
svc_att  = defaultdict(int)
svc_norm = defaultdict(int)
for r in attacks: svc_att[r.get('service','-')] += 1
for r in normals: svc_norm[r.get('service','-')] += 1
all_svc = sorted(set(list(svc_att)+list(svc_norm)),
                 key=lambda s:-(svc_att[s]+svc_norm[s]))
for s in all_svc[:10]:
    a = svc_att[s]; n = svc_norm[s]; tot = a+n
    ap = 100*a/tot if tot else 0
    verdict = "ATTACK" if ap>75 else ("NORMAL" if ap<25 else "mixed")
    print(f"{s:<15} {a:>8} {n:>8} {ap:>7.1f}%  {verdict}")

print("\n" + "="*65)
print("ANALYSIS COMPLETE — use these distributions to tune thresholds")
print("="*65)