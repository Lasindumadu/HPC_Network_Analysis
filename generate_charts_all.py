#!/usr/bin/env python3
"""
generate_charts_all.py
EC7207 — HPC Network Traffic Analysis

Convenience wrapper that runs all three chart generators in sequence.

  generate_charts.py   → charts/speedup.png ... all_charts.png
                         (CUDA: real log if available, else simulated from serial baseline)

  generate_charts2.py  → charts/speedup_v2.png ... all_charts_v2.png
                         (CUDA: hardcoded Tesla T4 values from Google Colab)

  generate_charts3.py  → charts/speedup_v3.png ... all_charts_v3.png
                         (CPU-only — no CUDA bars)

Usage:
    python3 generate_charts_all.py           # run all three
    python3 generate_charts_all.py --v1      # only generate_charts.py
    python3 generate_charts_all.py --v2      # only generate_charts2.py
    python3 generate_charts_all.py --v3      # only generate_charts3.py
    python3 generate_charts_all.py --v2 --v3 # pick any combination
"""

import sys
import os
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    '--v1': ('generate_charts.py',  'v1 (auto CUDA)'),
    '--v2': ('generate_charts2.py', 'v2 (Tesla T4)'),
    '--v3': ('generate_charts3.py', 'v3 (CPU-only)'),
}

# Default: run all three
flags = [f for f in sys.argv[1:] if f in SCRIPTS]
if not flags:
    flags = list(SCRIPTS.keys())

print("=" * 60)
print("  EC7207 — Generate All Performance Charts")
print("=" * 60)
print()

total_start = time.time()
generated = []

for flag in flags:
    script, label = SCRIPTS[flag]
    script_path = os.path.join(BASE, script)

    if not os.path.exists(script_path):
        print(f"  [skip] {script} not found")
        continue

    print(f"── Running {label}: {script} ──")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE,
    )

    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  Done in {elapsed:.1f}s\n")
        generated.append(label)
    else:
        print(f"  [error] {script} exited with code {result.returncode}\n")

total_elapsed = time.time() - total_start

print("=" * 60)
if generated:
    print(f"  Generated: {', '.join(generated)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Output directory: charts/")
    print()

    # List all generated PNG files
    charts_dir = os.path.join(BASE, 'charts')
    if os.path.isdir(charts_dir):
        pngs = sorted(f for f in os.listdir(charts_dir) if f.endswith('.png'))
        if pngs:
            print("  Charts:")
            for f in pngs:
                size_kb = os.path.getsize(os.path.join(charts_dir, f)) // 1024
                print(f"    charts/{f}  ({size_kb} KB)")
else:
    print("  No charts generated.")
print("=" * 60)
