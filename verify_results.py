#!/usr/bin/env python3
"""
Verification script for HPC Network Analysis outputs.
Compares all parallel implementation outputs against the serial baseline.
"""

import os
import re
import glob
from collections import defaultdict

VERIFICATION_DIR = "results/verification"

def extract_metrics(log_path):
    """Extract key metrics from a log file."""
    metrics = {}
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return None, str(e)

    # Extract Accuracy
    m = re.search(r'Accuracy:\s+([\d.]+)%', content)
    metrics['accuracy'] = float(m.group(1)) if m else None

    # Extract Precision
    m = re.search(r'Precision:\s+([\d.]+)%', content)
    metrics['precision'] = float(m.group(1)) if m else None

    # Extract Recall
    m = re.search(r'Recall:\s+([\d.]+)%', content)
    metrics['recall'] = float(m.group(1)) if m else None

    # Extract F1
    m = re.search(r'F1 Score:\s+([\d.]+)%', content)
    metrics['f1'] = float(m.group(1)) if m else None

    # Extract RMSE
    m = re.search(r'RMSE:\s+([\d.]+)\s+\(', content)
    metrics['rmse'] = float(m.group(1)) if m else None

    # Extract Confusion Matrix values
    m = re.search(r'Normal\s+(\d+)\s+(\d+)', content)
    if m:
        metrics['tn'] = int(m.group(1))
        metrics['fp'] = int(m.group(2))

    m = re.search(r'Attack\s+(\d+)\s+(\d+)', content)
    if m:
        metrics['fn'] = int(m.group(1))
        metrics['tp'] = int(m.group(2))

    # Extract timing
    m = re.search(r'Single-pass time:\s+([\d.]+)s', content)
    metrics['single_time'] = float(m.group(1)) if m else None

    m = re.search(r'Total time \(x\d+\):\s+([\d.]+)s', content)
    metrics['total_time'] = float(m.group(1)) if m else None

    # Extract throughput
    m = re.search(r'Throughput:\s+([\d.]+)\s+rec/s', content)
    metrics['throughput'] = float(m.group(1)) if m else None

    return metrics, None

def compare_to_baseline(baseline, metrics):
    """Compare metrics against baseline. Returns (pass, discrepancies)."""
    if not baseline or not metrics:
        return False, ["Missing metrics"]

    discrepancies = []
    tolerance = 0.001  # Small floating point tolerance

    keys = ['accuracy', 'precision', 'recall', 'f1', 'rmse', 'tp', 'tn', 'fp', 'fn']
    for key in keys:
        if key not in baseline or baseline[key] is None:
            continue
        if key not in metrics or metrics[key] is None:
            discrepancies.append(f"{key}: missing in output")
            continue

        diff = abs(baseline[key] - metrics[key])
        if diff > tolerance:
            discrepancies.append(f"{key}: {metrics[key]} (expected {baseline[key]}, diff={diff:.6f})")

    return len(discrepancies) == 0, discrepancies

def main():
    if not os.path.isdir(VERIFICATION_DIR):
        print(f"ERROR: {VERIFICATION_DIR} does not exist.")
        return 1

    # Get baseline
    baseline_path = os.path.join(VERIFICATION_DIR, "serial.log")
    baseline, err = extract_metrics(baseline_path)
    if err or not baseline:
        print(f"ERROR: Could not read baseline from {baseline_path}: {err}")
        return 1

    print("=" * 100)
    print("HPC Network Analysis — Verification Report")
    print("=" * 100)
    print(f"\nBaseline (Serial) Metrics:")
    print(f"  Accuracy:  {baseline['accuracy']:.3f}%")
    print(f"  Precision: {baseline['precision']:.3f}%")
    print(f"  Recall:    {baseline['recall']:.3f}%")
    print(f"  F1 Score:  {baseline['f1']:.3f}%")
    print(f"  RMSE:      {baseline['rmse']:.6f}")
    print(f"  TP={baseline['tp']}, TN={baseline['tn']}, FP={baseline['fp']}, FN={baseline['fn']}")
    print(f"  Time:      {baseline['total_time']:.4f}s")
    print(f"  Throughput: {baseline['throughput']:.0f} rec/s")
    print("\n" + "=" * 100)
    print(f"{'Implementation':<25} {'Status':<8} {'Time(s)':<12} {'Speedup':<10} {'Discrepancies':<40}")
    print("=" * 100)

    # Process all log files
    log_files = sorted(glob.glob(os.path.join(VERIFICATION_DIR, "*.log")))

    results = []
    for log_path in log_files:
        fname = os.path.basename(log_path)
        if fname == "serial.log":
            continue  # Already printed above

        metrics, err = extract_metrics(log_path)
        if err:
            print(f"{fname:<25} {'FAIL':<8} {'N/A':<12} {'N/A':<10} {err:<40}")
            results.append((fname, False, err))
            continue

        passed, discrepancies = compare_to_baseline(baseline, metrics)
        status = "PASS" if passed else "FAIL"

        speedup = baseline['total_time'] / metrics['total_time'] if metrics.get('total_time') else 0
        time_str = f"{metrics['total_time']:.4f}" if metrics.get('total_time') else "N/A"
        speedup_str = f"{speedup:.2f}x" if metrics.get('total_time') else "N/A"

        disc_str = "; ".join(discrepancies[:2]) if discrepancies else "None"
        if len(disc_str) > 38:
            disc_str = disc_str[:35] + "..."

        print(f"{fname:<25} {status:<8} {time_str:<12} {speedup_str:<10} {disc_str:<40}")
        results.append((fname, passed, discrepancies))

    print("=" * 100)

    # Summary
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed

    print(f"\nSUMMARY: {passed}/{total} implementations PASSED correctness check.")
    if failed > 0:
        print("\nFAILED implementations:")
        for fname, p, discs in results:
            if not p:
                print(f"  - {fname}: {', '.join(discs)}")
        return 1
    else:
        print("\n✅ ALL IMPLEMENTATIONS PRODUCED CORRECT OUTPUTS!")
        return 0

if __name__ == "__main__":
    exit(main())
