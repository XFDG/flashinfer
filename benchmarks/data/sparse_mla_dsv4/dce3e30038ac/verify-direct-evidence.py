#!/usr/bin/env python3
"""Reproduce the published 94-row table from the measured direct evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('evidence', type=Path)
p.add_argument('--table', type=Path, required=True)
a = p.parse_args()
e = json.loads(a.evidence.read_text())
assert e['schema'] == 'sparse-mla-direct94-public-evidence-v1'
rows = e['rows']
assert len(rows) == e['row_count'] == 94
assert [r['index'] for r in rows] == list(range(94))
assert e['correct_rows'] == e['strict_wins'] == 94 and e['fallback_count'] == 0
header = ['| # | Shape | DType | H | Profile | Baseline GPU active ms | CAKE GPU active ms | Speedup |', '|---:|---|---|---:|---|---:|---:|---:|']
lines = list(header)
for r in rows:
    assert r['correct'] and r['fallback_count'] == 0
    tolerance = 0.01 if r['params']['dtype'] == 'bfloat16' else 0.1
    assert r['atol'] == r['rtol'] == tolerance
    assert r['kernel_ms_source'] == 'active_union_ms' and r['launch_gaps_excluded']
    assert r['baseline_ms'] > r['candidate_ms'] > 0 and r['speedup'] > 1
    assert math.isclose(r['baseline_ms']/r['candidate_ms'], r['speedup'], rel_tol=1e-12)
    ratio = f"{r['speedup']:.4f}".rstrip('0').rstrip('.') + 'x'
    lines.append(f"| {r['index']+1} | {r['label']} | {r['params']['dtype']} | {r['params']['num_heads']} | {r['params']['profile']} | {r['baseline_ms']:.6f} | {r['candidate_ms']:.6f} | {ratio} |")
for name, expected in [('baseline_active_ms',sum(r['baseline_ms'] for r in rows)),('candidate_active_ms',sum(r['candidate_ms'] for r in rows)),('minimum_speedup',min(r['speedup'] for r in rows))]:
    assert math.isclose(e[name],expected,rel_tol=1e-12)
assert math.isclose(e['baseline_active_ms']/e['candidate_active_ms'],e['aggregate_speedup'],rel_tol=1e-12)
expected = ('\n'.join(lines)+'\n').encode()
assert a.table.read_bytes() == expected
assert hashlib.sha256(expected).hexdigest() == e['table_sha256']
print(f"94 correct, zero fallback, 94 strict wins; {e['baseline_active_ms']:.6f} -> {e['candidate_active_ms']:.6f} ms; {e['aggregate_speedup']:.9f}x; minimum {e['minimum_speedup']:.9f}x")
