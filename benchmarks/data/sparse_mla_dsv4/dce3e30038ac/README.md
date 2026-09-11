# Sparse MLA DSV4 direct validation on GB300

All 94 direct cases pass strict correctness with zero fallback and baseline/CAKE speedup above one. Summed CUPTI GPU active time is **1.736584 → 1.500323 ms (1.157473424×)**; minimum row speedup is **1.012062499×**. The exact 94 semantic/shape cases are retained, with BF16 atol=rtol=0.01 and FP8 atol=rtol=0.1.

Prepared source commit: `dce3e30038acf697bf832bcd7a0ba688d4f2d351`. Source tree: `fe01b04f07719e9d4830c70256afee87fa45bdbd`. Parent: `495b528f58bff1ac44790555a2b2b82d92d5cfe7`. Public source publication remains held while real SGLang DeepSeek-V4-Flash TP4 validation is completed.

The measurement used the explicit public CAKE backend and the current18 TRTLLM-Gen baseline on NVIDIA GB300, SM103a, with an aarch64 host, Python 3.12.3, PyTorch 2.13.0+cu130 and PyTorch CUDA 13.0. Times are CUPTI GPU kernel active-union, with launch gaps excluded. Benchmark runtime was **55.183819 s**, helper duration **64.835800 s**, and physical turnaround **76.874102 s**. These elapsed durations are distinct from GPU active milliseconds.

`direct94-table.md` contains the complete comparison. `direct94-evidence.json` contains the exact public source identity, public case parameters, raw-precision row medians/ratios, correctness/tolerance fields, four compiled module hashes and environment. `direct94-rows.csv` provides the same row comparisons for analysis. `public-source.json` maps the prepared commit and its complete changed-file Git blob identities. The exported parameter projection retains execution semantics and removes only a non-execution gate label.

To reproduce and verify the comparison table from this measured evidence bundle:

```bash
python3 verify-direct-evidence.py direct94-evidence.json --table direct94-table.md
sha256sum -c SHA256SUMS
```

These are evidence readback commands and launch no GPU kernels. This bundle verifies the reported measurements and does not include the 94-case GPU benchmark driver. To inspect the exact source and run the existing correctness tests after source delivery, run from a FlashInfer checkout:

```bash
git checkout --detach dce3e30038acf697bf832bcd7a0ba688d4f2d351
git rev-parse HEAD^{tree}
python3 -m pytest tests/mla/test_cake_dsv4.py -q
```

The tree readback must be `fe01b04f07719e9d4830c70256afee87fa45bdbd`. The pytest command checks correctness and does not repeat the performance measurements. The retained public tests passed 100/100; separate affected-route synccheck/racecheck reported zero errors and zero hazards/errors/warnings. This bundle reports direct kernel results; final TP4 comparisons will be reported separately with their own source/device association.
