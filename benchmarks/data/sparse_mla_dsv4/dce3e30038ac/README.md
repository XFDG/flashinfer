# Sparse MLA DSV4 direct validation on GB300

All 94 direct cases pass strict correctness with zero fallback and baseline/CAKE speedup above one. Summed CUPTI GPU active time is **1.736584 → 1.500323 ms (1.157473424×)**; minimum row speedup is **1.012062499×**. The exact 94 semantic/shape cases are retained, with BF16 atol=rtol=0.01 and FP8 atol=rtol=0.1.

Prepared source commit: `dce3e30038acf697bf832bcd7a0ba688d4f2d351`. Source tree: `fe01b04f07719e9d4830c70256afee87fa45bdbd`. Parent: `495b528f58bff1ac44790555a2b2b82d92d5cfe7`. Public source publication remains held because real SGLang DeepSeek-V4-Flash TP4 repeat B regresses.

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

The tree readback must be `fe01b04f07719e9d4830c70256afee87fa45bdbd`. The pytest command checks correctness and does not repeat the performance measurements. The retained public tests passed 100/100; separate affected-route synccheck/racecheck reported zero errors and zero hazards/errors/warnings. Fresh TP4 repeat A is recorded below. Repeat B regresses and overall TP4 qualification fails the strict performance gate.

## Real SGLang TP4 repeat A

Fresh repeat A for DeepSeek-V4-Flash on four GB300 GPUs passes: **142,518.727782 → 130,994.435459 ms (1.087975434×)**. All **512 output tokens across eight requests** match exactly; all **24 ordered route records per side** match, with **eight traces** and **zero fallback**. Repeat B regresses, overall TP4 qualification fails the strict performance gate, and public source publication remains held.

Times merge overlapping CUDA kernel intervals across all four devices, including NCCL kernels. Launch gaps and overlap double counting are excluded; neither elapsed wall time nor the sum of individual kernel durations is used for scoring. Startup, JIT, warmup, correctness, shutdown and export phases are excluded. Measured runtime was **3,162.423183 s**, allocation runtime **3,196.337725 s**, and physical turnaround **3,196.441646 s**.

`tp4-repeat-A.json` retains the full-precision comparison, token and trace hashes, route validation, environment identities and source association. TP4 exercises the authenticated retained BF16 H64 source families through standalone exports; their public wrapper forms are distinct. The ragged H64 SWA and FP8 H128 changes are outside these fixed-query TP4 inputs. The measured public source association remains `dce3e30038acf697bf832bcd7a0ba688d4f2d351`, tree `fe01b04f07719e9d4830c70256afee87fa45bdbd`.

The existing checksum command covers this result file. It verifies recorded evidence and does not rerun TP4 or reduce raw traces; those traces and a GPU benchmark driver are outside this bundle.

## Rejected real SGLang TP4 repeat B

Fresh repeat B measures **130,914.574901 → 131,352.260374 ms (0.996667850×)**, a **437.685473 ms increase** in GPU active-union time. All **512 output tokens across eight requests** match exactly; all **24 ordered route records per side** match, with **eight traces** and **zero fallback**. The strict finalizer rejected the performance result. No accepted B or combined pair qualification was produced, and source publication remains held. Repeat A remains a passing individual result.

The metric uses the same cross-device CUDA kernel active-union calculation as repeat A, including NCCL and excluding launch gaps and overlap double counting. Measured runtime was **2,822.044950 s**, allocation runtime **2,873.216167 s**, and physical turnaround **2,873.318190 s**. These elapsed durations are not performance scores.

`tp4-repeat-B-rejected.json` records the rejected comparison, correctness predicates and unchanged source association. Its result was reduced from the retained traces after the strict finalizer rejected performance; it is not a passing final receipt. The checksum command verifies this recorded evidence and launches no GPU workload.
