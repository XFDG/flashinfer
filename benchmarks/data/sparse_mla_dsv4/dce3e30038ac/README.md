# Sparse MLA DSV4 direct validation on GB300

All 94 direct cases pass strict correctness with zero fallback and baseline/CAKE speedup above one. Summed CUPTI GPU active time is **1.736584 → 1.500323 ms (1.157473424×)**; minimum row speedup is **1.012062499×**. The exact 94 semantic/shape cases are retained, with BF16 atol=rtol=0.01 and FP8 atol=rtol=0.1.

Prepared source commit: `dce3e30038acf697bf832bcd7a0ba688d4f2d351`. Source tree: `fe01b04f07719e9d4830c70256afee87fa45bdbd`. Parent: `495b528f58bff1ac44790555a2b2b82d92d5cfe7`. The fresh width-policy TP4 pair below supplies current end-to-end evidence; the earlier passing A and rejected B remain recorded.

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

The tree readback must be `fe01b04f07719e9d4830c70256afee87fa45bdbd`. The pytest command checks correctness and does not repeat the performance measurements. The retained public tests passed 100/100; separate affected-route synccheck/racecheck reported zero errors and zero hazards/errors/warnings. The fresh width-policy TP4 pair passes below; historical results remain separately recorded.

## Fresh TP4 pair with the width-640 policy

Fresh real SGLang DeepSeek-V4-Flash TP4 A: **130570.767202148 → 120242.376625977 ms (1.085896427×)**. Counterbalanced B: **126203.928368164 → 125388.164241211 ms (1.006505910×)**. The canonical pair passes. Each repeat has **512 exact output tokens across eight requests**, **24 exact ordered route records per side**, **eight traces** and **zero fallback**.

A measured/physical duration: **3179.785394688 / 3231.103028100 s**; B: **3072.548526656 / 3132.407019200 s**. Combined measured runtime is **6252.333921344 s** and physical span is **6524.988085600 s**. These elapsed durations are distinct from GPU active milliseconds.

Scores merge overlapping CUDA kernel intervals across all four devices, including NCCL. Launch gaps and overlap double counting are excluded. Wall time and individual kernel-duration sums do not determine the performance score.

The TP4 integration is a separate standalone-export policy: width640 selects the existing BF16 H64 fixed_q producer and reducer; widths132,136 and160 retain serial2. Public FlashInfer source and direct94 dispatch are unchanged. Public and standalone wrapper/device-source forms have distinct byte identities. This public evidence bundle does not include the separate TP4 integration capsule, exact collector/runtime bindings, model environment or raw traces; its checksum and correctness commands do not rerun TP4 GPU measurements.

`tp4-w640-policy.json`, `tp4-w640-repeat-A.json`, `tp4-w640-repeat-B.json` and `tp4-w640-pair.json` record the current policy and full-precision evidence. The original A and rejected B JSON files remain byte-identical historical records; their failure/hold fields describe the earlier configuration.

## Historical SGLang TP4 repeat A

Fresh repeat A for DeepSeek-V4-Flash on four GB300 GPUs passes: **142,518.727782 → 130,994.435459 ms (1.087975434×)**. All **512 output tokens across eight requests** match exactly; all **24 ordered route records per side** match, with **eight traces** and **zero fallback**. The matching historical B was rejected and publication was held for that configuration.

Times merge overlapping CUDA kernel intervals across all four devices, including NCCL kernels. Launch gaps and overlap double counting are excluded; neither elapsed wall time nor the sum of individual kernel durations is used for scoring. Startup, JIT, warmup, correctness, shutdown and export phases are excluded. Measured runtime was **3,162.423183 s**, allocation runtime **3,196.337725 s**, and physical turnaround **3,196.441646 s**.

`tp4-repeat-A.json` retains the full-precision comparison, token and trace hashes, route validation, environment identities and source association. TP4 exercises the authenticated retained BF16 H64 source families through standalone exports; their public wrapper forms are distinct. The ragged H64 SWA and FP8 H128 changes are outside these fixed-query TP4 inputs. The measured public source association remains `dce3e30038acf697bf832bcd7a0ba688d4f2d351`, tree `fe01b04f07719e9d4830c70256afee87fa45bdbd`.

The existing checksum command covers this result file. It verifies recorded evidence and does not rerun TP4 or reduce raw traces; those traces and a GPU benchmark driver are outside this bundle.

## Historical rejected SGLang TP4 repeat B

Fresh repeat B measures **130,914.574901 → 131,352.260374 ms (0.996667850×)**, a **437.685473 ms increase** in GPU active-union time. All **512 output tokens across eight requests** match exactly; all **24 ordered route records per side** match, with **eight traces** and **zero fallback**. The strict finalizer rejected the performance result. No accepted B or combined pair qualification was produced for that historical configuration, and publication was held. Repeat A remains a passing individual result.

The metric uses the same cross-device CUDA kernel active-union calculation as repeat A, including NCCL and excluding launch gaps and overlap double counting. Measured runtime was **2,822.044950 s**, allocation runtime **2,873.216167 s**, and physical turnaround **2,873.318190 s**. These elapsed durations are not performance scores.

`tp4-repeat-B-rejected.json` records the rejected comparison, correctness predicates and unchanged source association. Its result was reduced from the retained traces after the strict finalizer rejected performance; it is not a passing final receipt. The checksum command verifies this recorded evidence and launches no GPU workload.
