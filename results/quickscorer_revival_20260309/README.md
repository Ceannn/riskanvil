# QuickScorer Revival Snapshot

Date: 2026-03-09

## Scope
This snapshot freezes the recovered benchmark canon for the toy-project throughput path:

1. `L1` correctness-preserving
2. `L2` kernel perf from original standalone `rust_quickl1`
3. benchmark-only synthetic standalone-`L2` E2E path for throughput-max

## Saved artifacts
- `standalone_l2_battery_summary.json`
  - source: original standalone `qs-l2-prefix-cal`
  - single-core, battery-power run
- `bench3_tokio_synth_l2_5k_summary.json`
  - source: `risk-server-tokio` with `--l2-bench-mode standalone-sidecar`
- `bench3_tokio_synth_l2_5k_window.csv`
  - per-window E2E smoke results

## Canonical numbers in this snapshot
### L2 kernel perf (original standalone)
- median: `85,453.94 rows/s`
- mean: `85,687.79 rows/s`
- best: `101,707.74 rows/s`
- worst: `70,590.61 rows/s`
- rss_peak_mb: `~188.5`

### E2E throughput-max smoke
- attempted_rps: `4976.3`
- ok_rps: `4976.3`
- 2xx: `14929`
- 429/5xx/timeout: `0`
- used_l2: `1732`
- used_l2 ratio: `~11.6%`
- stage p99(us):
  - router: `28`
  - l1: `27`
  - l2: `23`
  - serialize: `9`

## Important contract note
This is benchmark-only:
- `L1` PASS/REFER remains real
- `L2` semantics are intentionally not aligned
- `REFER` requests use standalone sidecar `89`-dim rows via local mmap
- this path is for throughput only, not serving-truth correctness

## Commands
### L2 kernel perf
Run from:
`/home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308`

```bash
taskset -c 0 ./rust_quickl1/target/release/rust_quickl1 qs-l2-prefix-cal \
  --bundle-manifest runs/L2_Q2_4500_PREFIX_CAL_V5_ATLAS_20260307/atlas/manifest.json \
  --threads 1 \
  --chunk-rows 128 \
  --parallel-min-rows 4096 \
  --max-rows 120000 \
  --stats-json /tmp/standalone_l2_battery_20260309/run_1.json \
  --out-tsv /tmp/standalone_l2_battery_20260309/run_1.tsv
```

### E2E throughput-max
Start server from repo root:

```bash
target/release/risk-server-tokio \
  --bundle-dir /home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308 \
  --listen 127.0.0.1:18091 \
  --l2-bench-mode standalone-sidecar \
  --l2-bench-feat-bin /home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308/runs/L2_RUST_V1/l2_features_120k_v2.bin \
  --l2-bench-tau-mode request
```

Then run:

```bash
target/release/risk-bench3 \
  --url http://127.0.0.1:18091/score_dense_f32_bin_v2 \
  --dense-file quickscorer/dist/primitive_l1l2_contract_v2_20260308/primitive_l1l2_contract_v2_20260308/l1_dense_rows_528_f32le_rowmajor.bin \
  --route-meta-tsv quickscorer/dist/primitive_l1l2_contract_v2_20260308/primitive_l1l2_contract_v2_20260308/route_meta.tsv \
  --dense-dim 528 \
  --rps 5000 \
  --duration 3 \
  --warmup 1 \
  --workers 4 \
  --worker-cpus 12,14,16,18 \
  --pacer-cpu 20 \
  --conns-per-worker 16 \
  --max-inflight-per-conn 1 \
  --summary-json results/quickscorer_revival_20260309/bench3_tokio_synth_l2_5k_summary.json \
  --window-csv results/quickscorer_revival_20260309/bench3_tokio_synth_l2_5k_window.csv
```
