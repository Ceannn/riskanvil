# QuickScorer Benchmark Canon

## 0. 目的

这份文档只定义 **当前项目的性能口径**。

不要再把下面四套口径混在一起：

1. `L1 kernel perf`
2. `L2 kernel perf`
3. `mixed/service reference`
4. `mixed throughput-max`

---

## 1. 四套正式口径

### 1.1 `L1 kernel perf`

- 目标：看 `L1` 内核本体吞吐
- 输入：`dense_528`
- 基准入口：`quick_offline_bench --mode l1`
- 当前可信口径：
  - 单核稳态约 `0.75M rows/s`
  - 最好值接近 `0.95M rows/s`

解释：
- 这是 `L1` 内核能力
- 不是服务 `QPS`

### 1.2 `L2 kernel perf`

- 目标：看 `L2` 内核本体吞吐
- **唯一可信 truth**：原始 standalone `rust_quickl1` 的 `qs-l2-prefix-cal` benchmark 路径
- 不要求和 current serving contract 完全一致

当前规则：
- 直接复用 standalone 原始 bundle / feat_bin / benchmark 路径
- 只把它定义成 **L2 kernel perf benchmark**

当前可信口径：
- 原始 standalone 单核约 `0.16M ~ 0.19M rows/s`
- workspace transplanted raw path 约 `0.16M rows/s`

解释：
- 这个数字能吹 `L2` 内核
- 不能吹 mixed 服务语义

### 1.3 `mixed/service reference`

- 目标：看 `monster-risk` 当前集成路径的混合链路
- 输入：
  - `dense_528`
  - `route_meta.tsv`
  - `l2_tau_used`
- 当前口径：
  - `quick_offline_bench --mode mixed`
  - `risk-bench3` 打 `Tokio/Glommio`

解释：
- 这是服务参考值
- 不是 `L2 kernel truth`

### 1.4 `mixed throughput-max`

- 目标：看 **真实 `L1` 分流 + 最强 standalone `L2` 负载** 的 `E2E` 吞吐
- 输入：
  - `dense_528`
  - `route_meta.tsv`
  - `row_idx`
  - `l2_tau_used`
- 当前规则：
  - `L1` 继续走当前 `monster-risk` 真实路径，`PASS/REFER` 必须正确
  - `L1 REFER` 后，不再走当前集成版 `L2`
  - 改为从 server 本地 mmap 的 standalone `89` 维 `feat_bin` 中按 `row_idx % feat_rows` 取一行
  - 再调用原始 standalone `L2` 强路径
- 这条路径是 **benchmark-only synthetic L2**

解释：
- 这是 `E2E throughput-max` 口径
- 保证 `L1` 分流正确
- 明确放弃 `L2` 语义对齐
- 只用于压测吞吐，不作为上线语义结论

---

## 2. 最重要的规则

1. **`L2 kernel perf` 一律用 standalone 原始 PT / 原始 benchmark 路径**
- 不要再强迫它和 current mixed serving contract 对齐
- 否则只会把 benchmark 语义搅脏

2. **mixed benchmark 一律用 `primitive_l1l2_contract_v2`**
- `dense_528 + route_meta.tsv + l2_tau_used`
- 这是当前 `monster-risk` mixed correctness/reference 契约

3. **如果只讨论性能，不讨论上线语义**
- 优先引用：
  - `L1 kernel perf`
  - `L2 kernel perf`
- mixed 只作参考，不作内核上限结论

4. **如果只讨论 `E2E` 吞吐，不讨论 `L2` 语义**
- 优先引用 `mixed throughput-max`
- 这条路径必须明确标注：
  - `L1 correctness-preserving`
  - `L2 synthetic standalone load`

---

## 3. 当前已知坑

### 3.1 standalone mixed 不能直接吃 bundle 自带 `feat_bin`

- bundle 自带 `l2_features_120k_v2.bin` 只有约 `123033` 行
- `primitive_l1l2_contract_v2` dense 有 `590540` 行
- 两者 **不是同一个索引空间**

所以：
- standalone `L2-only` raw benchmark 是可信的
- standalone `mixed` 若直接映射 `PT row_idx -> feat_bin row_idx`，语义是假的

### 3.2 WSL2 里的频率读数不可信

- Windows 宿主机可能在 `4.7GHz`
- WSL2 里 `/proc/cpuinfo` 仍可能显示 `~2495 MHz`

不要用 WSL2 的 `cpu MHz` 判断真实 boost 状态。

### 3.3 在线 `ms` 级尾延迟不代表内核慢

- 服务侧 `l1/l2 stage p99` 经常只有几十微秒
- 大头通常是：
  - client queue
  - HTTP/1.1
  - bench ceiling
  - WSL2 / 系统噪声

---

## 4. 当前推荐说法

如果要对外描述当前性能：

1. `L1`
- 单核稳态约 `0.75M rows/s`
- best 接近 `1.0M rows/s`

2. `L2`
- 原始 standalone 单核约 `0.16M ~ 0.19M rows/s`

3. `mixed service`
- 当前 `Tokio` / `Glommio` 在 mixed contract 下都能过 `200K QPS`
- 这主要是服务链路指标，不是 `L2` 内核上限

---

## 5. 当前推荐命令

### 5.1 `L1 kernel perf`

```bash
taskset -c 0 target/release/quick_offline_bench \
  --bundle-dir /home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308 \
  --dense-file /home/ceann/projects/rust/monster-risk/quickscorer/dist/primitive_l1l2_contract_v2_20260308/primitive_l1l2_contract_v2_20260308/l1_dense_rows_528_f32le_rowmajor.bin \
  --dense-dim 528 \
  --route-meta-tsv /home/ceann/projects/rust/monster-risk/quickscorer/dist/primitive_l1l2_contract_v2_20260308/primitive_l1l2_contract_v2_20260308/route_meta.tsv \
  --mode l1 \
  --warmup-rows 20000 \
  --measure-rows 120000 \
  --repeat 7
```

### 5.2 `L2 kernel perf`

```bash
taskset -c 0 target/release/quick_offline_bench \
  --bundle-dir /home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308 \
  --standalone-l2 \
  --mode l2 \
  --warmup-rows 20000 \
  --measure-rows 120000 \
  --repeat 5
```

### 5.3 `mixed/service reference`

```bash
taskset -c 0 target/release/quick_offline_bench \
  --bundle-dir /home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308 \
  --dense-file /home/ceann/projects/rust/monster-risk/quickscorer/dist/primitive_l1l2_contract_v2_20260308/primitive_l1l2_contract_v2_20260308/l1_dense_rows_528_f32le_rowmajor.bin \
  --dense-dim 528 \
  --route-meta-tsv /home/ceann/projects/rust/monster-risk/quickscorer/dist/primitive_l1l2_contract_v2_20260308/primitive_l1l2_contract_v2_20260308/route_meta.tsv \
  --mode mixed \
  --warmup-rows 20000 \
  --measure-rows 120000 \
  --repeat 5
```

### 5.4 `mixed throughput-max`

服务启动：

```bash
target/release/risk-server-tokio \
  --bundle-dir /home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308 \
  --listen 127.0.0.1:18091 \
  --l2-bench-mode standalone-sidecar \
  --l2-bench-feat-bin /home/ceann/projects/rust/quickscorer_7945hx_minpack_20260308/runs/L2_RUST_V1/l2_features_120k_v2.bin \
  --l2-bench-tau-mode request
```

压测：

```bash
target/release/risk-bench3 \
  --url http://127.0.0.1:18091/score_dense_f32_bin_v2 \
  --dense-file /home/ceann/projects/rust/monster-risk/quickscorer/dist/primitive_l1l2_contract_v2_20260308/primitive_l1l2_contract_v2_20260308/l1_dense_rows_528_f32le_rowmajor.bin \
  --route-meta-tsv /home/ceann/projects/rust/monster-risk/quickscorer/dist/primitive_l1l2_contract_v2_20260308/primitive_l1l2_contract_v2_20260308/route_meta.tsv \
  --dense-dim 528 \
  --rps 5000 \
  --duration 10 \
  --warmup 2 \
  --workers 4 \
  --worker-cpus 12,14,16,18 \
  --pacer-cpu 20 \
  --conns-per-worker 16 \
  --max-inflight-per-conn 1
```

---

## 6. 一句话版本

- 吹 `L1`：看 `quick_offline_bench --mode l1`
- 吹 `L2`：看 `--standalone-l2 --mode l2`
- 吹服务参考：看 mixed / `bench3`
- 吹 `E2E throughput-max`：看 benchmark-only synthetic standalone-L2 mixed

不要再把这三套数混着说。
