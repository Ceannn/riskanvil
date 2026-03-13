# Riskanvil

Riskanvil is a Rust workspace for high-throughput tree-model inference and batch serving.

This repository was built around a simple requirement: if the model is large, the runtime still has to move. It keeps the inference runtime, two server implementations, and the benchmark client needed to push the whole stack instead of stopping at toy examples.

## Performance Snapshot

Representative results from this line of work:

- L1 runtime: about `503K rows/s` single-core median
- standalone L2 runtime: about `621K rows/s` single-core median
- end-to-end HTTP batch path: pushed past `400K rows/s`
- clean `420K rows/s` pass observed on the optimized batch-serving line

These numbers come from actual serving-oriented optimization work, not from a tiny isolated kernel demo that ignores the rest of the system.

## Highlights

- QuickScorer-oriented L1 runtime in Rust
- standalone L2 runtime and kernel-facing tooling
- Tokio and Glommio HTTP servers
- throughput-oriented benchmark client for HTTP batch serving
- end-to-end path built around dense `f32` batch scoring
- tuned for large-tree scoring workloads rather than framework aesthetics

## Workspace

The public workspace contains six crates:

- `crates/risk-core`
  - shared schema, pipeline, routing, and server-facing scoring integration
- `crates/risk-quickscorer`
  - QuickScorer-oriented L1 runtime and execution support
- `crates/risk-quickscorer-standalone-l2`
  - standalone L2 runtime and kernel-adjacent tooling
- `crates/risk-server-tokio`
  - Tokio-based HTTP server
- `crates/risk-server-glommio`
  - Glommio-based HTTP server
- `crates/risk-bench3`
  - high-rate throughput benchmark client

## Main End-to-End Path

The main throughput path in this repository is:

- server: `risk-server-tokio`
- client: `risk-bench3`
- transport: `HTTP/1.1 keepalive`
- endpoint: `/score_dense_f32_batch_v1`
- batch shape: `batch_records = 128`

This is the path the serving work was organized around.

This repository is here to show a tree-model stack that actually moves, not a server that merely compiles.

## Build

Check the public workspace:

```bash
cargo check -p risk-core \
  -p risk-quickscorer \
  -p risk-quickscorer-standalone-l2 \
  -p risk-server-tokio \
  -p risk-server-glommio \
  -p risk-bench3
```

Build release binaries:

```bash
cargo build --release -p risk-server-tokio -p risk-server-glommio -p risk-bench3
```

## Minimal Usage

Start the Tokio server:

```bash
cargo run -p risk-server-tokio -- \
  --listen 127.0.0.1:8080
```

Start the Glommio server:

```bash
cargo run -p risk-server-glommio -- \
  --listen 127.0.0.1:8080
```

Run the benchmark client:

```bash
cargo run -p risk-bench3 -- \
  --target http://127.0.0.1:8080/score_dense_f32_batch_v1 \
  --mode throughput \
  --batch-mode http1 \
  --batch-records 128
```

To run the full scoring path you will need a local model bundle and local dense-row inputs.

## Design

Riskanvil is split into three layers:

1. inference runtimes
   - `risk-quickscorer`
   - `risk-quickscorer-standalone-l2`
2. shared pipeline and schema
   - `risk-core`
3. serving and end-to-end measurement
   - `risk-server-tokio`
   - `risk-server-glommio`
   - `risk-bench3`

This separation keeps kernel-facing code, pipeline code, and server/runtime code in different places even when they are tuned against the same workload.

## Scope

This repository is focused on inference and serving:

- kernel-adjacent runtime work
- HTTP batch scoring
- throughput benchmarking

It optimizes for hot-path behavior, explicit ownership, and measurable throughput.

It is not a general web framework or a product application template.

## License

See [LICENSE](LICENSE).
