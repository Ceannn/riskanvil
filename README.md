# Riskanvil

`riskanvil` is a Rust workspace for high-throughput tree-model inference and serving.

This public snapshot keeps the parts that matter for the systems story:

- an inference core for dense row scoring
- a QuickScorer-oriented L1 runtime
- a standalone L2 runtime
- two HTTP servers
- a throughput benchmark client

What is *not* included:

- private training pipelines
- PT / parquet / raw datasets
- internal model bundles and release artifacts
- abandoned side tracks that were only useful during local iteration

The result is a smaller workspace that can be read, built, and profiled without dragging the private project along with it.

## Workspace

The public workspace contains six crates:

- `crates/risk-core`
  - shared scoring pipeline, schema, routing, and server-facing integration points
- `crates/risk-quickscorer`
  - QuickScorer-oriented L1 runtime and supporting execution code
- `crates/risk-quickscorer-standalone-l2`
  - standalone L2 runtime and kernel-facing tooling
- `crates/risk-server-tokio`
  - Tokio HTTP server
- `crates/risk-server-glommio`
  - Glommio HTTP server
- `crates/risk-bench3`
  - high-rate benchmark client used for end-to-end throughput work

## What This Repo Is

This is not a polished product server.

It is a working systems codebase that grew around one question:

> how far can a tree-model scoring stack be pushed when the kernel is fast enough that the bottleneck moves into serving?

That question ended up touching:

- model-side execution layout
- L1 / L2 runtime design
- HTTP batch serving
- benchmark-client behavior
- request lifecycle and completion turnover

The code reflects that history. Some crates are compact and clean. Some hot paths are intentionally dense. The public version keeps the useful parts of that work without pretending everything is a general-purpose framework.

## Main Throughput Path

The main end-to-end path in this workspace is:

- server: `risk-server-tokio`
- client: `risk-bench3`
- transport: `HTTP/1.1 keepalive`
- endpoint: `/score_dense_f32_batch_v1`
- batch shape: `batch_records = 128`

This is the path the later serving work was optimized around.

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

## Minimal Startup

Tokio server:

```bash
cargo run -p risk-server-tokio -- \
  --listen 127.0.0.1:8080
```

Glommio server:

```bash
cargo run -p risk-server-glommio -- \
  --listen 127.0.0.1:8080
```

Bench client:

```bash
cargo run -p risk-bench3 -- \
  --target http://127.0.0.1:8080/score_dense_f32_batch_v1 \
  --mode throughput \
  --batch-mode http1 \
  --batch-records 128
```

The benchmark expects local dense-row inputs and route metadata. Those artifacts are intentionally not shipped in this public repo.

## Notes On Artifacts

This repository does not include model bundles or production data.

If you want to run the end-to-end path for real, you will need to provide your own:

- QuickScorer bundle directory
- dense row input file
- optional route metadata file

The code paths remain here; the private artifacts do not.

## Design Shape

The public snapshot keeps a clear separation between three layers:

1. `risk-quickscorer` and `risk-quickscorer-standalone-l2`
   - kernel-adjacent inference runtimes
2. `risk-core`
   - pipeline and schema layer
3. `risk-server-*` and `risk-bench3`
   - serving and end-to-end throughput tooling

That split is deliberate. The inference core and the serving runtime are related, but they are not the same optimization problem.

## Why The Code Looks The Way It Does

Some files in this repo are straightforward. Some are not.

That is mostly a consequence of the target:

- fixed-shape batch scoring
- hot-path memory control
- explicit ownership over request progression
- low tolerance for abstraction overhead in kernel-adjacent code

Rust helps here by making the unsafe and performance-sensitive regions explicit instead of letting the whole codebase decay into one giant undefined-behavior zone.

## Status

This repository is best read as a compact public cut of a larger private workspace.

It is stable enough to build and inspect.
It is honest enough to show the real hot paths.
It is small enough to publish without shipping the private baggage.

## License

See [LICENSE](LICENSE).
