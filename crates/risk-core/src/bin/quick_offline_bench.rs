use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use memmap2::Mmap;
use pprof::ProfilerGuardBuilder;
use risk_core::quickscorer::policy::{L2FeatureSource, QuickPolicy};
use risk_core::quickscorer::{QuickRouteMeta, QuickScorerEngine};
use risk_quickscorer::MinpackRuntime;
use risk_quickscorer_standalone_l2::{
    benchmark_qs_l2_prefix_cal, StandaloneL2Runtime, StandaloneQsPrefixCalBenchConfig,
};
use serde::Serialize;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "quick_offline_bench")]
struct Cli {
    #[arg(long)]
    bundle_dir: PathBuf,
    #[arg(long)]
    dense_file: PathBuf,
    #[arg(long)]
    dense_dim: usize,
    #[arg(long)]
    route_meta_tsv: Option<PathBuf>,
    #[arg(long, value_enum)]
    mode: BenchMode,
    #[arg(long, default_value_t = 20000)]
    warmup_rows: usize,
    #[arg(long, default_value_t = 120000)]
    measure_rows: usize,
    #[arg(long, default_value_t = 7)]
    repeat: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    summary_json: Option<PathBuf>,
    #[arg(long)]
    dump_tsv: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    experimental_l2: bool,
    #[arg(long, default_value_t = false)]
    l2_kernel_7945hx_v1: bool,
    #[arg(long, default_value_t = false)]
    standalone_l2: bool,
    #[arg(long, default_value_t = false)]
    standalone_l2_synthetic: bool,
    #[arg(long)]
    standalone_l2_feat_bin: Option<PathBuf>,
    #[arg(long)]
    standalone_l2_fixed_tau: Option<f32>,
    #[arg(long)]
    pprof_flamegraph: Option<PathBuf>,
    #[arg(long, default_value_t = 999)]
    pprof_hz: i32,
    #[arg(long, default_value_t = false)]
    trace_l2_stages: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum BenchMode {
    L1,
    L2,
    Mixed,
}

#[derive(Debug, Clone, Copy)]
struct RouteMetaRow {
    row_idx: usize,
    transaction_id: u64,
    l2_tau_used: f32,
    fold_id: i32,
    seg_prod_amtbin: u32,
}

#[derive(Debug)]
struct DenseRows {
    _mmap: Mmap,
    dim: usize,
    n_rows: usize,
    bytes: *const u8,
    len: usize,
}

unsafe impl Send for DenseRows {}
unsafe impl Sync for DenseRows {}

impl DenseRows {
    fn open(path: &Path, dim: usize) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        let row_bytes = dim
            .checked_mul(4)
            .ok_or_else(|| anyhow!("dense_dim too large: {}", dim))?;
        if row_bytes == 0 {
            bail!("dense_dim must be > 0");
        }
        if mmap.len() % row_bytes != 0 {
            bail!(
                "dense file len {} is not divisible by row bytes {}",
                mmap.len(),
                row_bytes
            );
        }
        let n_rows = mmap.len() / row_bytes;
        let bytes = mmap.as_ptr();
        let len = mmap.len();
        Ok(Self {
            _mmap: mmap,
            dim,
            n_rows,
            bytes,
            len,
        })
    }

    #[inline]
    fn row_bytes(&self, row_idx: usize) -> &[u8] {
        let row_len = self.dim * 4;
        let off = row_idx * row_len;
        debug_assert!(off + row_len <= self.len);
        unsafe { std::slice::from_raw_parts(self.bytes.add(off), row_len) }
    }
}

#[derive(Debug, Serialize)]
struct BenchSummary {
    mode: BenchMode,
    dense_dim: usize,
    total_rows_in_file: usize,
    candidate_rows: usize,
    warmup_rows: usize,
    measure_rows: usize,
    repeat: usize,
    rows_per_sec_median: f64,
    rows_per_sec_mean: f64,
    rows_per_sec_best: f64,
    rows_per_sec_worst: f64,
    per_run_rows_per_sec: Vec<f64>,
    used_l2_rows: usize,
    allow_rows: usize,
    deny_rows: usize,
    manual_review_rows: usize,
}

#[derive(Clone, Copy, Debug)]
enum FinalDecision {
    Allow,
    Deny,
    ManualReview,
}

impl FinalDecision {
    fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "ALLOW",
            Self::Deny => "DENY",
            Self::ManualReview => "MANUAL_REVIEW",
        }
    }
}

struct PprofCfg<'a> {
    flamegraph: Option<&'a Path>,
    hz: i32,
}

fn maybe_start_profiler(cfg: &PprofCfg<'_>) -> Result<Option<pprof::ProfilerGuard<'static>>> {
    if cfg.flamegraph.is_none() {
        return Ok(None);
    }
    Ok(Some(
        ProfilerGuardBuilder::default()
            .frequency(cfg.hz)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .context("start pprof profiler")?,
    ))
}

fn maybe_write_flamegraph(
    guard: Option<pprof::ProfilerGuard<'static>>,
    cfg: &PprofCfg<'_>,
) -> Result<()> {
    let (Some(guard), Some(path)) = (guard, cfg.flamegraph) else {
        return Ok(());
    };
    let report = guard.report().build().context("build pprof report")?;
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    report
        .flamegraph(file)
        .with_context(|| format!("write flamegraph {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.trace_l2_stages {
        std::env::set_var("QS_TRACE_L2_STAGES", "1");
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("risk_quickscorer=trace")),
            )
            .with_target(false)
            .try_init();
    }
    let dense = DenseRows::open(&cli.dense_file, cli.dense_dim)?;
    let route_meta = match &cli.route_meta_tsv {
        Some(path) => Some(load_route_meta_tsv(path, dense.n_rows)?),
        None => None,
    };
    let pprof_cfg = PprofCfg {
        flamegraph: cli.pprof_flamegraph.as_deref(),
        hz: cli.pprof_hz,
    };
    let summary = match cli.mode {
        BenchMode::L1 => run_l1_bench(&cli, &dense, &pprof_cfg)?,
        BenchMode::L2 => run_l2_bench(&cli, &dense, route_meta.as_deref(), &pprof_cfg)?,
        BenchMode::Mixed => run_mixed_bench(&cli, &dense, route_meta.as_deref(), &pprof_cfg)?,
    };

    if let Some(path) = &cli.summary_json {
        fs::write(path, serde_json::to_vec_pretty(&summary)?)
            .with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode={:?} candidate_rows={} repeat={} rows/s median={:.1} mean={:.1} best={:.1} worst={:.1} used_l2={} allow={} deny={} manual_review={}",
        summary.mode,
        summary.candidate_rows,
        summary.repeat,
        summary.rows_per_sec_median,
        summary.rows_per_sec_mean,
        summary.rows_per_sec_best,
        summary.rows_per_sec_worst,
        summary.used_l2_rows,
        summary.allow_rows,
        summary.deny_rows,
        summary.manual_review_rows
    );
    Ok(())
}

fn run_l1_bench(cli: &Cli, dense: &DenseRows, pprof_cfg: &PprofCfg<'_>) -> Result<BenchSummary> {
    let runtime = MinpackRuntime::load(&cli.bundle_dir)?;
    let candidate_rows = limit_rows(dense.n_rows, cli.max_rows);
    let warmup_rows = cli.warmup_rows.min(candidate_rows);
    let measure_rows = cli.measure_rows.min(candidate_rows);

    for row_idx in 0..warmup_rows {
        std::hint::black_box(runtime.predict_l1_bytes_nomiss(dense.row_bytes(row_idx))?);
    }

    let guard = maybe_start_profiler(pprof_cfg)?;
    let mut runs = Vec::with_capacity(cli.repeat);
    let mut allow_rows = 0usize;
    let mut manual_review_rows = 0usize;
    for rep in 0..cli.repeat {
        let t0 = Instant::now();
        let mut allow = 0usize;
        let mut review = 0usize;
        for row_idx in 0..measure_rows {
            let out = runtime.predict_l1_bytes_nomiss(dense.row_bytes(row_idx))?;
            if out.passed {
                allow += 1;
            } else {
                review += 1;
            }
            std::hint::black_box(out.score);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        runs.push(measure_rows as f64 / elapsed.max(1e-9));
        if rep + 1 == cli.repeat {
            allow_rows = allow;
            manual_review_rows = review;
        }
    }
    maybe_write_flamegraph(guard, pprof_cfg)?;

    Ok(summarize_runs(
        BenchMode::L1,
        dense,
        candidate_rows,
        warmup_rows,
        measure_rows,
        cli.repeat,
        runs,
        0,
        allow_rows,
        0,
        manual_review_rows,
    ))
}

fn run_l2_bench(
    cli: &Cli,
    dense: &DenseRows,
    route_meta: Option<&[Option<RouteMetaRow>]>,
    pprof_cfg: &PprofCfg<'_>,
) -> Result<BenchSummary> {
    if cli.standalone_l2 {
        return run_l2_bench_standalone(cli, dense, route_meta, pprof_cfg);
    }
    let use_dedicated = cli.l2_kernel_7945hx_v1 || cli.experimental_l2;
    let route_meta =
        route_meta.ok_or_else(|| anyhow!("--route-meta-tsv is required for mode=l2"))?;
    let runtime = MinpackRuntime::load(&cli.bundle_dir)?;
    let policy = QuickPolicy::load_bundle(&cli.bundle_dir)?;
    let l2_policy = policy
        .l2
        .as_ref()
        .ok_or_else(|| anyhow!("bundle missing l2 policy"))?;

    let candidate_limit = limit_rows(dense.n_rows, cli.max_rows);
    let mut prepared = Vec::new();
    for row_idx in 0..candidate_limit {
        let Some(meta) = route_meta.get(row_idx).and_then(|x| *x) else {
            continue;
        };
        let l1_out = runtime.predict_l1_bytes_nomiss(dense.row_bytes(row_idx))?;
        if l1_out.passed {
            continue;
        }
        let mut l2_row = vec![0.0f32; l2_policy.dim];
        materialize_l2_row_from_bytes(
            dense.row_bytes(row_idx),
            l1_out.score,
            l2_policy,
            &mut l2_row,
        );
        prepared.push((l2_row, meta.fold_id, meta.l2_tau_used));
    }
    if prepared.is_empty() {
        bail!("no L2 candidate rows prepared");
    }

    let candidate_rows = prepared.len();
    let warmup_rows = cli.warmup_rows.min(candidate_rows);
    let measure_rows = cli.measure_rows.min(candidate_rows);
    for (row, fold, tau) in prepared.iter().take(warmup_rows) {
        let out = if use_dedicated {
            runtime.predict_l2_row_nomiss_experimental_7945hx(row.as_slice(), *tau, *fold)?
        } else {
            runtime.predict_l2_row_nomiss(row.as_slice(), *tau, *fold)?
        };
        std::hint::black_box(out);
    }

    let guard = maybe_start_profiler(pprof_cfg)?;
    let mut runs = Vec::with_capacity(cli.repeat);
    let mut deny_rows = 0usize;
    let mut manual_review_rows = 0usize;
    for rep in 0..cli.repeat {
        let t0 = Instant::now();
        let mut deny = 0usize;
        let mut review = 0usize;
        for (row, fold, tau) in prepared.iter().take(measure_rows) {
            let out = if use_dedicated {
                runtime.predict_l2_row_nomiss_experimental_7945hx(row.as_slice(), *tau, *fold)?
            } else {
                runtime.predict_l2_row_nomiss(row.as_slice(), *tau, *fold)?
            };
            if out.reject {
                deny += 1;
            } else {
                review += 1;
            }
            std::hint::black_box(out.score);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        runs.push(measure_rows as f64 / elapsed.max(1e-9));
        if rep + 1 == cli.repeat {
            deny_rows = deny;
            manual_review_rows = review;
        }
    }
    maybe_write_flamegraph(guard, pprof_cfg)?;

    Ok(summarize_runs(
        BenchMode::L2,
        dense,
        candidate_rows,
        warmup_rows,
        measure_rows,
        cli.repeat,
        runs,
        measure_rows,
        0,
        deny_rows,
        manual_review_rows,
    ))
}

fn run_mixed_bench(
    cli: &Cli,
    dense: &DenseRows,
    route_meta: Option<&[Option<RouteMetaRow>]>,
    pprof_cfg: &PprofCfg<'_>,
) -> Result<BenchSummary> {
    if cli.standalone_l2 {
        return run_mixed_bench_standalone(cli, dense, route_meta, pprof_cfg);
    }
    let use_dedicated = cli.l2_kernel_7945hx_v1 || cli.experimental_l2;
    let route_meta =
        route_meta.ok_or_else(|| anyhow!("--route-meta-tsv is required for mode=mixed"))?;
    let engine = (!use_dedicated)
        .then(|| QuickScorerEngine::load(&cli.bundle_dir))
        .transpose()?;
    let runtime = use_dedicated.then(|| MinpackRuntime::load(&cli.bundle_dir)).transpose()?;
    let policy = use_dedicated.then(|| QuickPolicy::load_bundle(&cli.bundle_dir)).transpose()?;
    let l2_policy = policy.as_ref().and_then(|x| x.l2.as_ref());
    let candidate_rows = limit_rows(dense.n_rows, cli.max_rows);
    let warmup_rows = cli.warmup_rows.min(candidate_rows);
    let measure_rows = cli.measure_rows.min(candidate_rows);

    for row_idx in 0..warmup_rows {
        let meta = route_meta_for_row(route_meta, row_idx)?;
        if let (Some(runtime), Some(l2_policy)) = (runtime.as_ref(), l2_policy) {
            let l1_out = runtime.predict_l1_bytes_nomiss(dense.row_bytes(row_idx))?;
            if !l1_out.passed {
                let mut l2_row = vec![0.0f32; l2_policy.dim];
                materialize_l2_row_from_bytes(
                    dense.row_bytes(row_idx),
                    l1_out.score,
                    l2_policy,
                    &mut l2_row,
                );
                std::hint::black_box(
                    runtime.predict_l2_row_nomiss_experimental_7945hx(
                        l2_row.as_slice(),
                        meta.l2_tau_used
                            .expect("route meta must carry l2_tau_used in experimental mode"),
                        meta.fold_id,
                    )?,
                );
            } else {
                std::hint::black_box(l1_out.score);
            }
        } else {
            std::hint::black_box(
                engine
                    .as_ref()
                    .unwrap()
                    .predict_from_l1_bytes_with_meta(dense.row_bytes(row_idx), Some(&meta))?,
            );
        }
    }

    let guard = maybe_start_profiler(pprof_cfg)?;
    let mut runs = Vec::with_capacity(cli.repeat);
    let mut used_l2_rows = 0usize;
    let mut allow_rows = 0usize;
    let mut deny_rows = 0usize;
    let mut manual_review_rows = 0usize;
    for rep in 0..cli.repeat {
        let t0 = Instant::now();
        let mut used_l2 = 0usize;
        let mut allow = 0usize;
        let mut deny = 0usize;
        let mut review = 0usize;
        for row_idx in 0..measure_rows {
            let meta = route_meta_for_row(route_meta, row_idx)?;
            if let (Some(runtime), Some(l2_policy)) = (runtime.as_ref(), l2_policy) {
                let l1_out = runtime.predict_l1_bytes_nomiss(dense.row_bytes(row_idx))?;
                if l1_out.passed {
                    allow += 1;
                    std::hint::black_box(l1_out.score);
                } else {
                    used_l2 += 1;
                    let mut l2_row = vec![0.0f32; l2_policy.dim];
                    materialize_l2_row_from_bytes(
                        dense.row_bytes(row_idx),
                        l1_out.score,
                        l2_policy,
                        &mut l2_row,
                    );
                    let l2_out = runtime.predict_l2_row_nomiss_experimental_7945hx(
                        l2_row.as_slice(),
                        meta.l2_tau_used
                            .expect("route meta must carry l2_tau_used in experimental mode"),
                        meta.fold_id,
                    )?;
                    if l2_out.reject {
                        deny += 1;
                    } else {
                        review += 1;
                    }
                    std::hint::black_box(l2_out.score);
                }
            } else {
                let out = engine
                    .as_ref()
                    .unwrap()
                    .predict_from_l1_bytes_with_meta(dense.row_bytes(row_idx), Some(&meta))?;
                used_l2 += usize::from(out.used_l2);
                match out.decision {
                    risk_core::schema::Decision::Allow => allow += 1,
                    risk_core::schema::Decision::Deny => deny += 1,
                    risk_core::schema::Decision::ManualReview => review += 1,
                    risk_core::schema::Decision::DegradeAllow => allow += 1,
                }
                std::hint::black_box(out.final_score);
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        runs.push(measure_rows as f64 / elapsed.max(1e-9));
        if rep + 1 == cli.repeat {
            used_l2_rows = used_l2;
            allow_rows = allow;
            deny_rows = deny;
            manual_review_rows = review;
        }
    }
    maybe_write_flamegraph(guard, pprof_cfg)?;

    Ok(summarize_runs(
        BenchMode::Mixed,
        dense,
        candidate_rows,
        warmup_rows,
        measure_rows,
        cli.repeat,
        runs,
        used_l2_rows,
        allow_rows,
        deny_rows,
        manual_review_rows,
    ))
}

fn run_l2_bench_standalone(
    cli: &Cli,
    dense: &DenseRows,
    route_meta: Option<&[Option<RouteMetaRow>]>,
    pprof_cfg: &PprofCfg<'_>,
) -> Result<BenchSummary> {
    let _ = dense;
    let _ = route_meta;
    let guard = maybe_start_profiler(pprof_cfg)?;
    let mut runs = Vec::with_capacity(cli.repeat);
    let mut candidate_rows = 0usize;
    let mut deny_rows = 0usize;
    let mut manual_review_rows = 0usize;
    let mut used_l2_rows = 0usize;
    for rep in 0..cli.repeat {
        let stats = benchmark_qs_l2_prefix_cal(
            &cli.bundle_dir,
            StandaloneQsPrefixCalBenchConfig {
                threads: 1,
                chunk_rows: 128,
                parallel_min_rows: 4096,
                max_rows: cli.max_rows.or(Some(cli.measure_rows)),
            },
        )?;
        runs.push(stats.rows_per_sec);
        if rep + 1 == cli.repeat {
            candidate_rows = stats.n_rows;
            used_l2_rows = stats.route_active_rows;
            deny_rows = stats.deny_rows;
            manual_review_rows = stats.manual_review_rows;
        }
    }
    maybe_write_flamegraph(guard, pprof_cfg)?;

    Ok(summarize_runs(
        BenchMode::L2,
        dense,
        candidate_rows,
        0,
        candidate_rows,
        cli.repeat,
        runs,
        used_l2_rows,
        0,
        deny_rows,
        manual_review_rows,
    ))
}

fn run_mixed_bench_standalone(
    cli: &Cli,
    dense: &DenseRows,
    route_meta: Option<&[Option<RouteMetaRow>]>,
    pprof_cfg: &PprofCfg<'_>,
) -> Result<BenchSummary> {
    let route_meta =
        route_meta.ok_or_else(|| anyhow!("--route-meta-tsv is required for mode=mixed"))?;
    let l1_runtime = MinpackRuntime::load(&cli.bundle_dir)?;
    let runtime = if let Some(path) = cli.standalone_l2_feat_bin.as_deref() {
        StandaloneL2Runtime::load_with_feat_bin_override(&cli.bundle_dir, Some(path))?
    } else {
        StandaloneL2Runtime::load(&cli.bundle_dir)?
    };
    if !cli.standalone_l2_synthetic && runtime.feat_rows() != dense.n_rows {
        bail!(
            "standalone mixed path requires aligned L2 feature rows: feat_bin_rows={} dense_rows={}. \
             Current bundle feat_bin is standalone benchmark data, not the primitive PT corpus.",
            runtime.feat_rows(),
            dense.n_rows
        );
    }

    let candidate_rows = limit_rows(dense.n_rows, cli.max_rows);
    let warmup_rows = cli.warmup_rows.min(candidate_rows);
    let measure_rows = cli.measure_rows.min(candidate_rows);

    let mut warm_scratch = runtime.new_scratch();
    for row_idx in 0..warmup_rows {
        let meta = route_meta_for_row(route_meta, row_idx)?;
        let l1_out = l1_runtime.predict_l1_bytes_nomiss(dense.row_bytes(row_idx))?;
        if !l1_out.passed {
            let sidecar_row_idx = if cli.standalone_l2_synthetic {
                meta.row_idx as usize % runtime.feat_rows()
            } else {
                meta.row_idx as usize
            };
            let tau = cli
                .standalone_l2_fixed_tau
                .unwrap_or(meta.l2_tau_used.expect("route meta must carry l2_tau_used"));
            std::hint::black_box(runtime.predict_l2_row_by_index_with_scratch(
                sidecar_row_idx,
                tau,
                meta.fold_id,
                &mut warm_scratch,
            )?);
        } else {
            std::hint::black_box(l1_out.score);
        }
    }

    let guard = maybe_start_profiler(pprof_cfg)?;
    let mut runs = Vec::with_capacity(cli.repeat);
    let mut used_l2_rows = 0usize;
    let mut allow_rows = 0usize;
    let mut deny_rows = 0usize;
    let mut manual_review_rows = 0usize;
    let mut dump_rows = Vec::new();
    for rep in 0..cli.repeat {
        let t0 = Instant::now();
        let mut used_l2 = 0usize;
        let mut allow = 0usize;
        let mut deny = 0usize;
        let mut review = 0usize;
        let mut scratch = runtime.new_scratch();
        let write_dump = cli.dump_tsv.is_some() && rep + 1 == cli.repeat;
        if write_dump {
            dump_rows.clear();
            dump_rows.push(
                "row_idx\ttransaction_id\tused_l2\tdecision\tl2_score\ttau_used\tfold_id\n"
                    .to_string(),
            );
        }
        for row_idx in 0..measure_rows {
            let meta = route_meta_for_row(route_meta, row_idx)?;
            let l1_out = l1_runtime.predict_l1_bytes_nomiss(dense.row_bytes(row_idx))?;
            if l1_out.passed {
                allow += 1;
                if write_dump {
                    dump_rows.push(format!(
                        "{}\t{}\t0\t{}\t\t{}\t{}\n",
                        meta.row_idx,
                        meta.transaction_id,
                        FinalDecision::Allow.as_str(),
                        meta.l2_tau_used.unwrap_or_default(),
                        meta.fold_id
                    ));
                }
                std::hint::black_box(l1_out.score);
            } else {
                used_l2 += 1;
                let sidecar_row_idx = if cli.standalone_l2_synthetic {
                    meta.row_idx as usize % runtime.feat_rows()
                } else {
                    meta.row_idx as usize
                };
                let tau = cli
                    .standalone_l2_fixed_tau
                    .unwrap_or(meta.l2_tau_used.expect("route meta must carry l2_tau_used"));
                let l2_out = runtime.predict_l2_row_by_index_with_scratch(
                    sidecar_row_idx,
                    tau,
                    meta.fold_id,
                    &mut scratch,
                )?;
                if l2_out.reject {
                    deny += 1;
                    if write_dump {
                        dump_rows.push(format!(
                            "{}\t{}\t1\t{}\t{}\t{}\t{}\n",
                            meta.row_idx,
                            meta.transaction_id,
                            FinalDecision::Deny.as_str(),
                            l2_out.score,
                            meta.l2_tau_used.unwrap_or_default(),
                            meta.fold_id
                        ));
                    }
                } else {
                    review += 1;
                    if write_dump {
                        dump_rows.push(format!(
                            "{}\t{}\t1\t{}\t{}\t{}\t{}\n",
                            meta.row_idx,
                            meta.transaction_id,
                            FinalDecision::ManualReview.as_str(),
                            l2_out.score,
                            meta.l2_tau_used.unwrap_or_default(),
                            meta.fold_id
                        ));
                    }
                }
                std::hint::black_box(l2_out.score);
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        runs.push(measure_rows as f64 / elapsed.max(1e-9));
        if rep + 1 == cli.repeat {
            used_l2_rows = used_l2;
            allow_rows = allow;
            deny_rows = deny;
            manual_review_rows = review;
        }
    }
    maybe_write_flamegraph(guard, pprof_cfg)?;
    if let Some(path) = &cli.dump_tsv {
        fs::write(path, dump_rows.concat()).with_context(|| format!("write {}", path.display()))?;
    }

    Ok(summarize_runs(
        BenchMode::Mixed,
        dense,
        candidate_rows,
        warmup_rows,
        measure_rows,
        cli.repeat,
        runs,
        used_l2_rows,
        allow_rows,
        deny_rows,
        manual_review_rows,
    ))
}

fn route_meta_for_row(
    route_meta: &[Option<RouteMetaRow>],
    row_idx: usize,
) -> Result<QuickRouteMeta> {
    let meta = route_meta
        .get(row_idx)
        .and_then(|x| *x)
        .ok_or_else(|| anyhow!("missing route meta for row {}", row_idx))?;
    Ok(QuickRouteMeta {
        row_idx: meta.row_idx as u32,
        transaction_id: meta.transaction_id,
        fold_id: meta.fold_id,
        seg_prod_amtbin: meta.seg_prod_amtbin,
        l2_tau_used: Some(meta.l2_tau_used),
    })
}

fn summarize_runs(
    mode: BenchMode,
    dense: &DenseRows,
    candidate_rows: usize,
    warmup_rows: usize,
    measure_rows: usize,
    repeat: usize,
    mut runs: Vec<f64>,
    used_l2_rows: usize,
    allow_rows: usize,
    deny_rows: usize,
    manual_review_rows: usize,
) -> BenchSummary {
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rows_per_sec_median = runs[runs.len() / 2];
    let rows_per_sec_mean = runs.iter().sum::<f64>() / runs.len().max(1) as f64;
    let rows_per_sec_best = *runs.last().unwrap_or(&0.0);
    let rows_per_sec_worst = *runs.first().unwrap_or(&0.0);
    BenchSummary {
        mode,
        dense_dim: dense.dim,
        total_rows_in_file: dense.n_rows,
        candidate_rows,
        warmup_rows,
        measure_rows,
        repeat,
        rows_per_sec_median,
        rows_per_sec_mean,
        rows_per_sec_best,
        rows_per_sec_worst,
        per_run_rows_per_sec: runs,
        used_l2_rows,
        allow_rows,
        deny_rows,
        manual_review_rows,
    }
}

fn limit_rows(total_rows: usize, max_rows: Option<usize>) -> usize {
    max_rows.unwrap_or(total_rows).min(total_rows)
}

fn materialize_l2_row_from_bytes(
    l1_bytes: &[u8],
    l1_score: f32,
    l2_policy: &risk_core::quickscorer::policy::L2Policy,
    out: &mut [f32],
) {
    for (i, src) in l2_policy.feature_sources.iter().enumerate() {
        out[i] = match src {
            L2FeatureSource::FromL1(idx) => read_l1_f32(l1_bytes, *idx),
            L2FeatureSource::L1Score => l1_score,
        };
    }
}

fn read_l1_f32(bytes: &[u8], idx: usize) -> f32 {
    let off = idx * 4;
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn load_route_meta_tsv(path: &Path, max_rows: usize) -> Result<Vec<Option<RouteMetaRow>>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| anyhow!("empty route meta tsv: {}", path.display()))?;
    let cols: Vec<&str> = header.split('\t').collect();
    let idx_row = find_col(&cols, "row_idx")?;
    let idx_txn = find_col(&cols, "TransactionID")?;
    let idx_tau = find_col(&cols, "l2_tau_used")?;
    let idx_fold = find_col(&cols, "fold_id")?;
    let idx_seg = find_col(&cols, "seg_prod_amtbin")?;

    let mut out = vec![None; max_rows];
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let row_idx: usize = parts
            .get(idx_row)
            .ok_or_else(|| anyhow!("route meta row_idx missing"))?
            .parse()
            .with_context(|| format!("parse row_idx in {}", path.display()))?;
        if row_idx >= max_rows {
            continue;
        }
        let transaction_id = parts
            .get(idx_txn)
            .ok_or_else(|| anyhow!("route meta TransactionID missing"))?
            .parse()
            .with_context(|| format!("parse TransactionID in {}", path.display()))?;
        let l2_tau_used = parts
            .get(idx_tau)
            .ok_or_else(|| anyhow!("route meta l2_tau_used missing"))?
            .parse()
            .with_context(|| format!("parse l2_tau_used in {}", path.display()))?;
        let fold_id = parts
            .get(idx_fold)
            .ok_or_else(|| anyhow!("route meta fold_id missing"))?
            .parse()
            .with_context(|| format!("parse fold_id in {}", path.display()))?;
        let seg_prod_amtbin = parts
            .get(idx_seg)
            .ok_or_else(|| anyhow!("route meta seg_prod_amtbin missing"))?
            .parse()
            .with_context(|| format!("parse seg_prod_amtbin in {}", path.display()))?;
        out[row_idx] = Some(RouteMetaRow {
            row_idx,
            transaction_id,
            l2_tau_used,
            fold_id,
            seg_prod_amtbin,
        });
    }
    Ok(out)
}

fn find_col(cols: &[&str], needle: &str) -> Result<usize> {
    cols.iter()
        .position(|x| *x == needle)
        .ok_or_else(|| anyhow!("missing column {}", needle))
}
