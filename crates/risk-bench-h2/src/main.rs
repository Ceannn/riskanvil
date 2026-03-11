mod config;
mod dataset;
mod metrics;
mod protocol;
mod worker;

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use config::Args;
use crossbeam_channel::RecvTimeoutError;
use dataset::{parse_target, WorkloadSource};
use metrics::{
    build_summary, make_window_row, write_summary, write_window_header, write_window_row, AggEvent,
    StatsAgg,
};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tracing_subscriber::Layer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use worker::spawn_worker;

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let target = parse_target(&args.url)?;
    let workload = WorkloadSource::load(&args)?;
    let worker_cpus = parse_cpu_list(&args.worker_cpus)?;
    let worker_count = resolve_worker_count(args.workers, worker_cpus.as_ref())?;

    if worker_count == 0 {
        bail!("worker count resolved to zero");
    }
    if args.conns_per_worker == 0 {
        bail!("--conns-per-worker must be > 0");
    }

    if args.progress {
        eprintln!(
            "[risk-bench-h2] start workers={} conns_per_worker={} qps={} workload={:?}",
            worker_count, args.conns_per_worker, args.qps, args.workload
        );
    }

    let (tx, rx) = crossbeam_channel::unbounded();
    let stop = Arc::new(AtomicBool::new(false));
    let mut joins = Vec::with_capacity(worker_count);
    for worker_id in 0..worker_count {
        let cpu = worker_cpus
            .as_ref()
            .and_then(|cpus| cpus.get(worker_id).copied());
        joins.push(spawn_worker(
            worker_id,
            worker_count,
            cpu,
            args.clone(),
            target.clone(),
            workload.clone(),
            tx.clone(),
            stop.clone(),
        ));
    }
    drop(tx);

    let mut total = StatsAgg::new()?;
    let mut window = StatsAgg::new()?;
    let mut worker_done = 0usize;
    let mut window_file = if let Some(path) = args.window_csv.as_ref() {
        Some(write_window_header(path)?)
    } else {
        None
    };
    let window_every = Duration::from_millis(args.window_ms.max(1));
    let mut last_window = Instant::now();

    while worker_done < worker_count {
        match rx.recv_timeout(window_every) {
            Ok(AggEvent::WorkerSnapshot(snapshot)) => {
                total.record_snapshot(&snapshot)?;
                window.record_snapshot(&snapshot)?;
            }
            Ok(AggEvent::WorkerDone(done)) => {
                total.record_worker_done(&done);
                window.record_worker_done(&done);
                worker_done += 1;
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Some(file) = window_file.as_mut() {
                    let elapsed = last_window.elapsed();
                    let row = make_window_row(&window, elapsed, args.qps, args.batch_records);
                    write_window_row(file, &row)?;
                    window.reset_window()?;
                    last_window = Instant::now();
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    for join in joins {
        join.join()
            .map_err(|_| anyhow!("worker thread panicked"))??;
    }

    let summary = build_summary(
        &args,
        worker_cpus,
        workload.payload_rows(),
        workload.dense_dim(),
        workload.route_meta_enabled(),
        &total,
    );

    println!(
        "attempted_rps={:.1} ok_rps={:.1} dropped={} verdict={:?}",
        summary.counts.attempted_rps,
        summary.counts.ok_rps,
        summary.counts.dropped,
        summary.verdict
    );

    if let Some(path) = args.summary_json.as_ref() {
        write_summary(path, &summary)?;
    }

    Ok(())
}

fn init_tracing() {
    if std::env::var_os("RISK_BENCH_H2_TOKIO_CONSOLE").is_some() {
        let builder = console_subscriber::ConsoleLayer::builder();
        let server_addr: Option<SocketAddr> = std::env::var("RISK_BENCH_H2_CONSOLE_ADDR")
            .ok()
            .and_then(|v| v.parse().ok());
        let publish_interval = std::env::var("RISK_BENCH_H2_CONSOLE_PUBLISH_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_millis);
        let mut builder = match server_addr {
            Some(addr) => builder.server_addr(addr),
            None => builder,
        };
        if let Some(interval) = publish_interval {
            builder = builder.publish_interval(interval);
        }
        tracing_subscriber::registry()
            .with(builder.spawn())
            .with(
                tracing_subscriber::fmt::layer().with_filter(
                    EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| EnvFilter::new("warn")),
                ),
            )
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("warn")),
            )
            .init();
    }
}

fn resolve_worker_count(explicit: usize, worker_cpus: Option<&Vec<usize>>) -> Result<usize> {
    if explicit > 0 {
        return Ok(explicit);
    }
    if let Some(cpus) = worker_cpus {
        return Ok(cpus.len());
    }
    let avail = std::thread::available_parallelism()
        .context("available_parallelism")?
        .get();
    Ok(avail.saturating_sub(1).max(1))
}

fn parse_cpu_list(spec: &Option<String>) -> Result<Option<Vec<usize>>> {
    let Some(spec) = spec.as_ref() else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a
                .trim()
                .parse()
                .with_context(|| format!("bad cpu range: {part}"))?;
            let end: usize = b
                .trim()
                .parse()
                .with_context(|| format!("bad cpu range: {part}"))?;
            if end < start {
                bail!("bad cpu range: {part}");
            }
            out.extend(start..=end);
        } else {
            out.push(part.parse().with_context(|| format!("bad cpu: {part}"))?);
        }
    }
    if out.is_empty() {
        bail!("empty cpu list");
    }
    Ok(Some(out))
}
