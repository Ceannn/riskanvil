use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Pure h2c throughput bench firehose")]
pub struct Args {
    #[arg(long)]
    pub url: String,

    #[arg(long, value_enum, default_value_t = BenchMode::Single)]
    pub mode: BenchMode,

    #[arg(long, value_enum, default_value_t = WorkloadMode::Corpus)]
    pub workload: WorkloadMode,

    #[arg(long)]
    pub payload_file: Option<String>,

    #[arg(long)]
    pub dense_file: Option<String>,

    #[arg(long)]
    pub route_meta_tsv: Option<String>,

    #[arg(long, default_value_t = 0)]
    pub dense_dim: usize,

    #[arg(long, default_value_t = 100_000)]
    pub qps: u64,

    #[arg(long, value_enum, default_value_t = SchedulerMode::SoftPace)]
    pub scheduler: SchedulerMode,

    #[arg(long, default_value_t = 20)]
    pub duration: u64,

    #[arg(long, default_value_t = 2)]
    pub warmup: u64,

    #[arg(long, default_value_t = 0)]
    pub workers: usize,

    #[arg(long)]
    pub worker_cpus: Option<String>,

    #[arg(long, default_value_t = 2)]
    pub conns_per_worker: usize,

    #[arg(long, default_value_t = 384)]
    pub max_inflight_per_conn: usize,

    #[arg(long, default_value_t = 2000)]
    pub timeout_ms: u64,

    #[arg(long, default_value_t = 256)]
    pub throughput_batch_size: usize,

    #[arg(long, default_value_t = 64)]
    pub batch_records: usize,

    #[arg(long, default_value_t = 16777216)]
    pub initial_stream_window: u32,

    #[arg(long, default_value_t = 33554432)]
    pub initial_conn_window: u32,

    #[arg(long, default_value_t = 1024)]
    pub initial_max_send_streams: usize,

    #[arg(long, default_value_t = 200)]
    pub window_ms: u64,

    #[arg(long)]
    pub window_csv: Option<PathBuf>,

    #[arg(long)]
    pub summary_json: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub progress: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, PartialEq, Eq)]
pub enum WorkloadMode {
    Corpus,
    Ceiling,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, PartialEq, Eq)]
pub enum BenchMode {
    Single,
    Batch,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, PartialEq, Eq)]
pub enum SchedulerMode {
    SoftPace,
    Burst,
}
