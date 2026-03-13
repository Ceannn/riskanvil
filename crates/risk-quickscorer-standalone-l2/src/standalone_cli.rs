#[derive(Parser, Debug)]
#[command(name = "rust_quickl1")]
#[command(about = "Rust L1 traversal runtime with dual-mode inference")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Infer(InferArgs),
    DispatchL2Exact(DispatchL2ExactArgs),
    QsL2Exact(QsL2ExactArgs),
    QsL2Fast(QsL2FastArgs),
    QsL2PrefixCal(QsL2PrefixCalArgs),
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum InferMode {
    RouteFast,
    MarginExact,
    RouteExactReordered,
    MarginExactReordered,
    RouteApprox,
    L2RouteExactReordered,
    L2RouteApprox,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum RankMode {
    RankPack,
    Float,
}

fn default_rank_mode() -> RankMode {
    RankMode::RankPack
}

fn default_zero_string() -> String {
    "zero".to_string()
}

fn parse_prefix_direct_kernel(raw: &str) -> Result<PrefixDirectKernel> {
    match raw {
        "qs" => Ok(PrefixDirectKernel::Qs),
        "hot_exact_96" => Ok(PrefixDirectKernel::HotExact96),
        "hot_exact_128" => Ok(PrefixDirectKernel::HotExact128),
        "hot_exact_192" => Ok(PrefixDirectKernel::HotExact192),
        "hot_exact_256" => Ok(PrefixDirectKernel::HotExact256),
        "hot_exact_384" => Ok(PrefixDirectKernel::HotExact384),
        other => bail!("unsupported direct_kernel: {}", other),
    }
}

fn prefix_direct_kernel_str(kernel: PrefixDirectKernel) -> &'static str {
    match kernel {
        PrefixDirectKernel::Qs => "qs",
        PrefixDirectKernel::HotExact96 => "hot_exact_96",
        PrefixDirectKernel::HotExact128 => "hot_exact_128",
        PrefixDirectKernel::HotExact192 => "hot_exact_192",
        PrefixDirectKernel::HotExact256 => "hot_exact_256",
        PrefixDirectKernel::HotExact384 => "hot_exact_384",
    }
}

fn parse_prefix_certifier_kind(raw: &str) -> Result<PrefixCertifierKind> {
    match raw {
        "table_v1" => Ok(PrefixCertifierKind::TableV1),
        "atlas_v1" => Ok(PrefixCertifierKind::AtlasV1),
        "mlp_v1" => Ok(PrefixCertifierKind::MlpV1),
        other => bail!("unsupported certifier_kind: {}", other),
    }
}

fn prefix_certifier_kind_str(kind: PrefixCertifierKind) -> &'static str {
    match kind {
        PrefixCertifierKind::TableV1 => "table_v1",
        PrefixCertifierKind::AtlasV1 => "atlas_v1",
        PrefixCertifierKind::MlpV1 => "mlp_v1",
    }
}

#[derive(Parser, Debug)]
struct InferArgs {
    #[arg(long)]
    soa: PathBuf,
    #[arg(long)]
    bounds: PathBuf,
    #[arg(long)]
    policy: PathBuf,
    #[arg(long)]
    model_json: Option<PathBuf>,
    #[arg(long)]
    threshold_override: Option<f32>,
    #[arg(long)]
    feat_bin: PathBuf,
    #[arg(long)]
    rank_pack: Option<PathBuf>,
    #[arg(long)]
    tree_order: Option<PathBuf>,
    #[arg(long)]
    approx_policy: Option<PathBuf>,
    #[arg(long)]
    route_meta: Option<PathBuf>,
    #[arg(long)]
    prefix16_pack: Option<PathBuf>,
    #[arg(long)]
    prefix32_pack: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "route-fast")]
    mode: InferMode,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 256)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long, default_value_t = 1)]
    bound_check_every: usize,
    #[arg(long, default_value_t = 0.0)]
    eps: f32,
    #[arg(long, default_value_t = 0.0)]
    bound_guard: f32,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    max_trees: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    fast_no_stats: bool,
}

#[derive(Parser, Debug)]
struct DispatchL2ExactArgs {
    #[arg(long)]
    feat_bin: PathBuf,
    #[arg(long)]
    dispatch_meta: PathBuf,
    #[arg(long)]
    global_model_dir: PathBuf,
    #[arg(long, default_value = "tau")]
    global_threshold_mode: String,
    #[arg(long)]
    expert_manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QsL2ExactArgs {
    #[arg(long)]
    qs_pack: PathBuf,
    #[arg(long)]
    feat_bin: PathBuf,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QsL2FastArgs {
    #[arg(long)]
    qs_pack: PathBuf,
    #[arg(long)]
    feat_bin: PathBuf,
    #[arg(long)]
    route_meta: PathBuf,
    #[arg(long)]
    soa: PathBuf,
    #[arg(long)]
    bounds: PathBuf,
    #[arg(long)]
    tree_order: PathBuf,
    #[arg(long)]
    model_json: PathBuf,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    shadow_only: bool,
}

#[derive(Parser, Debug, Clone)]
struct QsL2PrefixCalArgs {
    #[arg(long)]
    bundle_manifest: Option<PathBuf>,
    #[arg(long)]
    qs_pack: Option<PathBuf>,
    #[arg(long)]
    calibration_json: Option<PathBuf>,
    #[arg(long)]
    variant_key: Option<String>,
    #[arg(long)]
    feat_bin: Option<PathBuf>,
    #[arg(long)]
    route_meta: Option<PathBuf>,
    #[arg(long)]
    soa: Option<PathBuf>,
    #[arg(long)]
    bounds: Option<PathBuf>,
    #[arg(long)]
    tree_order: Option<PathBuf>,
    #[arg(long)]
    model_json: Option<PathBuf>,
    #[arg(long)]
    direct_kernel: Option<String>,
    #[arg(long)]
    certifier_kind: Option<String>,
    #[arg(long)]
    certifier_json: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
    #[arg(long)]
    trace_jsonl: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    shadow_only: bool,
}

