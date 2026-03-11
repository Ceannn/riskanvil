use crate::config::{Args, BenchMode, WorkloadMode};
use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use http::Uri;
use memmap2::Mmap;
use serde::Serialize;
use std::fs::File;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Serialize)]
pub enum TransportKind {
    H2c,
}

#[derive(Clone, Debug)]
pub struct Target {
    pub transport: TransportKind,
    pub addr: SocketAddr,
    pub authority: String,
    pub path_and_query: String,
    pub full_uri: String,
}

#[derive(Clone)]
pub struct PayloadCorpus {
    mmap: Arc<Mmap>,
    pub row_bytes: usize,
    pub rows: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RouteMetaEntry {
    pub row_idx: u32,
    pub transaction_id: u64,
    pub fold_id: i32,
    pub seg_prod_amtbin: u32,
    pub l2_tau_used: f32,
}

#[derive(Clone)]
pub struct RouteMetaCorpus {
    rows: Arc<Vec<RouteMetaEntry>>,
}

#[derive(Clone)]
pub enum WorkloadSource {
    Corpus {
        payload: PayloadCorpus,
        route_meta: Option<RouteMetaCorpus>,
    },
    Ceiling {
        body: Arc<Bytes>,
    },
}

impl PayloadCorpus {
    pub fn load(path: &Path, dense_dim: usize) -> Result<Self> {
        let row_bytes = dense_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow!("dense_dim too large"))?;
        let file = File::open(path).with_context(|| format!("open dense file: {}", path.display()))?;
        let mmap = unsafe {
            Mmap::map(&file).with_context(|| format!("mmap dense file: {}", path.display()))?
        };
        if mmap.len() < row_bytes {
            bail!(
                "dense file too small: len={} row_bytes={}",
                mmap.len(),
                row_bytes
            );
        }
        if mmap.len() % row_bytes != 0 {
            bail!(
                "dense file size not multiple of row_bytes: len={} row_bytes={}",
                mmap.len(),
                row_bytes
            );
        }
        let rows = mmap.len() / row_bytes;
        Ok(Self {
            mmap: Arc::new(mmap),
            row_bytes,
            rows,
        })
    }

    #[inline]
    pub fn row_slice(&self, idx: usize) -> &[u8] {
        let i = idx % self.rows;
        let off = i * self.row_bytes;
        &self.mmap[off..off + self.row_bytes]
    }
}

impl RouteMetaCorpus {
    pub fn load(path: &Path, expected_rows: usize) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read route meta: {}", path.display()))?;
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| anyhow!("empty route_meta.tsv: {}", path.display()))?;
        let cols: Vec<&str> = header.split('\t').collect();
        let idx_row = find_col(&cols, "row_idx")?;
        let idx_tx = find_col(&cols, "TransactionID")?;
        let idx_fold = find_col(&cols, "fold_id")?;
        let idx_seg = find_col(&cols, "seg_prod_amtbin")?;
        let idx_tau = find_col(&cols, "l2_tau_used")?;
        let mut rows = vec![RouteMetaEntry::default(); expected_rows];
        let mut seen = vec![false; expected_rows];
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() <= idx_seg {
                continue;
            }
            let row_idx: usize = parts[idx_row]
                .trim()
                .parse()
                .with_context(|| format!("bad row_idx in route_meta: {}", parts[idx_row]))?;
            if row_idx >= expected_rows {
                bail!(
                    "route_meta row_idx {} out of range {}",
                    row_idx,
                    expected_rows
                );
            }
            rows[row_idx] = RouteMetaEntry {
                row_idx: row_idx as u32,
                transaction_id: parts[idx_tx].trim().parse().with_context(|| {
                    format!("bad TransactionID in route_meta: {}", parts[idx_tx])
                })?,
                fold_id: parts[idx_fold]
                    .trim()
                    .parse()
                    .with_context(|| format!("bad fold_id in route_meta: {}", parts[idx_fold]))?,
                seg_prod_amtbin: parts[idx_seg].trim().parse().with_context(|| {
                    format!("bad seg_prod_amtbin in route_meta: {}", parts[idx_seg])
                })?,
                l2_tau_used: parts[idx_tau].trim().parse().with_context(|| {
                    format!("bad l2_tau_used in route_meta: {}", parts[idx_tau])
                })?,
            };
            seen[row_idx] = true;
        }
        if let Some((idx, _)) = seen.iter().enumerate().find(|(_, ok)| !**ok) {
            bail!("route_meta missing row_idx {}", idx);
        }
        Ok(Self {
            rows: Arc::new(rows),
        })
    }

    #[inline]
    pub fn row(&self, idx: usize) -> RouteMetaEntry {
        self.rows[idx % self.rows.len()]
    }
}

impl WorkloadSource {
    pub fn load(args: &Args) -> Result<Self> {
        match args.workload {
            WorkloadMode::Corpus => {
                let dense_file = args
                    .dense_file
                    .as_deref()
                    .context("--dense-file is required when --workload corpus")?;
                if args.dense_dim == 0 {
                    bail!("--dense-dim must be > 0 when --workload corpus");
                }
                let payload = PayloadCorpus::load(Path::new(dense_file), args.dense_dim)?;
                let route_meta = if let Some(path) = args.route_meta_tsv.as_deref() {
                    Some(RouteMetaCorpus::load(Path::new(path), payload.rows)?)
                } else {
                    None
                };
                Ok(Self::Corpus { payload, route_meta })
            }
            WorkloadMode::Ceiling => {
                let payload_file = args
                    .payload_file
                    .as_deref()
                    .context("--payload-file is required when --workload ceiling")?;
                let body = std::fs::read(payload_file)
                    .with_context(|| format!("read payload file: {payload_file}"))?;
                Ok(Self::Ceiling {
                    body: Arc::new(Bytes::from(body)),
                })
            }
        }
    }

    pub fn record_len(&self) -> usize {
        match self {
            Self::Corpus { payload, route_meta } => {
                payload.row_bytes + if route_meta.is_some() { 40 } else { 0 }
            }
            Self::Ceiling { body } => body.len(),
        }
    }

    pub fn body_len(&self) -> usize {
        self.body_len_for(BenchMode::Single, 1)
    }

    pub fn body_len_for(&self, mode: BenchMode, batch_records: usize) -> usize {
        match mode {
            BenchMode::Single => self.record_len(),
            BenchMode::Batch => 16 + self.record_len() * batch_records.max(1),
        }
    }

    pub fn payload_rows(&self) -> usize {
        match self {
            Self::Corpus { payload, .. } => payload.rows,
            Self::Ceiling { .. } => 1,
        }
    }

    pub fn dense_dim(&self) -> usize {
        match self {
            Self::Corpus { payload, .. } => payload.row_bytes / 4,
            Self::Ceiling { .. } => 0,
        }
    }

    pub fn route_meta_enabled(&self) -> bool {
        matches!(self, Self::Corpus { route_meta: Some(_), .. })
    }

    pub fn build_body(&self, seq: u64, worker_id: usize) -> Bytes {
        match self {
            Self::Corpus { payload, route_meta } => {
                let row_idx = select_row(seq, worker_id, payload.rows);
                if let Some(meta) = route_meta {
                    let mut out = Vec::with_capacity(payload.row_bytes + 40);
                    out.extend_from_slice(&encode_rvec_v3_route_header(
                        meta.row(row_idx),
                        payload.row_bytes / 4,
                    ));
                    out.extend_from_slice(payload.row_slice(row_idx));
                    Bytes::from(out)
                } else {
                    Bytes::copy_from_slice(payload.row_slice(row_idx))
                }
            }
            Self::Ceiling { body } => body.as_ref().clone(),
        }
    }

    pub fn build_body_for(
        &self,
        mode: BenchMode,
        seq: u64,
        worker_id: usize,
        batch_records: usize,
    ) -> Bytes {
        match mode {
            BenchMode::Single => self.build_body(seq, worker_id),
            BenchMode::Batch => {
                let body_len = self.body_len_for(mode, batch_records);
                let mut out = vec![0u8; body_len];
                self.fill_body_for_into(mode, seq, worker_id, batch_records, &mut out)
                    .expect("fill batch body");
                Bytes::from(out)
            }
        }
    }

    pub fn fill_body_into(&self, seq: u64, worker_id: usize, dst: &mut [u8]) -> Result<()> {
        self.fill_body_for_into(BenchMode::Single, seq, worker_id, 1, dst)
    }

    pub fn fill_body_for_into(
        &self,
        mode: BenchMode,
        seq: u64,
        worker_id: usize,
        batch_records: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        match mode {
            BenchMode::Single => self.fill_single_body_into(seq, worker_id, dst),
            BenchMode::Batch => self.fill_batch_body_into(seq, worker_id, batch_records, dst),
        }
    }

    fn fill_single_body_into(&self, seq: u64, worker_id: usize, dst: &mut [u8]) -> Result<()> {
        match self {
            Self::Corpus { payload, route_meta } => {
                let expect_len = self.record_len();
                if dst.len() != expect_len {
                    bail!(
                        "fill_body_into len mismatch: got {} expected {}",
                        dst.len(),
                        expect_len
                    );
                }
                let row_idx = select_row(seq, worker_id, payload.rows);
                if let Some(meta) = route_meta {
                    let hdr = encode_rvec_v3_route_header(meta.row(row_idx), payload.row_bytes / 4);
                    dst[..40].copy_from_slice(&hdr);
                    dst[40..].copy_from_slice(payload.row_slice(row_idx));
                } else {
                    dst.copy_from_slice(payload.row_slice(row_idx));
                }
                Ok(())
            }
            Self::Ceiling { body } => {
                if dst.len() != body.len() {
                    bail!(
                        "fill_body_into len mismatch: got {} expected {}",
                        dst.len(),
                        body.len()
                    );
                }
                dst.copy_from_slice(body);
                Ok(())
            }
        }
    }

    fn fill_batch_body_into(
        &self,
        seq: u64,
        worker_id: usize,
        batch_records: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        let batch_records = batch_records.max(1);
        let record_len = self.record_len();
        let expect_len = 16 + record_len * batch_records;
        if dst.len() != expect_len {
            bail!(
                "fill_batch_body_into len mismatch: got {} expected {}",
                dst.len(),
                expect_len
            );
        }
        dst[0..4].copy_from_slice(b"RBH1");
        dst[4..6].copy_from_slice(&1u16.to_le_bytes());
        let flags = if self.route_meta_enabled() { 1u16 } else { 0u16 };
        dst[6..8].copy_from_slice(&flags.to_le_bytes());
        dst[8..12].copy_from_slice(&(batch_records as u32).to_le_bytes());
        dst[12..16].copy_from_slice(&(record_len as u32).to_le_bytes());
        for i in 0..batch_records {
            let off = 16 + i * record_len;
            self.fill_single_body_into(seq.wrapping_add(i as u64), worker_id, &mut dst[off..off + record_len])?;
        }
        Ok(())
    }
}

#[inline]
fn select_row(seq: u64, worker_id: usize, rows: usize) -> usize {
    (((seq as usize).wrapping_mul(1_315_423_911)) ^ worker_id) % rows
}

pub fn parse_target(url: &str) -> Result<Target> {
    let uri: Uri = url
        .parse()
        .with_context(|| format!("invalid --url: {url}"))?;
    let scheme = uri.scheme_str().unwrap_or("http");
    if scheme != "h2c" {
        bail!("risk-bench-h2 supports only h2c:// URLs, got scheme={scheme}");
    }
    let host = uri.host().ok_or_else(|| anyhow!("url missing host"))?;
    let port = uri.port_u16().unwrap_or(8080);
    let path_and_query = uri
        .path_and_query()
        .map(|v| v.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());
    let authority = format!("{host}:{port}");
    let full_uri = format!("http://{host}:{port}{path_and_query}");
    let addr = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolve {host}:{port}"))?
        .next()
        .ok_or_else(|| anyhow!("resolve {host}:{port}: no addresses"))?;
    Ok(Target {
        transport: TransportKind::H2c,
        addr,
        authority,
        path_and_query,
        full_uri,
    })
}

fn find_col(cols: &[&str], name: &str) -> Result<usize> {
    cols.iter()
        .position(|x| *x == name)
        .ok_or_else(|| anyhow!("missing column '{name}'"))
}

fn encode_rvec_v3_route_header(meta: RouteMetaEntry, dim: usize) -> [u8; 40] {
    let mut out = [0u8; 40];
    out[0..4].copy_from_slice(b"RVEC");
    out[4..6].copy_from_slice(&3u16.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out[8..12].copy_from_slice(&(dim as u32).to_le_bytes());
    out[12..16].copy_from_slice(&meta.fold_id.to_le_bytes());
    out[16..20].copy_from_slice(&meta.seg_prod_amtbin.to_le_bytes());
    out[20..28].copy_from_slice(&meta.transaction_id.to_le_bytes());
    out[28..32].copy_from_slice(&meta.row_idx.to_le_bytes());
    out[32..36].copy_from_slice(&meta.l2_tau_used.to_le_bytes());
    out
}
