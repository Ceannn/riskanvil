use std::{
    collections::{HashMap, VecDeque},
    io::{ErrorKind, IoSlice, Read, Write},
    net::TcpStream as StdTcpStream,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use anyhow::Context;
use axum::http::{Method, StatusCode};
use bytes::BytesMut;
use crossbeam_channel::{bounded, Receiver, Sender};
use crossbeam_queue::ArrayQueue;
use mio::{Events, Interest, Poll, Token, Waker};
use slab::Slab;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::mpsc,
};
use tracing::{debug, info, warn};

use crate::{
    app::{BatchApp, BatchJobKind, BatchReply},
    completion_window::CompletionWindow,
    metrics::{duration_to_us, StageSampler},
    score_plane::{CompletionEntry, JobBody, ScorePlane},
    wire::{build_h1_text_reply, parse_h1_batch_request_from_buf, CompatPath, H1_BATCH_ACK_PREFIX},
};

const SHARD_WAKE_TOKEN: Token = Token(usize::MAX);
const STICKY_MAX_HEADER_BYTES: usize = 4096;
const STICKY_BODY_SLOT_LIMIT: usize = 3;

#[derive(Clone, Copy, Debug)]
pub enum CompatMode {
    StickyFastpath,
    ShardV2,
    ShardV3,
}

pub async fn run(
    listen: String,
    app: Arc<BatchApp>,
    score_plane: Arc<ScorePlane>,
    max_inflight_per_conn: usize,
    submit_burst: usize,
    write_burst: usize,
    read_budget: usize,
    completion_budget: usize,
    strict_single_inflight: bool,
    stage_sampler: Arc<StageSampler>,
    mode: CompatMode,
) -> anyhow::Result<()> {
    match mode {
        CompatMode::StickyFastpath => {
            run_sticky(
                listen,
                app,
                score_plane,
                max_inflight_per_conn,
                submit_burst,
                write_burst,
                read_budget,
                completion_budget,
                strict_single_inflight,
                stage_sampler,
            )
            .await
        }
        CompatMode::ShardV2 => {
            run_shard(
                listen,
                app,
                score_plane,
                max_inflight_per_conn,
                submit_burst,
                write_burst,
                read_budget,
                completion_budget,
                strict_single_inflight,
                stage_sampler,
            )
            .await
        }
        CompatMode::ShardV3 => {
            run_shard_v3(
                listen,
                app,
                score_plane,
                max_inflight_per_conn,
                submit_burst,
                write_burst,
                read_budget,
                completion_budget,
                strict_single_inflight,
                stage_sampler,
            )
            .await
        }
    }
}

async fn run_sticky(
    listen: String,
    _app: Arc<BatchApp>,
    score_plane: Arc<ScorePlane>,
    max_inflight_per_conn: usize,
    submit_burst: usize,
    write_burst: usize,
    _read_budget: usize,
    _completion_budget: usize,
    strict_single_inflight: bool,
    stage_sampler: Arc<StageSampler>,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&listen)
        .await
        .with_context(|| format!("bind compat listener {listen}"))?;
    if !score_plane.preserves_conn_order() {
        anyhow::bail!("compat sticky_fastpath requires sticky score routing");
    }
    info!(
        listen = %listen,
        strict_single_inflight,
        "compat HTTP sticky_fastpath listener ready"
    );

    let next_conn_id = Arc::new(AtomicU64::new(1));
    loop {
        let (stream, peer) = listener
            .accept()
            .await
            .context("accept compat connection")?;
        if let Err(err) = stream.set_nodelay(true) {
            debug!(error = %err, "failed to set TCP_NODELAY on compat connection");
        }
        let score_plane2 = score_plane.clone();
        let sampler2 = stage_sampler.clone();
        let conn_id = next_conn_id.fetch_add(1, Ordering::Relaxed);
        tokio::spawn(async move {
            if let Err(err) = handle_conn_sticky(
                conn_id,
                stream,
                score_plane2,
                max_inflight_per_conn,
                submit_burst,
                write_burst,
                strict_single_inflight,
                sampler2,
            )
            .await
            {
                debug!(conn_id, peer = %peer, error = %err, "compat sticky connection ended");
            }
        });
    }
}

async fn handle_conn_sticky(
    conn_id: u64,
    stream: TcpStream,
    score_plane: Arc<ScorePlane>,
    max_inflight_per_conn: usize,
    submit_burst: usize,
    write_burst: usize,
    strict_single_inflight: bool,
    stage_sampler: Arc<StageSampler>,
) -> anyhow::Result<()> {
    let (mut reader, mut writer) = stream.into_split();
    let inflight_limit = if strict_single_inflight {
        1
    } else {
        max_inflight_per_conn.max(1)
    };
    let (completion_tx, mut completion_rx) = mpsc::channel::<CompletionEntry>(inflight_limit);
    let mut next_seq = 0u64;
    let mut inflight = 0usize;
    let mut peer_closed = false;
    let mut header_buf = BytesMut::with_capacity(STICKY_MAX_HEADER_BYTES);
    let mut header_first_byte_at: Option<Instant> = None;
    let mut pending_header: Option<StickyPreparedRequest> = None;
    let mut active_fill: Option<StickyActiveFill> = None;
    let mut body_slot_pool = Vec::<Vec<u8>>::with_capacity(STICKY_BODY_SLOT_LIMIT);
    let mut allocated_body_slots = 0usize;

    loop {
        let mut wrote_this_turn = 0usize;
        while wrote_this_turn < write_burst {
            let mut completion = match completion_rx.try_recv() {
                Ok(completion) => completion,
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            };
            recycle_sticky_body_slot(
                &mut body_slot_pool,
                completion.recycled_body.take(),
                &mut allocated_body_slots,
            );
            let close_after = write_completion_h1(
                &mut writer,
                &stage_sampler,
                "compat_sticky",
                conn_id,
                completion,
            )
            .await?;
            inflight = inflight.saturating_sub(1);
            wrote_this_turn += 1;
            if peer_closed {
                break;
            }
            if close_after {
                peer_closed = true;
                break;
            }
        }
        if wrote_this_turn > 0 {
            continue;
        }

        if pending_header.is_none() && active_fill.is_none() {
            match try_parse_sticky_header(&mut header_buf, header_first_byte_at) {
                Ok(Some(prepared)) => {
                    header_first_byte_at = None;
                    pending_header = Some(prepared);
                }
                Ok(None) => {}
                Err(msg) => {
                    let reply = build_h1_text_reply(StatusCode::BAD_REQUEST, &msg);
                    writer
                        .write_all(&reply)
                        .await
                        .context("write compat bad-request")?;
                    peer_closed = true;
                }
            }
        }

        if active_fill.is_none() {
            if let Some(prepared) = pending_header.take() {
                if inflight < inflight_limit {
                    if let Some(body) = acquire_sticky_body_slot(
                        &mut body_slot_pool,
                        &mut allocated_body_slots,
                        prepared.content_len,
                    ) {
                        active_fill = Some(StickyActiveFill::from_prepared(prepared, body));
                    } else {
                        pending_header = Some(prepared);
                    }
                } else {
                    pending_header = Some(prepared);
                }
            }
        }

        let mut submitted_this_turn = 0usize;
        while submitted_this_turn < submit_burst {
            let Some(fill) = active_fill.take() else {
                break;
            };
            if !fill.is_complete() {
                active_fill = Some(fill);
                break;
            }
            let submit_at = Instant::now();
            let read_to_submit_us = duration_to_us(submit_at.duration_since(fill.first_byte_at));
            let first_byte_to_header_done_us =
                duration_to_us(fill.header_done_at.duration_since(fill.first_byte_at));
            let header_done_to_credit_acquired_us =
                duration_to_us(fill.credit_acquired_at.duration_since(fill.header_done_at));
            let credit_acquired_to_body_done_us =
                duration_to_us(fill.body_done_at.duration_since(fill.credit_acquired_at));
            let body_done_to_submit_us =
                duration_to_us(submit_at.duration_since(fill.body_done_at));
            let body = JobBody::Vec(fill.body);
            match score_plane.submit_tokio(
                conn_id,
                next_seq,
                fill.kind,
                body,
                fill.parse_us,
                read_to_submit_us,
                first_byte_to_header_done_us,
                header_done_to_credit_acquired_us,
                credit_acquired_to_body_done_us,
                body_done_to_submit_us,
                (inflight + 1).min(u16::MAX as usize) as u16,
                active_fill.is_some() as u8,
                0,
                completion_tx.clone(),
            ) {
                Ok(()) => {
                    inflight += 1;
                    next_seq = next_seq.wrapping_add(1);
                    submitted_this_turn += 1;
                    if !fill.carryover.is_empty() {
                        let was_empty = header_buf.is_empty();
                        header_buf.extend_from_slice(&fill.carryover);
                        if was_empty && !header_buf.is_empty() {
                            header_first_byte_at = Some(Instant::now());
                        }
                    }
                }
                Err(reply) => {
                    let close_after = write_reply_h1(&mut writer, reply).await?;
                    if close_after {
                        peer_closed = true;
                    }
                    break;
                }
            }
        }

        if inflight == 0 && peer_closed {
            break;
        }

        if active_fill.is_some() && !peer_closed {
            tokio::select! {
                completion = completion_rx.recv(), if inflight > 0 => {
                    let Some(mut completion) = completion else {
                        break;
                    };
                    recycle_sticky_body_slot(
                        &mut body_slot_pool,
                        completion.recycled_body.take(),
                        &mut allocated_body_slots,
                    );
                    let close_after = write_completion_h1(
                        &mut writer,
                        &stage_sampler,
                        "compat_sticky",
                        conn_id,
                        completion,
                    )
                    .await?;
                    inflight = inflight.saturating_sub(1);
                    if close_after {
                        peer_closed = true;
                    }
                }
                read = read_fill_direct(&mut reader, active_fill.as_mut().expect("active fill missing")) => {
                    let read = read.context("read compat sticky body")?;
                    if matches!(read, FillRead::Eof) {
                        peer_closed = true;
                    }
                }
            }
        } else if pending_header.is_none() && !peer_closed {
            tokio::select! {
                completion = completion_rx.recv(), if inflight > 0 => {
                    let Some(mut completion) = completion else {
                        break;
                    };
                    recycle_sticky_body_slot(
                        &mut body_slot_pool,
                        completion.recycled_body.take(),
                        &mut allocated_body_slots,
                    );
                    let close_after = write_completion_h1(
                        &mut writer,
                        &stage_sampler,
                        "compat_sticky",
                        conn_id,
                        completion,
                    )
                    .await?;
                    inflight = inflight.saturating_sub(1);
                    if close_after {
                        peer_closed = true;
                    }
                }
                read = reader.read_buf(&mut header_buf), if sticky_can_read_more(
                    inflight,
                    inflight_limit,
                    pending_header.is_some(),
                    active_fill.is_some(),
                    body_slot_pool.len(),
                    allocated_body_slots,
                ) => {
                    let was_empty = header_buf.is_empty();
                    let read = read.context("read compat sticky header")?;
                    if read == 0 {
                        peer_closed = true;
                    } else if was_empty {
                        header_first_byte_at = Some(Instant::now());
                    }
                }
            }
        } else if inflight > 0 {
            let Some(mut completion) = completion_rx.recv().await else {
                break;
            };
            recycle_sticky_body_slot(
                &mut body_slot_pool,
                completion.recycled_body.take(),
                &mut allocated_body_slots,
            );
            let close_after = write_completion_h1(
                &mut writer,
                &stage_sampler,
                "compat_sticky",
                conn_id,
                completion,
            )
            .await?;
            inflight = inflight.saturating_sub(1);
            if close_after {
                peer_closed = true;
            }
        } else if peer_closed {
            break;
        } else {
            break;
        }
    }

    let _ = writer.shutdown().await;
    Ok(())
}

struct StickyPreparedRequest {
    kind: BatchJobKind,
    content_len: usize,
    parse_us: u32,
    first_byte_at: Instant,
    header_done_at: Instant,
    spill: BytesMut,
}

struct StickyActiveFill {
    kind: BatchJobKind,
    content_len: usize,
    parse_us: u32,
    first_byte_at: Instant,
    header_done_at: Instant,
    credit_acquired_at: Instant,
    body_done_at: Instant,
    body: Vec<u8>,
    filled: usize,
    carryover: BytesMut,
}

impl StickyActiveFill {
    fn from_prepared(prepared: StickyPreparedRequest, mut body: Vec<u8>) -> Self {
        let credit_acquired_at = Instant::now();
        body.resize(prepared.content_len, 0);
        let initial = prepared.spill.len().min(prepared.content_len);
        if initial > 0 {
            body[..initial].copy_from_slice(&prepared.spill[..initial]);
        }
        let mut carryover = BytesMut::new();
        if prepared.spill.len() > prepared.content_len {
            carryover.extend_from_slice(&prepared.spill[prepared.content_len..]);
        }
        let body_done_at = if initial == prepared.content_len {
            credit_acquired_at
        } else {
            Instant::now()
        };
        Self {
            kind: prepared.kind,
            content_len: prepared.content_len,
            parse_us: prepared.parse_us,
            first_byte_at: prepared.first_byte_at,
            header_done_at: prepared.header_done_at,
            credit_acquired_at,
            body_done_at,
            body,
            filled: initial,
            carryover,
        }
    }

    fn is_complete(&self) -> bool {
        self.filled >= self.content_len
    }
}

fn acquire_sticky_body_slot(
    pool: &mut Vec<Vec<u8>>,
    allocated_slots: &mut usize,
    content_len: usize,
) -> Option<Vec<u8>> {
    if let Some(mut body) = pool.pop() {
        body.resize(content_len, 0);
        return Some(body);
    }
    if *allocated_slots < STICKY_BODY_SLOT_LIMIT {
        *allocated_slots += 1;
        return Some(vec![0u8; content_len]);
    }
    None
}

fn recycle_sticky_body_slot(
    pool: &mut Vec<Vec<u8>>,
    recycled_body: Option<Vec<u8>>,
    allocated_slots: &mut usize,
) {
    let Some(mut body) = recycled_body else {
        return;
    };
    if pool.len() >= STICKY_BODY_SLOT_LIMIT {
        *allocated_slots = (*allocated_slots).saturating_sub(1);
        return;
    }
    body.clear();
    pool.push(body);
}

fn sticky_can_read_more(
    _inflight: usize,
    _inflight_limit: usize,
    pending_header: bool,
    active_fill: bool,
    _pooled_slots: usize,
    _allocated_slots: usize,
) -> bool {
    !pending_header && !active_fill
}

enum FillRead {
    Progress,
    Pending,
    Eof,
}

async fn read_fill_direct(
    reader: &mut tokio::net::tcp::OwnedReadHalf,
    fill: &mut StickyActiveFill,
) -> std::io::Result<FillRead> {
    reader.readable().await?;
    let mut made_progress = false;
    loop {
        match reader.try_read(&mut fill.body[fill.filled..]) {
            Ok(0) => return Ok(FillRead::Eof),
            Ok(read) => {
                made_progress = true;
                fill.filled += read;
                if fill.filled == fill.content_len {
                    fill.body_done_at = Instant::now();
                    return Ok(FillRead::Progress);
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                return Ok(if made_progress {
                    FillRead::Progress
                } else {
                    FillRead::Pending
                });
            }
            Err(err) => return Err(err),
        }
    }
}

fn try_parse_sticky_header(
    buf: &mut BytesMut,
    first_byte_at: Option<Instant>,
) -> Result<Option<StickyPreparedRequest>, String> {
    let Some(header_end) = find_header_end_local(buf.as_ref()) else {
        if buf.len() > STICKY_MAX_HEADER_BYTES {
            return Err("request header too large".to_string());
        }
        return Ok(None);
    };

    let parse_start = Instant::now();
    let header_done_at = Instant::now();
    let first_byte_at = first_byte_at.unwrap_or(header_done_at);
    let header_bytes = &buf[..header_end];
    let request_line_end = header_bytes
        .windows(2)
        .position(|w| w == b"\r\n")
        .ok_or_else(|| "missing request line terminator".to_string())?;
    let request_line = &header_bytes[..request_line_end];
    let (method, path) = parse_request_line_local(request_line)?;
    if method != Method::POST {
        return Err("only POST is supported".to_string());
    }
    let content_len = parse_content_length_local(&header_bytes[request_line_end + 2..])?;
    let _header = buf.split_to(header_end + 4);
    let spill = buf.split();
    let kind = match path {
        CompatPath::Score => BatchJobKind::Score,
        CompatPath::Null => BatchJobKind::Null,
        CompatPath::ParseOnly => BatchJobKind::ParseOnly,
    };
    Ok(Some(StickyPreparedRequest {
        kind,
        content_len,
        parse_us: duration_to_us(parse_start.elapsed()),
        first_byte_at,
        header_done_at,
        spill,
    }))
}

fn find_header_end_local(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_request_line_local(request_line: &[u8]) -> Result<(Method, CompatPath), String> {
    let sp1 = request_line
        .iter()
        .position(|b| *b == b' ')
        .ok_or_else(|| "missing request method".to_string())?;
    let method =
        Method::from_bytes(&request_line[..sp1]).map_err(|e| format!("bad request method: {e}"))?;
    let rest = &request_line[sp1 + 1..];
    let sp2 = rest
        .iter()
        .position(|b| *b == b' ')
        .ok_or_else(|| "missing request path".to_string())?;
    let path = match &rest[..sp2] {
        b"/score_dense_f32_batch_v1" => CompatPath::Score,
        b"/score_dense_f32_batch_null_v1" => CompatPath::Null,
        b"/score_dense_f32_batch_parseonly_v1" => CompatPath::ParseOnly,
        _ => return Err("not found".to_string()),
    };
    if &rest[sp2 + 1..] != b"HTTP/1.1" {
        return Err("unsupported http version".to_string());
    }
    Ok((method, path))
}

fn parse_content_length_local(header_lines: &[u8]) -> Result<usize, String> {
    let mut start = 0usize;
    while start < header_lines.len() {
        let rem = &header_lines[start..];
        let (line, advance) = if let Some(line_end_rel) = rem.windows(2).position(|w| w == b"\r\n")
        {
            (&rem[..line_end_rel], line_end_rel + 2)
        } else {
            (rem, rem.len())
        };
        if line.is_empty() {
            break;
        }
        if line.len() >= 15 && line[..15].eq_ignore_ascii_case(b"content-length:") {
            let value = trim_ascii_ws_local(&line[15..]);
            return parse_ascii_usize_local(value);
        }
        start += advance;
    }
    Err("missing content-length".to_string())
}

fn trim_ascii_ws_local(mut buf: &[u8]) -> &[u8] {
    while let Some(first) = buf.first() {
        if !first.is_ascii_whitespace() {
            break;
        }
        buf = &buf[1..];
    }
    while let Some(last) = buf.last() {
        if !last.is_ascii_whitespace() {
            break;
        }
        buf = &buf[..buf.len() - 1];
    }
    buf
}

fn parse_ascii_usize_local(buf: &[u8]) -> Result<usize, String> {
    if buf.is_empty() {
        return Err("missing content-length".to_string());
    }
    let mut out = 0usize;
    for b in buf {
        if !b.is_ascii_digit() {
            return Err("bad content-length".to_string());
        }
        out = out
            .checked_mul(10)
            .and_then(|v| v.checked_add((b - b'0') as usize))
            .ok_or_else(|| "content-length overflow".to_string())?;
    }
    Ok(out)
}

async fn write_completion_h1(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    stage_sampler: &StageSampler,
    transport: &'static str,
    conn_id: u64,
    mut completion: CompletionEntry,
) -> anyhow::Result<bool> {
    completion.timings.completion_wait = duration_to_us(completion.completed_at.elapsed());
    let write_start = Instant::now();
    let close_after = write_reply_h1(writer, completion.reply).await?;
    completion.timings.write = duration_to_us(write_start.elapsed());
    completion.timings.total_residency = duration_to_us(completion.submitted_at.elapsed());
    stage_sampler.record(transport, conn_id, completion.seq, completion.timings);
    Ok(close_after)
}

async fn write_reply_h1(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    reply: BatchReply,
) -> anyhow::Result<bool> {
    match reply {
        Ok(ack) => {
            write_all_h1_ack_vectored(writer, &ack).await?;
            Ok(false)
        }
        Err((status, msg)) => {
            let buf = build_h1_text_reply(status, &msg);
            writer
                .write_all(&buf)
                .await
                .context("write compat h1 text reply")?;
            Ok(true)
        }
    }
}

async fn write_all_h1_ack_vectored(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    ack: &[u8; 40],
) -> anyhow::Result<()> {
    let mut prefix_off = 0usize;
    let mut ack_off = 0usize;
    while prefix_off < H1_BATCH_ACK_PREFIX.len() || ack_off < ack.len() {
        let slices = if prefix_off < H1_BATCH_ACK_PREFIX.len() {
            [
                IoSlice::new(&H1_BATCH_ACK_PREFIX[prefix_off..]),
                IoSlice::new(&ack[ack_off..]),
            ]
        } else {
            [IoSlice::new(&[]), IoSlice::new(&ack[ack_off..])]
        };
        let written = writer
            .write_vectored(&slices)
            .await
            .context("write compat h1 ack vectored")?;
        if written == 0 {
            anyhow::bail!("short write on compat h1 ack");
        }
        let prefix_rem = H1_BATCH_ACK_PREFIX.len().saturating_sub(prefix_off);
        if written < prefix_rem {
            prefix_off += written;
            continue;
        }
        prefix_off = H1_BATCH_ACK_PREFIX.len();
        ack_off += written.saturating_sub(prefix_rem);
    }
    Ok(())
}

async fn run_shard_v3(
    listen: String,
    _app: Arc<BatchApp>,
    score_plane: Arc<ScorePlane>,
    max_inflight_per_conn: usize,
    submit_burst: usize,
    write_burst: usize,
    read_budget: usize,
    completion_budget: usize,
    strict_single_inflight: bool,
    stage_sampler: Arc<StageSampler>,
) -> anyhow::Result<()> {
    if !score_plane.preserves_conn_order() {
        anyhow::bail!("compat shard-v3 requires sticky score routing");
    }
    let listener = TcpListener::bind(&listen)
        .await
        .with_context(|| format!("bind compat listener {listen}"))?;
    info!(
        listen = %listen,
        strict_single_inflight,
        "compat HTTP shard_v3 listener ready"
    );

    let shard_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
        .max(1);
    let inflight_limit = if strict_single_inflight {
        1
    } else {
        max_inflight_per_conn.max(1)
    };
    let shard_cfg = Arc::new(CompatShardV3Config {
        inflight_limit,
        read_budget: read_budget.max(1),
        submit_burst: submit_burst.max(1),
        write_burst: write_burst.max(1),
        completion_budget: completion_budget.max(1),
    });
    let shards = spawn_shards_v3(shard_count, score_plane, stage_sampler, shard_cfg)?;

    let next_conn_id = AtomicU64::new(1);
    let next_shard = AtomicUsize::new(0);
    loop {
        let (stream, peer) = listener
            .accept()
            .await
            .context("accept compat connection")?;
        if let Err(err) = stream.set_nodelay(true) {
            debug!(error = %err, "failed to set TCP_NODELAY on compat connection");
        }
        let std_stream = stream.into_std().context("convert compat stream to std")?;
        std_stream
            .set_nonblocking(true)
            .context("set compat std stream nonblocking")?;
        let conn_id = next_conn_id.fetch_add(1, Ordering::Relaxed);
        let shard_idx = next_shard.fetch_add(1, Ordering::Relaxed) % shards.len();
        let accepted = AcceptedConn {
            conn_id,
            stream: std_stream,
        };
        if let Err(err) = shards[shard_idx].accepted_tx.send(accepted) {
            warn!(conn_id, peer = %peer, error = %err, "compat shard-v3 receiver dropped");
            continue;
        }
        if let Err(err) = shards[shard_idx].waker.wake() {
            warn!(conn_id, peer = %peer, error = %err, "compat shard-v3 wake failed");
        }
    }
}

struct CompatShardV3Config {
    inflight_limit: usize,
    read_budget: usize,
    submit_burst: usize,
    write_burst: usize,
    completion_budget: usize,
}

struct CompatConnV3 {
    slot: usize,
    conn_id: u64,
    stream: mio::net::TcpStream,
    read_buf: BytesMut,
    write_queue: VecDeque<PendingWrite>,
    next_seq: u64,
    next_write_seq: u64,
    inflight: usize,
    peer_closed: bool,
    closing: bool,
    read_buf_nonempty_since: Option<Instant>,
    current_interest: Interest,
    interest_dirty: bool,
}

impl CompatConnV3 {
    fn new(slot: usize, conn_id: u64, stream: mio::net::TcpStream) -> Self {
        Self {
            slot,
            conn_id,
            stream,
            read_buf: BytesMut::with_capacity(64 * 1024),
            write_queue: VecDeque::new(),
            next_seq: 0,
            next_write_seq: 0,
            inflight: 0,
            peer_closed: false,
            closing: false,
            read_buf_nonempty_since: None,
            current_interest: Interest::READABLE,
            interest_dirty: false,
        }
    }

    fn interests(&self) -> Interest {
        if self.write_queue.is_empty() {
            Interest::READABLE
        } else {
            Interest::READABLE.add(Interest::WRITABLE)
        }
    }

    fn should_remove(&self) -> bool {
        (self.peer_closed || self.closing) && self.inflight == 0 && self.write_queue.is_empty()
    }

    fn mark_interest_dirty(&mut self, dirty_slots: &mut Vec<usize>) {
        if !self.interest_dirty {
            self.interest_dirty = true;
            dirty_slots.push(self.slot);
        }
    }

    fn enqueue_reply(
        &mut self,
        seq: u64,
        reply: BatchReply,
        timings: Option<crate::metrics::StageTimingsUs>,
        submitted_at: Option<Instant>,
        dirty_slots: &mut Vec<usize>,
    ) {
        let was_empty = self.write_queue.is_empty();
        let buf = match reply {
            Ok(ack) => {
                let mut out = Vec::with_capacity(H1_BATCH_ACK_PREFIX.len() + ack.len());
                out.extend_from_slice(H1_BATCH_ACK_PREFIX);
                out.extend_from_slice(&ack);
                out
            }
            Err((status, msg)) => {
                self.closing = true;
                build_h1_text_reply(status, &msg)
            }
        };
        self.write_queue.push_back(PendingWrite {
            seq,
            buf,
            off: 0,
            timings,
            submitted_at,
            write_started_at: None,
        });
        if was_empty {
            self.mark_interest_dirty(dirty_slots);
        }
    }

    fn enqueue_completion(
        &mut self,
        mut completion: CompletionEntry,
        dirty_slots: &mut Vec<usize>,
    ) {
        if completion.seq != self.next_write_seq {
            warn!(
                conn_id = self.conn_id,
                expected_seq = self.next_write_seq,
                got_seq = completion.seq,
                "compat shard-v3 completion order mismatch"
            );
            self.closing = true;
            return;
        }
        completion.timings.completion_wait = duration_to_us(completion.completed_at.elapsed());
        self.enqueue_reply(
            completion.seq,
            completion.reply,
            Some(completion.timings),
            Some(completion.submitted_at),
            dirty_slots,
        );
        self.next_write_seq = self.next_write_seq.wrapping_add(1);
    }
}

fn spawn_shards_v3(
    shard_count: usize,
    score_plane: Arc<ScorePlane>,
    stage_sampler: Arc<StageSampler>,
    cfg: Arc<CompatShardV3Config>,
) -> anyhow::Result<Vec<CompatShardHandle>> {
    let mut out = Vec::with_capacity(shard_count);
    for shard_idx in 0..shard_count {
        let poll = Poll::new().context("create compat shard-v3 poll")?;
        let waker = Arc::new(
            Waker::new(poll.registry(), SHARD_WAKE_TOKEN)
                .context("create compat shard-v3 waker")?,
        );
        let (accepted_tx, accepted_rx) = bounded::<AcceptedConn>(4096);
        let completion_queue = Arc::new(ArrayQueue::<CompletionEntry>::new(16384));
        let score_plane2 = score_plane.clone();
        let sampler2 = stage_sampler.clone();
        let cfg2 = cfg.clone();
        let waker2 = waker.clone();
        let completion_queue2 = completion_queue.clone();
        std::thread::Builder::new()
            .name(format!("batchplane-compat-shard-v3-{shard_idx}"))
            .spawn(move || {
                if let Err(err) = run_shard_v3_loop(
                    shard_idx,
                    poll,
                    accepted_rx,
                    completion_queue2,
                    waker2,
                    score_plane2,
                    sampler2,
                    cfg2,
                ) {
                    warn!(shard_idx, error = %err, "compat shard-v3 exited");
                }
            })
            .context("spawn compat shard-v3 thread")?;
        out.push(CompatShardHandle { accepted_tx, waker });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_shard_v3_loop(
    shard_idx: usize,
    mut poll: Poll,
    accepted_rx: Receiver<AcceptedConn>,
    completion_queue: Arc<ArrayQueue<CompletionEntry>>,
    completion_waker: Arc<Waker>,
    score_plane: Arc<ScorePlane>,
    stage_sampler: Arc<StageSampler>,
    cfg: Arc<CompatShardV3Config>,
) -> anyhow::Result<()> {
    let mut events = Events::with_capacity(1024);
    let mut conns = Slab::<CompatConnV3>::with_capacity(1024);
    let mut dirty_slots = Vec::<usize>::new();

    loop {
        while let Ok(accepted) = accepted_rx.try_recv() {
            let slot_entry = conns.vacant_entry();
            let slot = slot_entry.key();
            let token = Token(slot + 1);
            let mut stream = mio::net::TcpStream::from_std(accepted.stream);
            poll.registry()
                .register(&mut stream, token, Interest::READABLE)
                .context("register compat shard-v3 stream")?;
            slot_entry.insert(CompatConnV3::new(slot, accepted.conn_id, stream));
        }

        for _ in 0..cfg.completion_budget {
            let Some(completion) = completion_queue.pop() else {
                break;
            };
            let slot = completion.slot_token;
            let Some(conn) = conns.get_mut(slot) else {
                continue;
            };
            if conn.conn_id != completion.conn_id {
                continue;
            }
            conn.enqueue_completion(completion, &mut dirty_slots);
        }

        while let Some(slot) = dirty_slots.pop() {
            let Some(conn) = conns.get_mut(slot) else {
                continue;
            };
            if !conn.interest_dirty {
                continue;
            }
            let interests = conn.interests();
            if interests != conn.current_interest {
                poll.registry()
                    .reregister(&mut conn.stream, Token(slot + 1), interests)
                    .context("reregister compat shard-v3 interest")?;
                conn.current_interest = interests;
            }
            conn.interest_dirty = false;
        }

        poll.poll(&mut events, None).context("compat shard-v3 poll")?;

        for event in &events {
            if event.token() == SHARD_WAKE_TOKEN {
                continue;
            }
            let slot = event.token().0.saturating_sub(1);
            let Some(conn) = conns.get_mut(slot) else {
                continue;
            };
            if event.is_readable() {
                handle_readable_v3(
                    shard_idx,
                    conn,
                    &score_plane,
                    &completion_queue,
                    completion_waker.clone(),
                    &cfg,
                    &mut dirty_slots,
                );
            }
            if event.is_writable() {
                handle_writable_v3(conn, &stage_sampler, cfg.write_burst, &mut dirty_slots);
            }
        }

        let removed: Vec<usize> = conns
            .iter()
            .filter_map(|(slot, conn)| conn.should_remove().then_some(slot))
            .collect();
        for slot in removed {
            if conns.contains(slot) {
                let mut conn = conns.remove(slot);
                let _ = poll.registry().deregister(&mut conn.stream);
            }
        }
    }
}

fn handle_readable_v3(
    shard_idx: usize,
    conn: &mut CompatConnV3,
    score_plane: &ScorePlane,
    completion_queue: &Arc<ArrayQueue<CompletionEntry>>,
    completion_waker: Arc<Waker>,
    cfg: &CompatShardV3Config,
    dirty_slots: &mut Vec<usize>,
) {
    let mut tmp = [0u8; 64 * 1024];
    let mut reads = 0usize;
    while reads < cfg.read_budget {
        match conn.stream.read(&mut tmp) {
            Ok(0) => {
                conn.peer_closed = true;
                break;
            }
            Ok(n) => {
                let was_empty = conn.read_buf.is_empty();
                conn.read_buf.extend_from_slice(&tmp[..n]);
                if was_empty {
                    conn.read_buf_nonempty_since = Some(Instant::now());
                }
                reads += 1;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => {
                debug!(conn_id = conn.conn_id, shard_idx, error = %err, "compat shard-v3 read error");
                conn.closing = true;
                break;
            }
        }
    }

    let mut submitted = 0usize;
    while !conn.closing && conn.inflight < cfg.inflight_limit && submitted < cfg.submit_burst {
        let parse_start = Instant::now();
        let buffered_at = conn.read_buf_nonempty_since.unwrap_or(parse_start);
        match parse_h1_batch_request_from_buf(&mut conn.read_buf) {
            Ok(Some(req)) => {
                if req.method != Method::POST {
                    conn.enqueue_reply(
                        conn.next_seq,
                        Err((
                            StatusCode::METHOD_NOT_ALLOWED,
                            "only POST is supported".to_string(),
                        )),
                        None,
                        None,
                        dirty_slots,
                    );
                    conn.closing = true;
                    break;
                }
                let kind = match req.path {
                    CompatPath::Score => BatchJobKind::Score,
                    CompatPath::Null => BatchJobKind::Null,
                    CompatPath::ParseOnly => BatchJobKind::ParseOnly,
                };
                let parse_us = duration_to_us(parse_start.elapsed());
                let read_to_submit_us = duration_to_us(buffered_at.elapsed());
                conn.read_buf_nonempty_since = if conn.read_buf.is_empty() {
                    None
                } else {
                    Some(Instant::now())
                };
                match score_plane.submit_shard_slot(
                    conn.conn_id,
                    conn.slot,
                    conn.next_seq,
                    kind,
                    JobBody::Bytes(req.body),
                    parse_us,
                    read_to_submit_us,
                    0,
                    0,
                    0,
                    0,
                    (conn.inflight + 1).min(u16::MAX as usize) as u16,
                    0,
                    0,
                    completion_queue.clone(),
                    completion_waker.clone(),
                ) {
                    Ok(()) => {
                        conn.next_seq = conn.next_seq.wrapping_add(1);
                        conn.inflight += 1;
                        submitted += 1;
                    }
                    Err(reply) => {
                        conn.enqueue_reply(conn.next_seq, reply, None, None, dirty_slots);
                        conn.closing = true;
                        break;
                    }
                }
            }
            Ok(None) => break,
            Err(msg) => {
                conn.enqueue_reply(
                    conn.next_seq,
                    Err((StatusCode::BAD_REQUEST, msg)),
                    None,
                    None,
                    dirty_slots,
                );
                conn.closing = true;
                break;
            }
        }
    }
}

fn handle_writable_v3(
    conn: &mut CompatConnV3,
    stage_sampler: &StageSampler,
    write_burst: usize,
    dirty_slots: &mut Vec<usize>,
) {
    let mut wrote = 0usize;
    while wrote < write_burst {
        let Some(front) = conn.write_queue.front_mut() else {
            break;
        };
        if front.write_started_at.is_none() {
            front.write_started_at = Some(Instant::now());
        }
        match conn.stream.write(&front.buf[front.off..]) {
            Ok(0) => break,
            Ok(n) => {
                front.off += n;
                if front.off == front.buf.len() {
                    let finished = conn.write_queue.pop_front().expect("front write missing");
                    if conn.write_queue.is_empty() {
                        conn.mark_interest_dirty(dirty_slots);
                    }
                    if let Some(mut timings) = finished.timings {
                        timings.write = duration_to_us(
                            finished
                                .write_started_at
                                .map(|ts| ts.elapsed())
                                .unwrap_or_default(),
                        );
                        timings.total_residency = duration_to_us(
                            finished
                                .submitted_at
                                .map(|ts| ts.elapsed())
                                .unwrap_or_default(),
                        );
                        stage_sampler.record(
                            "compat_shard_v3",
                            conn.conn_id,
                            finished.seq,
                            timings,
                        );
                    }
                    conn.inflight = conn.inflight.saturating_sub(1);
                    wrote += 1;
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => {
                debug!(conn_id = conn.conn_id, error = %err, "compat shard-v3 write error");
                conn.closing = true;
                conn.write_queue.clear();
                break;
            }
        }
    }
}

async fn run_shard(
    listen: String,
    _app: Arc<BatchApp>,
    score_plane: Arc<ScorePlane>,
    max_inflight_per_conn: usize,
    submit_burst: usize,
    write_burst: usize,
    _read_budget: usize,
    _completion_budget: usize,
    strict_single_inflight: bool,
    stage_sampler: Arc<StageSampler>,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&listen)
        .await
        .with_context(|| format!("bind compat listener {listen}"))?;
    info!(
        listen = %listen,
        strict_single_inflight,
        "compat HTTP shard_v2 listener ready"
    );

    let shard_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
        .max(1);
    let inflight_limit = if strict_single_inflight {
        1
    } else {
        max_inflight_per_conn.max(1)
    };
    let shard_cfg = Arc::new(CompatShardConfig {
        inflight_limit,
        submit_burst: submit_burst.max(1),
        write_burst: write_burst.max(1),
    });
    let shards = spawn_shards(shard_count, score_plane, stage_sampler, shard_cfg)?;

    let next_conn_id = AtomicU64::new(1);
    let next_shard = AtomicUsize::new(0);
    loop {
        let (stream, peer) = listener
            .accept()
            .await
            .context("accept compat connection")?;
        if let Err(err) = stream.set_nodelay(true) {
            debug!(error = %err, "failed to set TCP_NODELAY on compat connection");
        }
        let std_stream = stream.into_std().context("convert compat stream to std")?;
        std_stream
            .set_nonblocking(true)
            .context("set compat std stream nonblocking")?;
        let conn_id = next_conn_id.fetch_add(1, Ordering::Relaxed);
        let shard_idx = next_shard.fetch_add(1, Ordering::Relaxed) % shards.len();
        let accepted = AcceptedConn {
            conn_id,
            stream: std_stream,
        };
        if let Err(err) = shards[shard_idx].accepted_tx.send(accepted) {
            warn!(conn_id, peer = %peer, error = %err, "compat shard receiver dropped");
            continue;
        }
        if let Err(err) = shards[shard_idx].waker.wake() {
            warn!(conn_id, peer = %peer, error = %err, "compat shard wake failed");
        }
    }
}

struct CompatShardHandle {
    accepted_tx: Sender<AcceptedConn>,
    waker: Arc<Waker>,
}

struct CompatShardConfig {
    inflight_limit: usize,
    submit_burst: usize,
    write_burst: usize,
}

struct AcceptedConn {
    conn_id: u64,
    stream: StdTcpStream,
}

struct PendingWrite {
    seq: u64,
    buf: Vec<u8>,
    off: usize,
    timings: Option<crate::metrics::StageTimingsUs>,
    submitted_at: Option<Instant>,
    write_started_at: Option<Instant>,
}

struct CompatConn {
    token_id: usize,
    conn_id: u64,
    stream: mio::net::TcpStream,
    read_buf: BytesMut,
    write_queue: VecDeque<PendingWrite>,
    pending_completions: CompletionWindow<CompletionEntry>,
    next_seq: u64,
    next_write_seq: u64,
    inflight: usize,
    peer_closed: bool,
    closing: bool,
    read_buf_nonempty_since: Option<Instant>,
    current_interest: Interest,
    interest_dirty: bool,
}

impl CompatConn {
    fn new(token_id: usize, conn_id: u64, stream: mio::net::TcpStream) -> Self {
        Self {
            token_id,
            conn_id,
            stream,
            read_buf: BytesMut::with_capacity(64 * 1024),
            write_queue: VecDeque::new(),
            pending_completions: CompletionWindow::new(),
            next_seq: 0,
            next_write_seq: 0,
            inflight: 0,
            peer_closed: false,
            closing: false,
            read_buf_nonempty_since: None,
            current_interest: Interest::READABLE,
            interest_dirty: false,
        }
    }

    fn interests(&self) -> Interest {
        if self.write_queue.is_empty() {
            Interest::READABLE
        } else {
            Interest::READABLE.add(Interest::WRITABLE)
        }
    }

    fn should_remove(&self) -> bool {
        (self.peer_closed || self.closing) && self.inflight == 0 && self.write_queue.is_empty()
    }

    fn enqueue_reply(
        &mut self,
        seq: u64,
        reply: BatchReply,
        timings: Option<crate::metrics::StageTimingsUs>,
        submitted_at: Option<Instant>,
    ) {
        let was_empty = self.write_queue.is_empty();
        let buf = match reply {
            Ok(ack) => {
                let mut out = Vec::with_capacity(H1_BATCH_ACK_PREFIX.len() + ack.len());
                out.extend_from_slice(H1_BATCH_ACK_PREFIX);
                out.extend_from_slice(&ack);
                out
            }
            Err((status, msg)) => {
                self.closing = true;
                build_h1_text_reply(status, &msg)
            }
        };
        self.write_queue.push_back(PendingWrite {
            seq,
            buf,
            off: 0,
            timings,
            submitted_at,
            write_started_at: None,
        });
        if was_empty {
            self.interest_dirty = true;
        }
    }

    fn enqueue_completion(&mut self, completion: CompletionEntry) {
        if completion.seq == self.next_write_seq {
            let mut timings = completion.timings;
            timings.completion_wait = duration_to_us(completion.completed_at.elapsed());
            self.enqueue_reply(
                completion.seq,
                completion.reply,
                Some(timings),
                Some(completion.submitted_at),
            );
            self.next_write_seq = self.next_write_seq.wrapping_add(1);
        } else {
            self.pending_completions.insert(completion.seq, completion);
        }
        while let Some((_, ready)) = self.pending_completions.pop_ready() {
            let mut timings = ready.timings;
            timings.completion_wait = duration_to_us(ready.completed_at.elapsed());
            self.enqueue_reply(
                ready.seq,
                ready.reply,
                Some(timings),
                Some(ready.submitted_at),
            );
            self.next_write_seq = self.next_write_seq.wrapping_add(1);
        }
    }
}

fn spawn_shards(
    shard_count: usize,
    score_plane: Arc<ScorePlane>,
    stage_sampler: Arc<StageSampler>,
    cfg: Arc<CompatShardConfig>,
) -> anyhow::Result<Vec<CompatShardHandle>> {
    let mut out = Vec::with_capacity(shard_count);
    for shard_idx in 0..shard_count {
        let poll = Poll::new().context("create compat shard poll")?;
        let waker = Arc::new(
            Waker::new(poll.registry(), SHARD_WAKE_TOKEN).context("create compat shard waker")?,
        );
        let (accepted_tx, accepted_rx) = bounded::<AcceptedConn>(4096);
        let completion_queue = Arc::new(ArrayQueue::<CompletionEntry>::new(16384));
        let score_plane2 = score_plane.clone();
        let sampler2 = stage_sampler.clone();
        let cfg2 = cfg.clone();
        let waker2 = waker.clone();
        let completion_queue2 = completion_queue.clone();
        std::thread::Builder::new()
            .name(format!("batchplane-compat-shard-{shard_idx}"))
            .spawn(move || {
                if let Err(err) = run_shard_loop(
                    shard_idx,
                    poll,
                    accepted_rx,
                    completion_queue2,
                    waker2,
                    score_plane2,
                    sampler2,
                    cfg2,
                ) {
                    warn!(shard_idx, error = %err, "compat shard exited");
                }
            })
            .context("spawn compat shard thread")?;
        out.push(CompatShardHandle { accepted_tx, waker });
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_shard_loop(
    shard_idx: usize,
    mut poll: Poll,
    accepted_rx: Receiver<AcceptedConn>,
    completion_queue: Arc<ArrayQueue<CompletionEntry>>,
    completion_waker: Arc<Waker>,
    score_plane: Arc<ScorePlane>,
    stage_sampler: Arc<StageSampler>,
    cfg: Arc<CompatShardConfig>,
) -> anyhow::Result<()> {
    let mut events = Events::with_capacity(1024);
    let mut conns = HashMap::<usize, CompatConn>::new();
    let mut conn_token_by_id = HashMap::<u64, usize>::new();
    let mut next_token = 1usize;

    loop {
        while let Ok(accepted) = accepted_rx.try_recv() {
            let token = Token(next_token);
            next_token = next_token.wrapping_add(1).max(1);
            let mut stream = mio::net::TcpStream::from_std(accepted.stream);
            poll.registry()
                .register(&mut stream, token, Interest::READABLE)
                .context("register compat shard stream")?;
            conn_token_by_id.insert(accepted.conn_id, token.0);
            conns.insert(token.0, CompatConn::new(token.0, accepted.conn_id, stream));
        }

        while let Some(completion) = completion_queue.pop() {
            if let Some(token_id) = conn_token_by_id.get(&completion.conn_id).copied() {
                if let Some(conn) = conns.get_mut(&token_id) {
                    conn.enqueue_completion(completion);
                }
            }
        }

        for conn in conns.values_mut() {
            if !conn.interest_dirty {
                continue;
            }
            let interests = conn.interests();
            if interests != conn.current_interest {
                poll.registry()
                    .reregister(&mut conn.stream, Token(conn.token_id), interests)
                    .context("reregister compat shard interest")?;
                conn.current_interest = interests;
            }
            conn.interest_dirty = false;
        }

        poll.poll(&mut events, None).context("compat shard poll")?;

        for event in &events {
            if event.token() == SHARD_WAKE_TOKEN {
                continue;
            }
            let token_id = event.token().0;
            let Some(conn) = conns.get_mut(&token_id) else {
                continue;
            };
            if event.is_readable() {
                handle_readable(
                    shard_idx,
                    conn,
                    &score_plane,
                    &completion_queue,
                    completion_waker.clone(),
                    &cfg,
                );
            }
            if event.is_writable() {
                handle_writable(conn, &stage_sampler, cfg.write_burst);
            }
        }

        let removed: Vec<(usize, u64)> = conns
            .iter()
            .filter_map(|(token_id, conn)| {
                conn.should_remove().then_some((*token_id, conn.conn_id))
            })
            .collect();
        for (token_id, conn_id) in removed {
            if let Some(mut conn) = conns.remove(&token_id) {
                let _ = poll.registry().deregister(&mut conn.stream);
            }
            conn_token_by_id.remove(&conn_id);
        }
    }
}

fn handle_readable(
    shard_idx: usize,
    conn: &mut CompatConn,
    score_plane: &ScorePlane,
    completion_queue: &Arc<ArrayQueue<CompletionEntry>>,
    completion_waker: Arc<Waker>,
    cfg: &CompatShardConfig,
) {
    let mut tmp = [0u8; 64 * 1024];
    loop {
        match conn.stream.read(&mut tmp) {
            Ok(0) => {
                conn.peer_closed = true;
                break;
            }
            Ok(n) => {
                let was_empty = conn.read_buf.is_empty();
                conn.read_buf.extend_from_slice(&tmp[..n]);
                if was_empty {
                    conn.read_buf_nonempty_since = Some(Instant::now());
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => {
                debug!(conn_id = conn.conn_id, shard_idx, error = %err, "compat shard read error");
                conn.closing = true;
                break;
            }
        }
    }

    let mut submitted = 0usize;
    while !conn.closing && conn.inflight < cfg.inflight_limit && submitted < cfg.submit_burst {
        let parse_start = Instant::now();
        let buffered_at = conn.read_buf_nonempty_since.unwrap_or(parse_start);
        match parse_h1_batch_request_from_buf(&mut conn.read_buf) {
            Ok(Some(req)) => {
                if req.method != Method::POST {
                    conn.enqueue_reply(
                        conn.next_seq,
                        Err((
                            StatusCode::METHOD_NOT_ALLOWED,
                            "only POST is supported".to_string(),
                        )),
                        None,
                        None,
                    );
                    conn.closing = true;
                    break;
                }

                let kind = match req.path {
                    CompatPath::Score => BatchJobKind::Score,
                    CompatPath::Null => BatchJobKind::Null,
                    CompatPath::ParseOnly => BatchJobKind::ParseOnly,
                };
                let parse_us = duration_to_us(parse_start.elapsed());
                let read_to_submit_us = duration_to_us(buffered_at.elapsed());
                conn.read_buf_nonempty_since = if conn.read_buf.is_empty() {
                    None
                } else {
                    Some(Instant::now())
                };
                match score_plane.submit_shard(
                    conn.conn_id,
                    conn.next_seq,
                    kind,
                    JobBody::Bytes(req.body),
                    parse_us,
                    read_to_submit_us,
                    0,
                    0,
                    0,
                    0,
                    (conn.inflight + 1).min(u16::MAX as usize) as u16,
                    0,
                    0,
                    completion_queue.clone(),
                    completion_waker.clone(),
                ) {
                    Ok(()) => {
                        conn.next_seq = conn.next_seq.wrapping_add(1);
                        conn.inflight += 1;
                        submitted += 1;
                    }
                    Err(reply) => {
                        conn.enqueue_reply(conn.next_seq, reply, None, None);
                        conn.closing = true;
                        break;
                    }
                }
            }
            Ok(None) => break,
            Err(msg) => {
                conn.enqueue_reply(
                    conn.next_seq,
                    Err((StatusCode::BAD_REQUEST, msg)),
                    None,
                    None,
                );
                conn.closing = true;
                break;
            }
        }
    }
}

fn handle_writable(conn: &mut CompatConn, stage_sampler: &StageSampler, write_burst: usize) {
    let mut wrote = 0usize;
    while wrote < write_burst {
        let Some(front) = conn.write_queue.front_mut() else {
            break;
        };
        if front.write_started_at.is_none() {
            front.write_started_at = Some(Instant::now());
        }
        match conn.stream.write(&front.buf[front.off..]) {
            Ok(0) => break,
            Ok(n) => {
                front.off += n;
                if front.off == front.buf.len() {
                    let finished = conn.write_queue.pop_front().expect("front write missing");
                    if conn.write_queue.is_empty() {
                        conn.interest_dirty = true;
                    }
                    if let Some(mut timings) = finished.timings {
                        timings.write = duration_to_us(
                            finished
                                .write_started_at
                                .map(|ts| ts.elapsed())
                                .unwrap_or_default(),
                        );
                        timings.total_residency = duration_to_us(
                            finished
                                .submitted_at
                                .map(|ts| ts.elapsed())
                                .unwrap_or_default(),
                        );
                        stage_sampler.record(
                            "compat_shard_v2",
                            conn.conn_id,
                            finished.seq,
                            timings,
                        );
                    }
                    conn.inflight = conn.inflight.saturating_sub(1);
                    wrote += 1;
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => {
                debug!(conn_id = conn.conn_id, error = %err, "compat shard write error");
                conn.closing = true;
                conn.write_queue.clear();
                break;
            }
        }
    }
}
