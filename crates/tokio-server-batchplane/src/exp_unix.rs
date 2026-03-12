use std::{
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use anyhow::Context;
use bytes::BytesMut;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream},
    sync::mpsc,
};
use tracing::{debug, info};

use crate::{
    app::{BatchApp, BatchJobKind, BatchReply},
    completion_window::CompletionWindow,
    metrics::{duration_to_us, StageSampler},
    score_plane::{CompletionEntry, JobBody, ScorePlane},
    wire::parse_unix_batch_request_from_buf,
};

pub async fn run(
    socket_path: String,
    app: Arc<BatchApp>,
    score_plane: Arc<ScorePlane>,
    max_inflight_per_conn: usize,
    submit_burst: usize,
    write_burst: usize,
    strict_single_inflight: bool,
    stage_sampler: Arc<StageSampler>,
) -> anyhow::Result<()> {
    if Path::new(&socket_path).exists() {
        std::fs::remove_file(&socket_path)
            .with_context(|| format!("remove stale unix socket {socket_path}"))?;
    }
    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("bind unix socket {socket_path}"))?;
    info!(
        socket = %socket_path,
        strict_single_inflight,
        "experimental unix listener ready"
    );

    let next_conn_id = Arc::new(AtomicU64::new(1));
    loop {
        let (stream, _) = listener.accept().await.context("accept unix connection")?;
        let score_plane2 = score_plane.clone();
        let app2 = app.clone();
        let sampler2 = stage_sampler.clone();
        let conn_id = next_conn_id.fetch_add(1, Ordering::Relaxed);
        tokio::spawn(async move {
            if let Err(err) = handle_conn(
                conn_id,
                stream,
                app2,
                score_plane2,
                max_inflight_per_conn,
                submit_burst,
                write_burst,
                strict_single_inflight,
                sampler2,
            )
            .await
            {
                debug!(conn_id, error = %err, "unix experimental connection ended");
            }
        });
    }
}

async fn handle_conn(
    conn_id: u64,
    stream: UnixStream,
    app: Arc<BatchApp>,
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
    let mut read_buf = BytesMut::with_capacity(64 * 1024);
    let preserves_conn_order = score_plane.preserves_conn_order();
    let mut next_seq = 0u64;
    let mut next_write_seq = 0u64;
    let mut inflight = 0usize;
    let mut peer_closed = false;
    let mut pending_completions = CompletionWindow::<CompletionEntry>::new();
    let mut buf_nonempty_since: Option<Instant> = None;

    loop {
        let mut wrote_this_turn = 0usize;
        while wrote_this_turn < write_burst {
            let completion = match completion_rx.try_recv() {
                Ok(completion) => completion,
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            };
            if preserves_conn_order && completion.seq == next_write_seq {
                let mut completion = completion;
                completion.timings.completion_wait =
                    duration_to_us(completion.completed_at.elapsed());
                let write_start = Instant::now();
                write_unix_reply(&mut writer, completion.reply).await?;
                completion.timings.write = duration_to_us(write_start.elapsed());
                completion.timings.total_residency =
                    duration_to_us(completion.submitted_at.elapsed());
                stage_sampler.record("unix", conn_id, completion.seq, completion.timings);
                inflight = inflight.saturating_sub(1);
                next_write_seq = next_write_seq.wrapping_add(1);
                wrote_this_turn += 1;
            } else {
                pending_completions.insert(completion.seq, completion);
            }
            while wrote_this_turn < write_burst {
                let Some((_, mut completion)) = pending_completions.pop_ready() else {
                    break;
                };
                completion.timings.completion_wait =
                    duration_to_us(completion.completed_at.elapsed());
                let write_start = Instant::now();
                write_unix_reply(&mut writer, completion.reply).await?;
                completion.timings.write = duration_to_us(write_start.elapsed());
                completion.timings.total_residency =
                    duration_to_us(completion.submitted_at.elapsed());
                stage_sampler.record("unix", conn_id, completion.seq, completion.timings);
                inflight = inflight.saturating_sub(1);
                next_write_seq = next_write_seq.wrapping_add(1);
                wrote_this_turn += 1;
            }
        }
        if wrote_this_turn > 0 {
            continue;
        }

        let mut submitted_this_turn = 0usize;

        while !peer_closed && inflight < inflight_limit && submitted_this_turn < submit_burst {
            let parse_start = Instant::now();
            let buffered_at = buf_nonempty_since.unwrap_or(parse_start);
            match parse_unix_batch_request_from_buf(&mut read_buf, app.expected_dim) {
                Ok(Some(body)) => {
                    let parse_us = duration_to_us(parse_start.elapsed());
                    let read_to_submit_us = duration_to_us(buffered_at.elapsed());
                    buf_nonempty_since = if read_buf.is_empty() {
                        None
                    } else {
                        Some(Instant::now())
                    };
                    match score_plane.submit_tokio(
                        conn_id,
                        next_seq,
                        BatchJobKind::Score,
                        JobBody::Bytes(body),
                        parse_us,
                        read_to_submit_us,
                        0,
                        0,
                        0,
                        0,
                        (inflight + 1).min(u16::MAX as usize) as u16,
                        0,
                        0,
                        completion_tx.clone(),
                    ) {
                        Ok(()) => {
                            inflight += 1;
                            next_seq = next_seq.wrapping_add(1);
                            submitted_this_turn += 1;
                            continue;
                        }
                        Err(_) => {
                            let _ = writer.shutdown().await;
                            return Ok(());
                        }
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    let _ = writer.shutdown().await;
                    return Ok(());
                }
            }
        }

        if inflight == 0 && peer_closed {
            break;
        }

        tokio::select! {
            completion = completion_rx.recv(), if inflight > 0 => {
                let Some(completion) = completion else {
                    break;
                };
                let mut writes_this_turn = 0usize;
                if preserves_conn_order && completion.seq == next_write_seq {
                    let mut completion = completion;
                    completion.timings.completion_wait =
                        duration_to_us(completion.completed_at.elapsed());
                    let write_start = Instant::now();
                    write_unix_reply(&mut writer, completion.reply).await?;
                    completion.timings.write = duration_to_us(write_start.elapsed());
                    completion.timings.total_residency =
                        duration_to_us(completion.submitted_at.elapsed());
                    stage_sampler.record("unix", conn_id, completion.seq, completion.timings);
                    inflight = inflight.saturating_sub(1);
                    next_write_seq = next_write_seq.wrapping_add(1);
                    writes_this_turn += 1;
                } else {
                    pending_completions.insert(completion.seq, completion);
                }
                while writes_this_turn < write_burst {
                    let Some((_, mut completion)) = pending_completions.pop_ready() else {
                        break;
                    };
                    completion.timings.completion_wait =
                        duration_to_us(completion.completed_at.elapsed());
                    let write_start = Instant::now();
                    write_unix_reply(&mut writer, completion.reply).await?;
                    completion.timings.write = duration_to_us(write_start.elapsed());
                    completion.timings.total_residency =
                        duration_to_us(completion.submitted_at.elapsed());
                    stage_sampler.record("unix", conn_id, completion.seq, completion.timings);
                    inflight = inflight.saturating_sub(1);
                    next_write_seq = next_write_seq.wrapping_add(1);
                    writes_this_turn += 1;
                }
            }
            read = reader.read_buf(&mut read_buf), if !peer_closed && inflight < inflight_limit => {
                let was_empty = read_buf.is_empty();
                let read = read.context("read unix batch request")?;
                if read == 0 {
                    peer_closed = true;
                } else if was_empty {
                    buf_nonempty_since = Some(Instant::now());
                }
            }
            else => break,
        }
    }

    let _ = writer.shutdown().await;
    Ok(())
}

async fn write_unix_reply(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    reply: BatchReply,
) -> anyhow::Result<()> {
    match reply {
        Ok(ack) => {
            writer
                .write_all(&ack)
                .await
                .context("write unix batch ack")?;
        }
        Err(_) => {
            writer
                .shutdown()
                .await
                .context("shutdown unix batch connection")?;
        }
    }
    Ok(())
}
