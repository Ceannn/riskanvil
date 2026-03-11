use anyhow::{anyhow, bail, Result};
use bytes::{Bytes, BytesMut};

#[derive(Clone, Copy, Debug)]
pub struct ThroughputResponse {
    pub status_code: u16,
    pub timeout: bool,
    pub rows: u32,
    pub used_l2_count: u32,
    pub decision_counts: [u32; 5],
    pub qsb2_samples: u32,
    pub rsk1_samples: u32,
}

pub async fn decode_h2_response(
    response: h2::client::ResponseFuture,
) -> Result<ThroughputResponse> {
    let response = response.await?;
    let status_code = response.status().as_u16();
    let mut body = response.into_body();
    let mut out = BytesMut::with_capacity(256);
    while let Some(chunk) = body.data().await {
        out.extend_from_slice(&chunk?);
    }
    decode_response_body(status_code, out.freeze())
}

pub fn timeout_response(rows: u32) -> ThroughputResponse {
    ThroughputResponse {
        status_code: 0,
        timeout: true,
        rows,
        used_l2_count: 0,
        decision_counts: [0, 0, 0, 0, rows],
        qsb2_samples: 0,
        rsk1_samples: 0,
    }
}

pub fn error_response(rows: u32) -> ThroughputResponse {
    ThroughputResponse {
        status_code: 0,
        timeout: false,
        rows,
        used_l2_count: 0,
        decision_counts: [0, 0, 0, 0, rows],
        qsb2_samples: 0,
        rsk1_samples: 0,
    }
}

fn decode_response_body(status_code: u16, body: Bytes) -> Result<ThroughputResponse> {
    if body.len() >= 16 && &body[0..4] == b"RBR1" {
        return decode_batch_qsb2(status_code, &body);
    }
    decode_single(status_code, &body)
}

fn decode_single(status_code: u16, buf: &[u8]) -> Result<ThroughputResponse> {
    if buf.len() == 24 && &buf[0..4] == b"QSB2" {
        let decision = buf[6];
        let used_l2 = u32::from((buf[7] & 1) != 0);
        let mut decision_counts = [0u32; 5];
        decision_counts[decision_bucket(decision)] = 1;
        return Ok(ThroughputResponse {
            status_code,
            timeout: false,
            rows: 1,
            used_l2_count: used_l2,
            decision_counts,
            qsb2_samples: 1,
            rsk1_samples: 0,
        });
    }
    if buf.len() == 48 && &buf[0..4] == b"QSB2" {
        let decision = buf[6];
        let used_l2 = u32::from((buf[7] & 1) != 0);
        let mut decision_counts = [0u32; 5];
        decision_counts[decision_bucket(decision)] = 1;
        return Ok(ThroughputResponse {
            status_code,
            timeout: false,
            rows: 1,
            used_l2_count: used_l2,
            decision_counts,
            qsb2_samples: 1,
            rsk1_samples: 0,
        });
    }
    if buf.len() == 48 && &buf[0..4] == b"RSK1" {
        let decision = buf[20];
        let flags = u16::from_le_bytes([buf[6], buf[7]]);
        let used_l2 = u32::from((flags & 1) != 0);
        let mut decision_counts = [0u32; 5];
        decision_counts[decision_bucket(decision)] = 1;
        return Ok(ThroughputResponse {
            status_code,
            timeout: false,
            rows: 1,
            used_l2_count: used_l2,
            decision_counts,
            qsb2_samples: 0,
            rsk1_samples: 1,
        });
    }
    Err(anyhow!("unsupported response body"))
}

fn decode_batch_qsb2(status_code: u16, buf: &[u8]) -> Result<ThroughputResponse> {
    if buf.len() < 16 {
        bail!("short batch response");
    }
    let version = u16::from_le_bytes([buf[4], buf[5]]);
    if version != 1 {
        bail!("unsupported batch response version {}", version);
    }
    let record_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let expect_len = 16 + record_count * 24;
    if buf.len() != expect_len {
        bail!(
            "bad batch response len: got {} expected {}",
            buf.len(),
            expect_len
        );
    }
    let mut used_l2_count = 0u32;
    let mut decision_counts = [0u32; 5];
    for i in 0..record_count {
        let off = 16 + i * 24;
        if &buf[off..off + 4] != b"QSB2" {
            bail!("batch record {} is not QSB2", i);
        }
        let decision = buf[off + 6];
        let flags = buf[off + 7];
        used_l2_count += u32::from((flags & 1) != 0);
        decision_counts[decision_bucket(decision)] += 1;
    }
    Ok(ThroughputResponse {
        status_code,
        timeout: false,
        rows: record_count as u32,
        used_l2_count,
        decision_counts,
        qsb2_samples: record_count as u32,
        rsk1_samples: 0,
    })
}

fn decision_bucket(decision: u8) -> usize {
    match decision {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        _ => 4,
    }
}
