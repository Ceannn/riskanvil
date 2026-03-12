use axum::http::{Method, StatusCode};
use bytes::{Bytes, BytesMut};
use risk_core::quickscorer::QuickRouteMeta;

pub const H1_BATCH_ACK_PREFIX: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 40\r\nContent-Type: application/octet-stream\r\n\r\n";
pub const H1_TEXT_PREFIX_CONTENT_TYPE: &str = "text/plain; charset=utf-8";
const MAX_H1_BATCH_HEADER_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub enum CompatPath {
    Score,
    Null,
    ParseOnly,
}

#[derive(Clone, Copy, Debug)]
pub struct Batch128Shape {
    pub has_route_meta: bool,
    pub record_bytes: usize,
}

pub struct CompatRequest {
    pub path: CompatPath,
    pub method: Method,
    pub body: Bytes,
}

pub struct Batch128Refs<'a> {
    pub rows: [&'a [u8]; 128],
    pub metas: [Option<QuickRouteMeta>; 128],
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

pub fn parse_h1_batch_request_from_buf(
    buf: &mut BytesMut,
) -> Result<Option<CompatRequest>, String> {
    let Some(header_end) = find_header_end(buf.as_ref()) else {
        if buf.len() > MAX_H1_BATCH_HEADER_BYTES {
            return Err("request header too large".to_string());
        }
        return Ok(None);
    };

    let header_bytes = &buf[..header_end];
    let request_line_end = header_bytes
        .windows(2)
        .position(|w| w == b"\r\n")
        .ok_or_else(|| "missing request line terminator".to_string())?;
    let request_line = &header_bytes[..request_line_end];
    let (method, path) = parse_request_line(request_line)?;
    let content_len = parse_content_length(&header_bytes[request_line_end + 2..])?;
    let total_len = (header_end + 4)
        .checked_add(content_len)
        .ok_or_else(|| "request length overflow".to_string())?;
    if buf.len() < total_len {
        return Ok(None);
    }

    let req_bytes = buf.split_to(total_len).freeze();
    let body = req_bytes.slice((header_end + 4)..total_len);
    Ok(Some(CompatRequest { path, method, body }))
}

fn parse_request_line(request_line: &[u8]) -> Result<(Method, CompatPath), String> {
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

fn parse_content_length(header_lines: &[u8]) -> Result<usize, String> {
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
        if let Some(value) = strip_ascii_case_prefix(line, b"content-length:") {
            return parse_ascii_usize(trim_ascii_ws(value));
        }
        start += advance;
    }
    Err("missing content-length".to_string())
}

fn strip_ascii_case_prefix<'a>(line: &'a [u8], prefix: &[u8]) -> Option<&'a [u8]> {
    if line.len() < prefix.len() {
        return None;
    }
    for (lhs, rhs) in line[..prefix.len()].iter().zip(prefix.iter()) {
        if lhs.to_ascii_lowercase() != *rhs {
            return None;
        }
    }
    Some(&line[prefix.len()..])
}

fn trim_ascii_ws(mut buf: &[u8]) -> &[u8] {
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

fn parse_ascii_usize(buf: &[u8]) -> Result<usize, String> {
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

pub fn parse_unix_batch_request_from_buf(
    buf: &mut BytesMut,
    expected_dim: usize,
) -> Result<Option<Bytes>, String> {
    if buf.len() < 16 {
        return Ok(None);
    }
    if &buf[0..4] != b"RBH1" {
        return Err("bad batch magic".to_string());
    }
    let flags = u16::from_le_bytes([buf[6], buf[7]]);
    let record_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let record_bytes = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    if record_count != 128 {
        return Err(format!(
            "record_count mismatch: got {record_count} expected 128"
        ));
    }
    let has_route_meta = (flags & 1) != 0;
    let expected_record_bytes = expected_dim
        .checked_mul(4)
        .and_then(|n| n.checked_add(if has_route_meta { 40 } else { 0 }))
        .ok_or_else(|| "record_bytes overflow".to_string())?;
    if record_bytes != expected_record_bytes {
        return Err(format!(
            "record_bytes mismatch: got {record_bytes} expected {expected_record_bytes}"
        ));
    }
    let total_len = 16usize
        .checked_add(
            record_count
                .checked_mul(record_bytes)
                .ok_or_else(|| "batch length overflow".to_string())?,
        )
        .ok_or_else(|| "batch length overflow".to_string())?;
    if buf.len() < total_len {
        return Ok(None);
    }
    Ok(Some(buf.split_to(total_len).freeze()))
}

pub fn encode_batch_aggregate_ack(
    record_count: u32,
    ok_count: u32,
    used_l2_count: u32,
    decision_counts: [u32; 5],
) -> [u8; 40] {
    let mut out = [0u8; 40];
    out[0..4].copy_from_slice(b"RBA1");
    out[4..6].copy_from_slice(&1u16.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out[8..12].copy_from_slice(&record_count.to_le_bytes());
    out[12..16].copy_from_slice(&ok_count.to_le_bytes());
    out[16..20].copy_from_slice(&used_l2_count.to_le_bytes());
    for (i, count) in decision_counts.into_iter().enumerate() {
        let off = 20 + i * 4;
        out[off..off + 4].copy_from_slice(&count.to_le_bytes());
    }
    out
}

pub fn build_h1_text_reply(status: StatusCode, msg: &str) -> Vec<u8> {
    let reason = status.canonical_reason().unwrap_or("Error");
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: {}\r\n\r\n",
        status.as_u16(),
        reason,
        msg.len(),
        H1_TEXT_PREFIX_CONTENT_TYPE
    );
    let mut out = Vec::with_capacity(head.len() + msg.len());
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(msg.as_bytes());
    out
}

pub fn batch128_shape_for_content_len(
    expected_dim: usize,
    content_len: usize,
) -> Option<Batch128Shape> {
    let raw_record_bytes = expected_dim.checked_mul(4)?;
    let route_record_bytes = raw_record_bytes.checked_add(40)?;
    let raw_len = 16usize.checked_add(128usize.checked_mul(raw_record_bytes)?)?;
    if content_len == raw_len {
        return Some(Batch128Shape {
            has_route_meta: false,
            record_bytes: raw_record_bytes,
        });
    }
    let route_len = 16usize.checked_add(128usize.checked_mul(route_record_bytes)?)?;
    if content_len == route_len {
        return Some(Batch128Shape {
            has_route_meta: true,
            record_bytes: route_record_bytes,
        });
    }
    None
}

pub fn validate_batch128_header(
    body: &[u8],
    has_route_meta: bool,
    record_bytes: usize,
) -> Result<(), String> {
    if body.len() < 16 {
        return Err("body too short for batch header".to_string());
    }
    if &body[0..4] != b"RBH1" {
        return Err("bad batch magic".to_string());
    }
    let version = u16::from_le_bytes([body[4], body[5]]);
    if version != 1 {
        return Err(format!("unsupported batch version: {version}"));
    }
    let flags = u16::from_le_bytes([body[6], body[7]]);
    let expect_flags = if has_route_meta { 1u16 } else { 0u16 };
    if flags != expect_flags {
        return Err(format!(
            "batch flags mismatch: got {flags} expected {expect_flags}"
        ));
    }
    let record_count = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as usize;
    if record_count != 128 {
        return Err(format!(
            "record_count mismatch: got {record_count} expected 128"
        ));
    }
    let got_record_bytes = u32::from_le_bytes([body[12], body[13], body[14], body[15]]) as usize;
    if got_record_bytes != record_bytes {
        return Err(format!(
            "record_bytes mismatch: got {got_record_bytes} expected {record_bytes}"
        ));
    }
    Ok(())
}

pub fn parse_batch128_refs<'a>(
    body: &'a [u8],
    has_route_meta: bool,
    record_bytes: usize,
) -> Result<Batch128Refs<'a>, String> {
    let mut rows = std::array::from_fn(|_| &body[0..0]);
    let mut metas = [None; 128];
    if has_route_meta {
        for i in 0..128 {
            let off = 16 + i * record_bytes;
            let rec = &body[off..off + record_bytes];
            if rec.len() < 40 || &rec[0..4] != b"RVEC" {
                return Err("batch record missing RVEC header".to_string());
            }
            let ver = u16::from_le_bytes([rec[4], rec[5]]);
            if ver != 3 {
                return Err(format!("unsupported batch RVEC version: {ver}"));
            }
            let flags = u16::from_le_bytes([rec[6], rec[7]]);
            if flags != 0 {
                return Err(format!("unsupported batch RVEC flags: {flags}"));
            }
            let dim = u32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]) as usize;
            let payload_len = dim
                .checked_mul(4)
                .ok_or_else(|| "batch record payload overflow".to_string())?;
            if rec.len() != 40 + payload_len {
                return Err(format!(
                    "batch record len mismatch: got {} expected {}",
                    rec.len(),
                    40 + payload_len
                ));
            }
            rows[i] = &rec[40..];
            metas[i] = Some(QuickRouteMeta {
                fold_id: i32::from_le_bytes([rec[12], rec[13], rec[14], rec[15]]),
                seg_prod_amtbin: u32::from_le_bytes([rec[16], rec[17], rec[18], rec[19]]),
                transaction_id: u64::from_le_bytes([
                    rec[20], rec[21], rec[22], rec[23], rec[24], rec[25], rec[26], rec[27],
                ]),
                row_idx: u32::from_le_bytes([rec[28], rec[29], rec[30], rec[31]]),
                l2_tau_used: Some(f32::from_le_bytes([rec[32], rec[33], rec[34], rec[35]])),
            });
        }
    } else {
        for i in 0..128 {
            let off = 16 + i * record_bytes;
            rows[i] = &body[off..off + record_bytes];
        }
    }
    Ok(Batch128Refs { rows, metas })
}
