use crate::{KernelError, ModelConfig};

const SSM_LAYOUT_V2_MAX_RATIO: usize = 2;

#[derive(Debug, Clone, Copy)]
struct Range {
    start: usize,
    len: usize,
}

#[derive(Debug, Clone)]
pub struct ScratchLayout {
    pub total_f32: usize,
    x_cur: Range,
    ln1_out: Range,
    in_proj_out: Range,
    x: Range,
    z: Range,
    conv_out: Range,
    x_proj_out: Range,
    dt_in: Range,
    b_vec: Range,
    c_vec: Range,
    delta: Range,
    a_pre: Range,
    scan_out: Range,
    out_proj_out: Range,
    resid1_out: Range,
    ln2_out: Range,
    mlp_fc1: Range,
    mlp_gelu: Range,
    mlp_fc2: Range,
    resid2_out: Range,
    final_norm_out: Range,
    head_in: Range,
}

fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    let rem = value % align;
    if rem == 0 {
        value
    } else {
        value + (align - rem)
    }
}

impl ScratchLayout {
    pub fn new(cfg: &ModelConfig, batch: usize) -> Self {
        let align = 16; // 16 f32 = 64 bytes
        let mut offset = 0usize;

        let mut alloc = |len: usize| {
            offset = align_up(offset, align);
            let start = offset;
            offset += len;
            Range { start, len }
        };

        let x_cur = alloc(batch * cfg.d_model);
        let ln1_out = alloc(batch * cfg.d_model);
        let in_proj_out = alloc(batch * 2 * cfg.d_inner);
        let x = alloc(batch * cfg.d_inner);
        let z = alloc(batch * cfg.d_inner);
        let conv_out = alloc(batch * cfg.d_inner);
        let x_proj_out = alloc(batch * (cfg.dt_rank + 2 * cfg.d_state_pad));
        let dt_in = alloc(batch * cfg.dt_rank);
        let b_vec = alloc(batch * cfg.d_state_pad);
        let c_vec = alloc(batch * cfg.d_state_pad);
        let delta = alloc(batch * cfg.d_inner);
        let a_pre = alloc(cfg.d_inner * cfg.d_state_pad * SSM_LAYOUT_V2_MAX_RATIO);
        let scan_out = alloc(batch * cfg.d_inner);
        let out_proj_out = alloc(batch * cfg.d_model);
        let resid1_out = alloc(batch * cfg.d_model);
        let ln2_out = alloc(batch * cfg.d_model);
        let mlp_fc1 = alloc(batch * cfg.d_mlp);
        let mlp_gelu = alloc(batch * cfg.d_mlp);
        let mlp_fc2 = alloc(batch * cfg.d_model);
        let resid2_out = alloc(batch * cfg.d_model);
        let final_norm_out = alloc(batch * cfg.d_model);
        let head_in = alloc(batch * cfg.d_model);

        ScratchLayout {
            total_f32: offset,
            x_cur,
            ln1_out,
            in_proj_out,
            x,
            z,
            conv_out,
            x_proj_out,
            dt_in,
            b_vec,
            c_vec,
            delta,
            a_pre,
            scan_out,
            out_proj_out,
            resid1_out,
            ln2_out,
            mlp_fc1,
            mlp_gelu,
            mlp_fc2,
            resid2_out,
            final_norm_out,
            head_in,
        }
    }

}

pub struct Scratch<'a> {
    pub x_cur: &'a mut [f32],
    pub ln1_out: &'a mut [f32],
    pub in_proj_out: &'a mut [f32],
    pub x: &'a mut [f32],
    pub z: &'a mut [f32],
    pub conv_out: &'a mut [f32],
    pub x_proj_out: &'a mut [f32],
    pub dt_in: &'a mut [f32],
    pub b_vec: &'a mut [f32],
    pub c_vec: &'a mut [f32],
    pub delta: &'a mut [f32],
    pub a_pre: &'a mut [f32],
    pub scan_out: &'a mut [f32],
    pub out_proj_out: &'a mut [f32],
    pub resid1_out: &'a mut [f32],
    pub ln2_out: &'a mut [f32],
    pub mlp_fc1: &'a mut [f32],
    pub mlp_gelu: &'a mut [f32],
    pub mlp_fc2: &'a mut [f32],
    pub resid2_out: &'a mut [f32],
    pub final_norm_out: &'a mut [f32],
    pub head_in: &'a mut [f32],
}

pub fn scratch_bytes(cfg: &ModelConfig, batch: usize) -> usize {
    let layout = ScratchLayout::new(cfg, batch);
    layout.total_f32 * std::mem::size_of::<f32>() + 64
}

pub fn scratch_from_bytes<'a>(cfg: &ModelConfig, batch: usize, bytes: &'a mut [u8]) -> Result<Scratch<'a>, KernelError> {
    let layout = ScratchLayout::new(cfg, batch);
    let (prefix, aligned, _) = unsafe { bytes.align_to_mut::<f32>() };
    let required = layout.total_f32;
    if aligned.len() < required {
        return Err(KernelError::BadLen("scratch too small"));
    }
    if !prefix.is_empty() {
        // Caller should provide aligned buffer; we tolerate prefix but it reduces capacity.
    }
    let buf = &mut aligned[..required];
    let base = buf.as_mut_ptr();

    unsafe {
        Ok(Scratch {
            x_cur: std::slice::from_raw_parts_mut(base.add(layout.x_cur.start), layout.x_cur.len),
            ln1_out: std::slice::from_raw_parts_mut(base.add(layout.ln1_out.start), layout.ln1_out.len),
            in_proj_out: std::slice::from_raw_parts_mut(base.add(layout.in_proj_out.start), layout.in_proj_out.len),
            x: std::slice::from_raw_parts_mut(base.add(layout.x.start), layout.x.len),
            z: std::slice::from_raw_parts_mut(base.add(layout.z.start), layout.z.len),
            conv_out: std::slice::from_raw_parts_mut(base.add(layout.conv_out.start), layout.conv_out.len),
            x_proj_out: std::slice::from_raw_parts_mut(base.add(layout.x_proj_out.start), layout.x_proj_out.len),
            dt_in: std::slice::from_raw_parts_mut(base.add(layout.dt_in.start), layout.dt_in.len),
            b_vec: std::slice::from_raw_parts_mut(base.add(layout.b_vec.start), layout.b_vec.len),
            c_vec: std::slice::from_raw_parts_mut(base.add(layout.c_vec.start), layout.c_vec.len),
            delta: std::slice::from_raw_parts_mut(base.add(layout.delta.start), layout.delta.len),
            a_pre: std::slice::from_raw_parts_mut(base.add(layout.a_pre.start), layout.a_pre.len),
            scan_out: std::slice::from_raw_parts_mut(base.add(layout.scan_out.start), layout.scan_out.len),
            out_proj_out: std::slice::from_raw_parts_mut(base.add(layout.out_proj_out.start), layout.out_proj_out.len),
            resid1_out: std::slice::from_raw_parts_mut(base.add(layout.resid1_out.start), layout.resid1_out.len),
            ln2_out: std::slice::from_raw_parts_mut(base.add(layout.ln2_out.start), layout.ln2_out.len),
            mlp_fc1: std::slice::from_raw_parts_mut(base.add(layout.mlp_fc1.start), layout.mlp_fc1.len),
            mlp_gelu: std::slice::from_raw_parts_mut(base.add(layout.mlp_gelu.start), layout.mlp_gelu.len),
            mlp_fc2: std::slice::from_raw_parts_mut(base.add(layout.mlp_fc2.start), layout.mlp_fc2.len),
            resid2_out: std::slice::from_raw_parts_mut(base.add(layout.resid2_out.start), layout.resid2_out.len),
            final_norm_out: std::slice::from_raw_parts_mut(base.add(layout.final_norm_out.start), layout.final_norm_out.len),
            head_in: std::slice::from_raw_parts_mut(base.add(layout.head_in.start), layout.head_in.len),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ScratchFullLayout {
    pub total_f32: usize,
    x_cur: Range,
    ln1_out: Range,
    in_proj_out: Range,
    x: Range,
    z: Range,
    conv_out: Range,
    x_proj_out: Range,
    dt_in: Range,
    b_vec: Range,
    c_vec: Range,
    delta: Range,
    a_pre: Range,
    scan_out: Range,
    out_proj_out: Range,
    resid1_out: Range,
    ln2_out: Range,
    mlp_fc1: Range,
    mlp_gelu: Range,
    mlp_fc2: Range,
    resid2_out: Range,
    final_norm_out: Range,
    head_in: Range,
    layer_state: Range,
}

impl ScratchFullLayout {
    pub fn new(cfg: &ModelConfig) -> Self {
        let align = 16;
        let mut offset = 0usize;

        let mut alloc = |len: usize| {
            offset = align_up(offset, align);
            let start = offset;
            offset += len;
            Range { start, len }
        };

        let seq = cfg.seq_len;
        let x_cur = alloc(seq * cfg.d_model);
        let ln1_out = alloc(seq * cfg.d_model);
        let in_proj_out = alloc(seq * 2 * cfg.d_inner);
        let x = alloc(seq * cfg.d_inner);
        let z = alloc(seq * cfg.d_inner);
        let conv_out = alloc(seq * cfg.d_inner);
        let x_proj_out = alloc(seq * (cfg.dt_rank + 2 * cfg.d_state_pad));
        let dt_in = alloc(seq * cfg.dt_rank);
        let b_vec = alloc(seq * cfg.d_state_pad);
        let c_vec = alloc(seq * cfg.d_state_pad);
        let delta = alloc(seq * cfg.d_inner);
        let a_pre = alloc(cfg.d_inner * cfg.d_state_pad * SSM_LAYOUT_V2_MAX_RATIO);
        let scan_out = alloc(seq * cfg.d_inner);
        let out_proj_out = alloc(seq * cfg.d_model);
        let resid1_out = alloc(seq * cfg.d_model);
        let ln2_out = alloc(seq * cfg.d_model);
        let mlp_fc1 = alloc(seq * cfg.d_mlp);
        let mlp_gelu = alloc(seq * cfg.d_mlp);
        let mlp_fc2 = alloc(seq * cfg.d_model);
        let resid2_out = alloc(seq * cfg.d_model);
        let final_norm_out = alloc(seq * cfg.d_model);
        let head_in = alloc(cfg.d_model);
        let layer_state = alloc(cfg.d_inner * cfg.d_state_pad * SSM_LAYOUT_V2_MAX_RATIO);

        ScratchFullLayout {
            total_f32: offset,
            x_cur,
            ln1_out,
            in_proj_out,
            x,
            z,
            conv_out,
            x_proj_out,
            dt_in,
            b_vec,
            c_vec,
            delta,
            a_pre,
            scan_out,
            out_proj_out,
            resid1_out,
            ln2_out,
            mlp_fc1,
            mlp_gelu,
            mlp_fc2,
            resid2_out,
            final_norm_out,
            head_in,
            layer_state,
        }
    }
}

pub struct ScratchFull<'a> {
    pub x_cur: &'a mut [f32],
    pub ln1_out: &'a mut [f32],
    pub in_proj_out: &'a mut [f32],
    pub x: &'a mut [f32],
    pub z: &'a mut [f32],
    pub conv_out: &'a mut [f32],
    pub x_proj_out: &'a mut [f32],
    pub dt_in: &'a mut [f32],
    pub b_vec: &'a mut [f32],
    pub c_vec: &'a mut [f32],
    pub delta: &'a mut [f32],
    pub a_pre: &'a mut [f32],
    pub scan_out: &'a mut [f32],
    pub out_proj_out: &'a mut [f32],
    pub resid1_out: &'a mut [f32],
    pub ln2_out: &'a mut [f32],
    pub mlp_fc1: &'a mut [f32],
    pub mlp_gelu: &'a mut [f32],
    pub mlp_fc2: &'a mut [f32],
    pub resid2_out: &'a mut [f32],
    pub final_norm_out: &'a mut [f32],
    pub head_in: &'a mut [f32],
    pub layer_state: &'a mut [f32],
}

pub fn scratch_bytes_full(cfg: &ModelConfig) -> usize {
    let layout = ScratchFullLayout::new(cfg);
    layout.total_f32 * std::mem::size_of::<f32>() + 64
}

pub fn scratch_full_from_bytes<'a>(cfg: &ModelConfig, bytes: &'a mut [u8]) -> Result<ScratchFull<'a>, KernelError> {
    let layout = ScratchFullLayout::new(cfg);
    let (prefix, aligned, _) = unsafe { bytes.align_to_mut::<f32>() };
    let required = layout.total_f32;
    if aligned.len() < required {
        return Err(KernelError::BadLen("scratch too small"));
    }
    if !prefix.is_empty() {
        // Caller should provide aligned buffer; we tolerate prefix but it reduces capacity.
    }
    let buf = &mut aligned[..required];
    let base = buf.as_mut_ptr();

    unsafe {
        Ok(ScratchFull {
            x_cur: std::slice::from_raw_parts_mut(base.add(layout.x_cur.start), layout.x_cur.len),
            ln1_out: std::slice::from_raw_parts_mut(base.add(layout.ln1_out.start), layout.ln1_out.len),
            in_proj_out: std::slice::from_raw_parts_mut(base.add(layout.in_proj_out.start), layout.in_proj_out.len),
            x: std::slice::from_raw_parts_mut(base.add(layout.x.start), layout.x.len),
            z: std::slice::from_raw_parts_mut(base.add(layout.z.start), layout.z.len),
            conv_out: std::slice::from_raw_parts_mut(base.add(layout.conv_out.start), layout.conv_out.len),
            x_proj_out: std::slice::from_raw_parts_mut(base.add(layout.x_proj_out.start), layout.x_proj_out.len),
            dt_in: std::slice::from_raw_parts_mut(base.add(layout.dt_in.start), layout.dt_in.len),
            b_vec: std::slice::from_raw_parts_mut(base.add(layout.b_vec.start), layout.b_vec.len),
            c_vec: std::slice::from_raw_parts_mut(base.add(layout.c_vec.start), layout.c_vec.len),
            delta: std::slice::from_raw_parts_mut(base.add(layout.delta.start), layout.delta.len),
            a_pre: std::slice::from_raw_parts_mut(base.add(layout.a_pre.start), layout.a_pre.len),
            scan_out: std::slice::from_raw_parts_mut(base.add(layout.scan_out.start), layout.scan_out.len),
            out_proj_out: std::slice::from_raw_parts_mut(base.add(layout.out_proj_out.start), layout.out_proj_out.len),
            resid1_out: std::slice::from_raw_parts_mut(base.add(layout.resid1_out.start), layout.resid1_out.len),
            ln2_out: std::slice::from_raw_parts_mut(base.add(layout.ln2_out.start), layout.ln2_out.len),
            mlp_fc1: std::slice::from_raw_parts_mut(base.add(layout.mlp_fc1.start), layout.mlp_fc1.len),
            mlp_gelu: std::slice::from_raw_parts_mut(base.add(layout.mlp_gelu.start), layout.mlp_gelu.len),
            mlp_fc2: std::slice::from_raw_parts_mut(base.add(layout.mlp_fc2.start), layout.mlp_fc2.len),
            resid2_out: std::slice::from_raw_parts_mut(base.add(layout.resid2_out.start), layout.resid2_out.len),
            final_norm_out: std::slice::from_raw_parts_mut(base.add(layout.final_norm_out.start), layout.final_norm_out.len),
            head_in: std::slice::from_raw_parts_mut(base.add(layout.head_in.start), layout.head_in.len),
            layer_state: std::slice::from_raw_parts_mut(base.add(layout.layer_state.start), layout.layer_state.len),
        })
    }
}
