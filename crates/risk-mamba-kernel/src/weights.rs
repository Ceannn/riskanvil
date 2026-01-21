use crate::{KernelError, ModelConfig};
use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};

static PACK_ONCE: AtomicU64 = AtomicU64::new(0);
static PACK_REPEAT: AtomicU64 = AtomicU64::new(0);

pub fn weights_pack_stats() -> (u64, u64) {
    (
        PACK_ONCE.load(Ordering::Relaxed),
        PACK_REPEAT.load(Ordering::Relaxed),
    )
}

pub struct AlignedVec {
    ptr: *mut f32,
    len: usize,
    layout: Layout,
}

impl AlignedVec {
    pub fn new(len: usize, align: usize) -> Self {
        let size = len * std::mem::size_of::<f32>();
        let layout =
            Layout::from_size_align(size, align).expect("aligned layout for packed weights");
        let ptr = unsafe { alloc_zeroed(layout) } as *mut f32;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Self { ptr, len, layout }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Deref for AlignedVec {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Clone for AlignedVec {
    fn clone(&self) -> Self {
        let mut out = AlignedVec::new(self.len, self.layout.align());
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

impl fmt::Debug for AlignedVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("align", &self.layout.align())
            .finish()
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

pub struct AlignedVecU16 {
    ptr: *mut u16,
    len: usize,
    layout: Layout,
}

impl AlignedVecU16 {
    pub fn new(len: usize, align: usize) -> Self {
        let size = len * std::mem::size_of::<u16>();
        let layout =
            Layout::from_size_align(size, align).expect("aligned layout for packed weights");
        let ptr = unsafe { alloc_zeroed(layout) } as *mut u16;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Self { ptr, len, layout }
    }

    pub fn as_slice(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u16] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Deref for AlignedVecU16 {
    type Target = [u16];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedVecU16 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Clone for AlignedVecU16 {
    fn clone(&self) -> Self {
        let mut out = AlignedVecU16::new(self.len, self.layout.align());
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

impl fmt::Debug for AlignedVecU16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVecU16")
            .field("len", &self.len)
            .field("align", &self.layout.align())
            .finish()
    }
}

impl Drop for AlignedVecU16 {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights<'a> {
    pub ln1_w: &'a [f32],
    pub ln1_b: &'a [f32],
    pub ln2_w: &'a [f32],
    pub ln2_b: &'a [f32],
    pub in_proj_w: &'a [f32],
    pub in_proj_w_packed: Option<AlignedVec>,
    pub in_proj_w_packed_bf16: Option<AlignedVecU16>,
    pub in_proj_w_gamma_sum: Option<Vec<f32>>,
    pub in_proj_w_beta_sum: Option<Vec<f32>>,
    pub in_proj_b: Option<&'a [f32]>,
    pub in_proj_b_zero: Option<Vec<f32>>,
    pub conv_w: &'a [f32],
    pub conv_b: &'a [f32],
    pub conv_w_packed: Option<AlignedVec>,
    pub x_proj_w: &'a [f32],
    pub x_proj_w_packed: Option<AlignedVec>,
    pub x_proj_w_packed_bf16: Option<AlignedVecU16>,
    pub x_proj_b_zero: Option<Vec<f32>>,
    pub dt_proj_w: &'a [f32],
    pub dt_proj_w_packed: Option<AlignedVec>,
    pub dt_proj_w_packed_bf16: Option<AlignedVecU16>,
    pub dt_proj_b: &'a [f32],
    pub a_log: &'a [f32],
    pub a_pre: Option<Vec<f32>>,
    pub a_pre_v2p_16: Option<AlignedVec>,
    pub a_pre_v2p_8: Option<AlignedVec>,
    pub d: &'a [f32],
    pub out_proj_w: &'a [f32],
    pub out_proj_w_packed: Option<AlignedVec>,
    pub out_proj_w_packed_bf16: Option<AlignedVecU16>,
    pub out_proj_b: Option<&'a [f32]>,
    pub out_proj_b_zero: Option<Vec<f32>>,
    pub fc1_w: &'a [f32],
    pub fc1_w_packed: Option<AlignedVec>,
    pub fc1_w_packed_bf16: Option<AlignedVecU16>,
    pub fc1_b: &'a [f32],
    pub fc2_w: &'a [f32],
    pub fc2_w_packed: Option<AlignedVec>,
    pub fc2_w_packed_bf16: Option<AlignedVecU16>,
    pub fc2_b: &'a [f32],
}

#[derive(Debug, Clone)]
pub struct WeightsView<'a> {
    pub cfg: ModelConfig,
    pub emb_w: Option<&'a [f32]>,
    pub pos_w: Option<&'a [f32]>,
    pub static_proj_w: Option<&'a [f32]>,
    pub static_proj_b: Option<&'a [f32]>,
    pub prior_gate_w: Option<&'a [f32]>,
    pub prior_gate_b: Option<&'a [f32]>,
    pub fuse_w: Option<&'a [f32]>,
    pub fuse_b: Option<&'a [f32]>,
    pub norm_w: &'a [f32],
    pub norm_b: &'a [f32],
    pub head_w: &'a [f32],
    pub head_b: &'a [f32],
    pub head_w_packed: Option<AlignedVec>,
    pub head_w_packed_bf16: Option<AlignedVecU16>,
    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> WeightsView<'a> {
    pub fn new(
        cfg: ModelConfig,
        norm_w: &'a [f32],
        norm_b: &'a [f32],
        head_w: &'a [f32],
        head_b: &'a [f32],
        layers: Vec<LayerWeights<'a>>,
    ) -> Result<Self, KernelError> {
        if layers.len() != cfg.n_layers {
            return Err(KernelError::BadLen("layers length"));
        }
        Ok(Self {
            cfg,
            emb_w: None,
            pos_w: None,
            static_proj_w: None,
            static_proj_b: None,
            prior_gate_w: None,
            prior_gate_b: None,
            fuse_w: None,
            fuse_b: None,
            norm_w,
            norm_b,
            head_w,
            head_b,
            head_w_packed: None,
            head_w_packed_bf16: None,
            layers,
        })
    }

    pub fn precompute_a_pre(&mut self) {
        let want_v2p = {
            let from_layout = matches!(
                std::env::var("RISK_MAMBA_SSM_LAYOUT").ok().as_deref(),
                Some("v2p") | Some("panel") | Some("2")
            );
            let from_env = std::env::var("RISK_MAMBA_A_PRE_V2P")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(false);
            from_layout || from_env
        };
        for layer in &mut self.layers {
            if layer.a_pre.is_some() {
                continue;
            }
            let mut buf = Vec::with_capacity(layer.a_log.len());
            for &v in layer.a_log {
                buf.push(-libm::expf(v));
            }
            layer.a_pre = Some(buf);

            if want_v2p {
                if layer.a_pre_v2p_16.is_none() && self.cfg.d_inner % 16 == 0 {
                    layer.a_pre_v2p_16 = Some(build_a_pre_v2_panel(&self.cfg, 16, layer.a_log));
                }
                if layer.a_pre_v2p_8.is_none() && self.cfg.d_inner % 8 == 0 {
                    layer.a_pre_v2p_8 = Some(build_a_pre_v2_panel(&self.cfg, 8, layer.a_log));
                }
            }
        }
    }

    pub fn precompute_packed(&mut self, block: usize) {
        if block == 0 {
            return;
        }
        let mut did_pack = false;
        for layer in &mut self.layers {
            if layer.conv_w_packed.is_none() {
                let k = self.cfg.conv_kernel;
                if k > 0 {
                    let mut packed = AlignedVec::new(k * self.cfg.d_inner, 64);
                    let dst = packed.as_mut_slice();
                    for kk in 0..k {
                        let base = kk * self.cfg.d_inner;
                        for i in 0..self.cfg.d_inner {
                            dst[base + i] = layer.conv_w[i * k + kk];
                        }
                    }
                    layer.conv_w_packed = Some(packed);
                    did_pack = true;
                }
            }
            if layer.in_proj_w_packed.is_none() {
                let out_dim = self.cfg.d_inner * 2;
                if out_dim > 0 {
                    layer.in_proj_w_packed =
                        Some(pack_blocked(layer.in_proj_w, out_dim, self.cfg.d_model, block));
                    did_pack = true;
                }
            }
            if layer.in_proj_w_gamma_sum.is_none() || layer.in_proj_w_beta_sum.is_none() {
                let out_dim = self.cfg.d_inner * 2;
                let in_dim = self.cfg.d_model;
                let mut gamma_sum = Vec::with_capacity(out_dim);
                let mut beta_sum = Vec::with_capacity(out_dim);
                for o in 0..out_dim {
                    let row = &layer.in_proj_w[o * in_dim..][..in_dim];
                    let mut acc_gamma = 0.0f32;
                    let mut acc_beta = 0.0f32;
                    for i in 0..in_dim {
                        let w = row[i];
                        acc_gamma += w * layer.ln1_w[i];
                        acc_beta += w * layer.ln1_b[i];
                    }
                    gamma_sum.push(acc_gamma);
                    beta_sum.push(acc_beta);
                }
                layer.in_proj_w_gamma_sum = Some(gamma_sum);
                layer.in_proj_w_beta_sum = Some(beta_sum);
            }
            if layer.in_proj_b_zero.is_none() && layer.in_proj_b.is_none() {
                layer.in_proj_b_zero = Some(vec![0.0f32; self.cfg.d_inner * 2]);
            }
            if layer.x_proj_w_packed.is_none() {
                let out_dim = self.cfg.dt_rank + 2 * self.cfg.d_state_pad;
                if out_dim > 0 {
                    layer.x_proj_w_packed =
                        Some(pack_blocked(layer.x_proj_w, out_dim, self.cfg.d_inner, block));
                    did_pack = true;
                }
            }
            if layer.x_proj_b_zero.is_none() {
                layer.x_proj_b_zero = Some(vec![0.0f32; self.cfg.dt_rank + 2 * self.cfg.d_state_pad]);
            }
            if layer.dt_proj_w_packed.is_none() {
                let out_dim = self.cfg.d_inner;
                if out_dim > 0 {
                    layer.dt_proj_w_packed =
                        Some(pack_blocked(layer.dt_proj_w, out_dim, self.cfg.dt_rank, block));
                    did_pack = true;
                }
            }
            if layer.out_proj_w_packed.is_none() {
                let out_dim = self.cfg.d_model;
                if out_dim > 0 {
                    layer.out_proj_w_packed =
                        Some(pack_blocked(layer.out_proj_w, out_dim, self.cfg.d_inner, block));
                    did_pack = true;
                }
            }
            if layer.out_proj_b_zero.is_none() && layer.out_proj_b.is_none() {
                layer.out_proj_b_zero = Some(vec![0.0f32; self.cfg.d_model]);
            }
            if layer.fc1_w_packed.is_none() {
                if self.cfg.d_mlp > 0 {
                    layer.fc1_w_packed = Some(pack_blocked(
                        layer.fc1_w,
                        self.cfg.d_mlp,
                        self.cfg.d_model,
                        block,
                    ));
                    did_pack = true;
                }
            }
            if layer.fc2_w_packed.is_none() {
                if self.cfg.d_model > 0 {
                    layer.fc2_w_packed = Some(pack_blocked(
                        layer.fc2_w,
                        self.cfg.d_model,
                        self.cfg.d_mlp,
                        block,
                    ));
                    did_pack = true;
                }
            }
        }
        if self.head_w_packed.is_none() && self.cfg.n_class > 0 {
            self.head_w_packed = Some(pack_blocked(
                self.head_w,
                self.cfg.n_class,
                self.cfg.d_model,
                block,
            ));
            did_pack = true;
        }
        if did_pack {
            if PACK_ONCE
                .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
            {
                PACK_REPEAT.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn precompute_packed_bf16(&mut self, block: usize) {
        if block == 0 {
            return;
        }
        let mut did_pack = false;
        for layer in &mut self.layers {
            if layer.in_proj_w_packed_bf16.is_none() {
                let out_dim = self.cfg.d_inner * 2;
                if out_dim % block == 0 {
                    layer.in_proj_w_packed_bf16 = Some(pack_blocked_bf16(
                        layer.in_proj_w,
                        out_dim,
                        self.cfg.d_model,
                        block,
                    ));
                    did_pack = true;
                }
            }
            if layer.x_proj_w_packed_bf16.is_none() {
                let out_dim = self.cfg.dt_rank + 2 * self.cfg.d_state_pad;
                if out_dim % block == 0 {
                    layer.x_proj_w_packed_bf16 = Some(pack_blocked_bf16(
                        layer.x_proj_w,
                        out_dim,
                        self.cfg.d_inner,
                        block,
                    ));
                    did_pack = true;
                }
            }
            if layer.dt_proj_w_packed_bf16.is_none() {
                let out_dim = self.cfg.d_inner;
                if out_dim % block == 0 {
                    layer.dt_proj_w_packed_bf16 = Some(pack_blocked_bf16(
                        layer.dt_proj_w,
                        out_dim,
                        self.cfg.dt_rank,
                        block,
                    ));
                    did_pack = true;
                }
            }
            if layer.out_proj_w_packed_bf16.is_none() {
                let out_dim = self.cfg.d_model;
                if out_dim % block == 0 {
                    layer.out_proj_w_packed_bf16 = Some(pack_blocked_bf16(
                        layer.out_proj_w,
                        out_dim,
                        self.cfg.d_inner,
                        block,
                    ));
                    did_pack = true;
                }
            }
            if layer.fc1_w_packed_bf16.is_none() {
                if self.cfg.d_mlp % block == 0 {
                    layer.fc1_w_packed_bf16 = Some(pack_blocked_bf16(
                        layer.fc1_w,
                        self.cfg.d_mlp,
                        self.cfg.d_model,
                        block,
                    ));
                    did_pack = true;
                }
            }
            if layer.fc2_w_packed_bf16.is_none() {
                if self.cfg.d_model % block == 0 {
                    layer.fc2_w_packed_bf16 = Some(pack_blocked_bf16(
                        layer.fc2_w,
                        self.cfg.d_model,
                        self.cfg.d_mlp,
                        block,
                    ));
                    did_pack = true;
                }
            }
        }
        if self.head_w_packed_bf16.is_none() && self.cfg.n_class % block == 0 {
            self.head_w_packed_bf16 = Some(pack_blocked_bf16(
                self.head_w,
                self.cfg.n_class,
                self.cfg.d_model,
                block,
            ));
            did_pack = true;
        }
        if did_pack {
            if PACK_ONCE
                .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
            {
                PACK_REPEAT.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn build_a_pre_v2_panel(cfg: &ModelConfig, tile: usize, a_log: &[f32]) -> AlignedVec {
    let lane_stride = 16usize;
    let tiles = cfg.d_inner / tile;
    let len = tiles * cfg.d_state_pad * lane_stride;
    let mut out = AlignedVec::new(len, 64);
    let dst = out.as_mut_slice();
    let d_state_pad = cfg.d_state_pad;
    for ti in 0..tiles {
        let base_i = ti * tile;
        for j in 0..d_state_pad {
            let dst_base = (ti * d_state_pad + j) * lane_stride;
            let src_base = (base_i * d_state_pad) + j;
            for lane in 0..tile {
                let src = a_log[src_base + lane * d_state_pad];
                dst[dst_base + lane] = -libm::expf(src);
            }
            for lane in tile..lane_stride {
                dst[dst_base + lane] = 0.0;
            }
        }
    }
    out
}

fn pack_blocked(weight: &[f32], out_dim: usize, in_dim: usize, block: usize) -> AlignedVec {
    if block == 0 || in_dim == 0 || out_dim == 0 {
        return AlignedVec::new(0, 64);
    }
    let blocks = (out_dim + block - 1) / block;
    let mut packed = AlignedVec::new(blocks * in_dim * block, 64);
    let packed_slice = packed.as_mut_slice();
    for o in 0..out_dim {
        let blk = o / block;
        let lane = o % block;
        let base = blk * in_dim * block;
        let w_row = &weight[o * in_dim..][..in_dim];
        for k in 0..in_dim {
            packed_slice[base + k * block + lane] = w_row[k];
        }
    }
    packed
}

fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

fn pack_blocked_bf16(weight: &[f32], out_dim: usize, in_dim: usize, block: usize) -> AlignedVecU16 {
    let blocks = out_dim / block;
    let pairs = (in_dim + 1) / 2;
    let mut packed = AlignedVecU16::new(blocks * pairs * block * 2, 64);
    let packed_slice = packed.as_mut_slice();
    for blk in 0..blocks {
        for k in 0..pairs {
            let k0 = k * 2;
            let k1 = k0 + 1;
            for lane in 0..block {
                let row = blk * block + lane;
                let w0 = weight[row * in_dim + k0];
                let w1 = if k1 < in_dim { weight[row * in_dim + k1] } else { 0.0 };
                let base = blk * pairs * block * 2 + k * block * 2 + lane * 2;
                packed_slice[base] = f32_to_bf16_bits(w0);
                packed_slice[base + 1] = f32_to_bf16_bits(w1);
            }
        }
    }
    packed
}
