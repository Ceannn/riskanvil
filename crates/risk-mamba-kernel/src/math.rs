use crate::{CpuDispatch, GeluKind, MathBackend, ModelConfig, SoftplusKind};
use std::sync::OnceLock;

fn use_matmul_m1_kernel() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RISK_MAMBA_MATMUL_M1")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

pub fn layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len();
    let mut mean = 0.0f32;
    for &v in input {
        mean += v;
    }
    mean /= n as f32;

    let mut var = 0.0f32;
    for &v in input {
        let d = v - mean;
        var += d * d;
    }
    var /= n as f32;

    let denom = (var + eps).sqrt();
    for i in 0..n {
        output[i] = (input[i] - mean) / denom * gamma[i] + beta[i];
    }
}

pub fn layer_norm_dispatch(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    match dispatch {
        CpuDispatch::Scalar => layer_norm(input, gamma, beta, eps, output),
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                layer_norm_avx2(input, gamma, beta, eps, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            layer_norm(input, gamma, beta, eps, output);
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                layer_norm_avx512(input, gamma, beta, eps, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            layer_norm(input, gamma, beta, eps, output);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn layer_norm_avx2(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32, output: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = input.len();
    let mut sum_v = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        sum_v = _mm256_add_ps(sum_v, x);
        i += 8;
    }
    let mut sum = hsum256_ps(sum_v);
    while i < n {
        sum += input[i];
        i += 1;
    }
    let mean = sum / n as f32;

    let mean_v = _mm256_set1_ps(mean);
    let mut var_v = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let d = _mm256_sub_ps(x, mean_v);
        var_v = _mm256_fmadd_ps(d, d, var_v);
        i += 8;
    }
    let mut var = hsum256_ps(var_v);
    while i < n {
        let d = input[i] - mean;
        var += d * d;
        i += 1;
    }
    var /= n as f32;
    let denom = (var + eps).sqrt();
    let inv = 1.0f32 / denom;
    let inv_v = _mm256_set1_ps(inv);
    i = 0;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let g = _mm256_loadu_ps(gamma.as_ptr().add(i));
        let b = _mm256_loadu_ps(beta.as_ptr().add(i));
        let y = _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(x, mean_v), inv_v), g);
        let y = _mm256_add_ps(y, b);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), y);
        i += 8;
    }
    while i < n {
        output[i] = (input[i] - mean) * inv * gamma[i] + beta[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn layer_norm_avx512(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32, output: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = input.len();
    let mut sum_v = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let x = _mm512_loadu_ps(input.as_ptr().add(i));
        sum_v = _mm512_add_ps(sum_v, x);
        i += 16;
    }
    let mut sum = hsum512_ps(sum_v);
    while i < n {
        sum += input[i];
        i += 1;
    }
    let mean = sum / n as f32;

    let mean_v = _mm512_set1_ps(mean);
    let mut var_v = _mm512_setzero_ps();
    i = 0;
    while i + 16 <= n {
        let x = _mm512_loadu_ps(input.as_ptr().add(i));
        let d = _mm512_sub_ps(x, mean_v);
        var_v = _mm512_fmadd_ps(d, d, var_v);
        i += 16;
    }
    let mut var = hsum512_ps(var_v);
    while i < n {
        let d = input[i] - mean;
        var += d * d;
        i += 1;
    }
    var /= n as f32;
    let denom = (var + eps).sqrt();
    let inv = 1.0f32 / denom;
    let inv_v = _mm512_set1_ps(inv);
    i = 0;
    while i + 16 <= n {
        let x = _mm512_loadu_ps(input.as_ptr().add(i));
        let g = _mm512_loadu_ps(gamma.as_ptr().add(i));
        let b = _mm512_loadu_ps(beta.as_ptr().add(i));
        let y = _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(x, mean_v), inv_v), g);
        let y = _mm512_add_ps(y, b);
        _mm512_storeu_ps(output.as_mut_ptr().add(i), y);
        i += 16;
    }
    while i < n {
        output[i] = (input[i] - mean) * inv * gamma[i] + beta[i];
        i += 1;
    }
}

pub fn gelu(x: f32, kind: GeluKind) -> f32 {
    match kind {
        GeluKind::Erf => {
            let inv_sqrt2 = 0.7071067811865476f32;
            0.5 * x * (1.0 + libm::erff(x * inv_sqrt2))
        }
    }
}

pub fn gelu_tanh(x: f32) -> f32 {
    let k = 0.7978845608028654f32; // sqrt(2/pi)
    let x3 = x * x * x;
    let inner = k * (x + 0.044715f32 * x3);
    0.5 * x * (1.0 + libm::tanhf(inner))
}

pub fn gelu_sigmoid_fast(x: f32) -> f32 {
    let k = 1.702f32;
    x * sigmoid_fast(k * x)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn gelu_sigmoid_fast_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let k = _mm256_set1_ps(1.702f32);
    let t = _mm256_mul_ps(k, x);
    let sig = sigmoid_fast_avx2(t);
    _mm256_mul_ps(x, sig)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub(crate) unsafe fn gelu_sigmoid_fast_avx512(
    x: std::arch::x86_64::__m512,
) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let k = _mm512_set1_ps(1.702f32);
    let t = _mm512_mul_ps(k, x);
    let sig = sigmoid_fast_avx512(t);
    _mm512_mul_ps(x, sig)
}

pub fn gelu_dispatch(x: f32, kind: GeluKind, backend: MathBackend) -> f32 {
    match backend {
        MathBackend::FastWild => gelu_sigmoid_fast(x),
        _ => gelu(x, kind),
    }
}

pub fn softplus(x: f32, kind: SoftplusKind, beta: f32, threshold: f32) -> f32 {
    match kind {
        SoftplusKind::Exact => {
            let bx = beta * x;
            if bx > threshold {
                x
            } else {
                libm::log1pf(libm::expf(bx)) / beta
            }
        }
    }
}

pub fn softplus_dispatch(
    x: f32,
    kind: SoftplusKind,
    beta: f32,
    threshold: f32,
    backend: MathBackend,
) -> f32 {
    match backend {
        MathBackend::FastWild => {
            let bx = beta * x;
            if bx > threshold {
                x
            } else {
                libm::log1pf(exp_approx_scalar(bx)) / beta
            }
        }
        _ => softplus(x, kind, beta, threshold),
    }
}

pub fn exp_approx_scalar(x: f32) -> f32 {
    let exp_hi = 88.3762626647949f32;
    let exp_lo = -88.3762626647949f32;
    let log2e = 1.44269504088896341f32;
    let ln2_hi = 0.693359375f32;
    let ln2_lo = -2.12194440e-4f32;

    let c0 = 1.9875691500e-4f32;
    let c1 = 1.3981999507e-3f32;
    let c2 = 8.3334519073e-3f32;
    let c3 = 4.1665795894e-2f32;
    let c4 = 1.6666665459e-1f32;
    let c5 = 5.0000001201e-1f32;

    let mut x = x;
    if x > exp_hi {
        x = exp_hi;
    } else if x < exp_lo {
        x = exp_lo;
    }

    let fx = x * log2e + 0.5;
    let fx_floor = libm::floorf(fx);
    let tmp = fx_floor * ln2_hi;
    let z = fx_floor * ln2_lo;
    let y = x - tmp - z;

    let mut p = c0;
    p = p * y + c1;
    p = p * y + c2;
    p = p * y + c3;
    p = p * y + c4;
    p = p * y + c5;
    p = p * y + 1.0;

    let emm0 = (fx_floor as i32) + 0x7f;
    let pow2n = f32::from_bits((emm0 as u32) << 23);
    p * pow2n
}

pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + libm::expf(-x))
    } else {
        let e = libm::expf(x);
        e / (1.0 + e)
    }
}

fn sigmoid_fast(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + exp_approx_scalar(-x))
    } else {
        let e = exp_approx_scalar(x);
        e / (1.0 + e)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn sigmoid_fast_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0f32);
    let neg = _mm256_sub_ps(zero, x);
    let exp = exp256_ps(neg);
    let denom = _mm256_add_ps(one, exp);
    _mm256_div_ps(one, denom)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub(crate) unsafe fn sigmoid_fast_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0f32);
    let neg = _mm512_sub_ps(zero, x);
    let exp = exp512_ps(neg);
    let denom = _mm512_add_ps(one, exp);
    _mm512_div_ps(one, denom)
}

#[inline]
fn exp_sleef_scalar(x: f32) -> f32 {
    // Placeholder: wire to SLEEF when available.
    let _ = x;
    libm::expf(x)
}

fn sigmoid_sleef(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + exp_sleef_scalar(-x))
    } else {
        let e = exp_sleef_scalar(x);
        e / (1.0 + e)
    }
}

pub fn sigmoid_dispatch(x: f32, backend: MathBackend) -> f32 {
    match backend {
        MathBackend::Exact => sigmoid(x),
        MathBackend::Approx | MathBackend::FastBf16 | MathBackend::FastWild => sigmoid_fast(x),
        MathBackend::Sleef => sigmoid_sleef(x),
    }
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn silu_dispatch(x: f32, backend: MathBackend) -> f32 {
    x * sigmoid_dispatch(x, backend)
}

pub fn l2norm(input: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in input {
        sum += v * v;
    }
    sum.sqrt()
}

pub fn matmul_vec(weight: &[f32], out_dim: usize, in_dim: usize, input: &[f32], bias: Option<&[f32]>, output: &mut [f32]) {
    for o in 0..out_dim {
        let mut acc = match bias {
            Some(b) => b[o],
            None => 0.0,
        };
        let w_row = &weight[o * in_dim..][..in_dim];
        for i in 0..in_dim {
            acc += w_row[i] * input[i];
        }
        output[o] = acc;
    }
}

pub fn matmul_vec_dispatch(
    weight: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    match dispatch {
        CpuDispatch::Scalar => matmul_vec(weight, out_dim, in_dim, input, bias, output),
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                matmul_vec_avx2(weight, out_dim, in_dim, input, bias, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_vec(weight, out_dim, in_dim, input, bias, output);
            }
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                matmul_vec_avx512(weight, out_dim, in_dim, input, bias, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_vec(weight, out_dim, in_dim, input, bias, output);
            }
        }
    }
}

pub fn matmul_packed16_dispatch(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    match dispatch {
        CpuDispatch::Scalar => matmul_packed16_scalar(weight_packed, out_dim, in_dim, input, bias, output),
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_matmul_m1_kernel() {
                    matmul_packed16_avx2_m1(weight_packed, out_dim, in_dim, input, bias, output);
                } else {
                    matmul_packed16_avx2(weight_packed, out_dim, in_dim, input, bias, output);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_vec(weight_packed, out_dim, in_dim, input, Some(bias), output);
            }
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_matmul_m1_kernel() {
                    matmul_packed16_avx512_m1(weight_packed, out_dim, in_dim, input, bias, output);
                } else {
                    matmul_packed16_avx512(weight_packed, out_dim, in_dim, input, bias, output);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_vec(weight_packed, out_dim, in_dim, input, Some(bias), output);
            }
        }
    }
}

pub fn matmul_packed16_dispatch_m1(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    match dispatch {
        CpuDispatch::Scalar => matmul_packed16_scalar(weight_packed, out_dim, in_dim, input, bias, output),
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                matmul_packed16_avx2_gemv_m1(weight_packed, out_dim, in_dim, input, bias, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_vec(weight_packed, out_dim, in_dim, input, Some(bias), output);
            }
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                matmul_packed16_avx512_gemv_m1(weight_packed, out_dim, in_dim, input, bias, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_vec(weight_packed, out_dim, in_dim, input, Some(bias), output);
            }
        }
    }
}

fn matmul_packed16_tail_scalar(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    tail: usize,
    output: &mut [f32],
) {
    let block = 16usize;
    for lane in 0..tail {
        let mut acc = bias[lane];
        let mut k = 0usize;
        while k < in_dim {
            acc += weight_packed[k * block + lane] * input[k];
            k += 1;
        }
        output[lane] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn matmul_packed16_tail_avx2(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    tail: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let tail_lo = tail.min(8);
    let tail_hi = tail.saturating_sub(8);
    let mut bias_lo = [0.0f32; 8];
    let mut bias_hi = [0.0f32; 8];
    if tail_lo > 0 {
        bias_lo[..tail_lo].copy_from_slice(&bias[..tail_lo]);
    }
    if tail_hi > 0 {
        bias_hi[..tail_hi].copy_from_slice(&bias[8..8 + tail_hi]);
    }
    let mut acc_lo = _mm256_loadu_ps(bias_lo.as_ptr());
    let mut acc_hi = _mm256_loadu_ps(bias_hi.as_ptr());
    let mut k = 0usize;
    while k < in_dim {
        let x = _mm256_set1_ps(*input.get_unchecked(k));
        let w_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(k * 16));
        let w_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(k * 16 + 8));
        acc_lo = _mm256_fmadd_ps(w_lo, x, acc_lo);
        acc_hi = _mm256_fmadd_ps(w_hi, x, acc_hi);
        k += 1;
    }
    let mut out_lo = [0.0f32; 8];
    let mut out_hi = [0.0f32; 8];
    _mm256_storeu_ps(out_lo.as_mut_ptr(), acc_lo);
    _mm256_storeu_ps(out_hi.as_mut_ptr(), acc_hi);
    for i in 0..tail_lo {
        output[i] = out_lo[i];
    }
    for i in 0..tail_hi {
        output[8 + i] = out_hi[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn matmul_packed16_tail_avx2_m1(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    tail: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let tail_lo = tail.min(8);
    let tail_hi = tail.saturating_sub(8);
    let mut bias_lo = [0.0f32; 8];
    let mut bias_hi = [0.0f32; 8];
    if tail_lo > 0 {
        bias_lo[..tail_lo].copy_from_slice(&bias[..tail_lo]);
    }
    if tail_hi > 0 {
        bias_hi[..tail_hi].copy_from_slice(&bias[8..8 + tail_hi]);
    }
    let mut acc_lo = _mm256_loadu_ps(bias_lo.as_ptr());
    let mut acc_hi = _mm256_loadu_ps(bias_hi.as_ptr());
    let mut k = 0usize;
    while k + 2 <= in_dim {
        let x0 = _mm256_set1_ps(*input.get_unchecked(k));
        let x1 = _mm256_set1_ps(*input.get_unchecked(k + 1));
        let w0_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(k * 16));
        let w0_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(k * 16 + 8));
        let w1_lo = _mm256_loadu_ps(weight_packed.as_ptr().add((k + 1) * 16));
        let w1_hi = _mm256_loadu_ps(weight_packed.as_ptr().add((k + 1) * 16 + 8));
        acc_lo = _mm256_fmadd_ps(w0_lo, x0, acc_lo);
        acc_hi = _mm256_fmadd_ps(w0_hi, x0, acc_hi);
        acc_lo = _mm256_fmadd_ps(w1_lo, x1, acc_lo);
        acc_hi = _mm256_fmadd_ps(w1_hi, x1, acc_hi);
        k += 2;
    }
    while k < in_dim {
        let x = _mm256_set1_ps(*input.get_unchecked(k));
        let w_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(k * 16));
        let w_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(k * 16 + 8));
        acc_lo = _mm256_fmadd_ps(w_lo, x, acc_lo);
        acc_hi = _mm256_fmadd_ps(w_hi, x, acc_hi);
        k += 1;
    }
    let mut out_lo = [0.0f32; 8];
    let mut out_hi = [0.0f32; 8];
    _mm256_storeu_ps(out_lo.as_mut_ptr(), acc_lo);
    _mm256_storeu_ps(out_hi.as_mut_ptr(), acc_hi);
    for i in 0..tail_lo {
        output[i] = out_lo[i];
    }
    for i in 0..tail_hi {
        output[8 + i] = out_hi[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn matmul_packed16_tail_avx512(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    tail: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let mask = if tail >= 16 { 0xFFFFu16 } else { (1u16 << tail) - 1 };
    let mut acc = _mm512_maskz_loadu_ps(mask, bias.as_ptr());
    for k in 0..in_dim {
        let x = _mm512_set1_ps(*input.get_unchecked(k));
        let w = _mm512_loadu_ps(weight_packed.as_ptr().add(k * 16));
        acc = _mm512_fmadd_ps(w, x, acc);
    }
    _mm512_mask_storeu_ps(output.as_mut_ptr(), mask, acc);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn matmul_packed16_tail_avx512_m1(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    tail: usize,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let mask = if tail >= 16 { 0xFFFFu16 } else { (1u16 << tail) - 1 };
    let mut acc = _mm512_maskz_loadu_ps(mask, bias.as_ptr());
    let mut k = 0usize;
    while k + 2 <= in_dim {
        let x0 = _mm512_set1_ps(*input.get_unchecked(k));
        let x1 = _mm512_set1_ps(*input.get_unchecked(k + 1));
        let w0 = _mm512_loadu_ps(weight_packed.as_ptr().add(k * 16));
        let w1 = _mm512_loadu_ps(weight_packed.as_ptr().add((k + 1) * 16));
        acc = _mm512_fmadd_ps(w0, x0, acc);
        acc = _mm512_fmadd_ps(w1, x1, acc);
        k += 2;
    }
    while k < in_dim {
        let x = _mm512_set1_ps(*input.get_unchecked(k));
        let w = _mm512_loadu_ps(weight_packed.as_ptr().add(k * 16));
        acc = _mm512_fmadd_ps(w, x, acc);
        k += 1;
    }
    _mm512_mask_storeu_ps(output.as_mut_ptr(), mask, acc);
}

pub fn matmul_packed16_tail_dispatch(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    tail: usize,
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    if tail == 0 {
        return;
    }
    match dispatch {
        CpuDispatch::Scalar => matmul_packed16_tail_scalar(weight_packed, in_dim, input, bias, tail, output),
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_matmul_m1_kernel() {
                    matmul_packed16_tail_avx2_m1(weight_packed, in_dim, input, bias, tail, output);
                } else {
                    matmul_packed16_tail_avx2(weight_packed, in_dim, input, bias, tail, output);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_packed16_tail_scalar(weight_packed, in_dim, input, bias, tail, output);
            }
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if use_matmul_m1_kernel() {
                    matmul_packed16_tail_avx512_m1(weight_packed, in_dim, input, bias, tail, output);
                } else {
                    matmul_packed16_tail_avx512(weight_packed, in_dim, input, bias, tail, output);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_packed16_tail_scalar(weight_packed, in_dim, input, bias, tail, output);
            }
        }
    }
}

pub fn matmul_packed16_tail_dispatch_m1(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    tail: usize,
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    if tail == 0 {
        return;
    }
    match dispatch {
        CpuDispatch::Scalar => matmul_packed16_tail_scalar(weight_packed, in_dim, input, bias, tail, output),
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                matmul_packed16_tail_avx2_m1(weight_packed, in_dim, input, bias, tail, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_packed16_tail_scalar(weight_packed, in_dim, input, bias, tail, output);
            }
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                matmul_packed16_tail_avx512_m1(weight_packed, in_dim, input, bias, tail, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_packed16_tail_scalar(weight_packed, in_dim, input, bias, tail, output);
            }
        }
    }
}

pub fn matmul_packed16_batch_dispatch(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    seq_len: usize,
    bias: &[f32],
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    match dispatch {
        CpuDispatch::Scalar => {
            matmul_packed16_batch_scalar(weight_packed, out_dim, in_dim, input, seq_len, bias, output);
        }
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                matmul_packed16_batch_avx2(weight_packed, out_dim, in_dim, input, seq_len, bias, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_packed16_batch_scalar(weight_packed, out_dim, in_dim, input, seq_len, bias, output);
            }
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                matmul_packed16_batch_avx512(weight_packed, out_dim, in_dim, input, seq_len, bias, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                matmul_packed16_batch_scalar(weight_packed, out_dim, in_dim, input, seq_len, bias, output);
            }
        }
    }
}

fn matmul_packed16_scalar(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    let block = 16usize;
    let blocks = out_dim / block;
    for blk in 0..blocks {
        let base = blk * in_dim * block;
        for lane in 0..block {
            let mut acc = bias[blk * block + lane];
            for k in 0..in_dim {
                acc += weight_packed[base + k * block + lane] * input[k];
            }
            output[blk * block + lane] = acc;
        }
    }
}

fn matmul_packed16_batch_scalar(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    seq_len: usize,
    bias: &[f32],
    output: &mut [f32],
) {
    let block = 16usize;
    let blocks = out_dim / block;
    for t in 0..seq_len {
        let out_t = &mut output[t * out_dim..][..out_dim];
        for blk in 0..blocks {
            let base = blk * in_dim * block;
            for lane in 0..block {
                out_t[blk * block + lane] = bias[blk * block + lane];
            }
            for k in 0..in_dim {
                let x = input[t * in_dim + k];
                for lane in 0..block {
                    out_t[blk * block + lane] += weight_packed[base + k * block + lane] * x;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    let mut tmp = [0.0f32; 8];
    std::arch::x86_64::_mm256_storeu_ps(tmp.as_mut_ptr(), v);
    tmp.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn matmul_vec_avx2(
    weight: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps};
    for o in 0..out_dim {
        let mut acc = match bias {
            Some(b) => b[o],
            None => 0.0,
        };
        let w_row = &weight[o * in_dim..][..in_dim];
        let mut v_acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= in_dim {
            let w = _mm256_loadu_ps(w_row.as_ptr().add(i));
            let x = _mm256_loadu_ps(input.as_ptr().add(i));
            v_acc = _mm256_fmadd_ps(w, x, v_acc);
            i += 8;
        }
        acc += hsum256_ps(v_acc);
        while i < in_dim {
            acc += w_row[i] * input[i];
            i += 1;
        }
        output[o] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn matmul_packed16_avx2(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let tile = 2usize;
    let mut blk = 0usize;
    while blk + tile <= blocks {
        let base0 = (blk + 0) * in_dim * block;
        let base1 = (blk + 1) * in_dim * block;
        let mut acc0_lo = _mm256_loadu_ps(bias.as_ptr().add((blk + 0) * block));
        let mut acc0_hi = _mm256_loadu_ps(bias.as_ptr().add((blk + 0) * block + 8));
        let mut acc1_lo = _mm256_loadu_ps(bias.as_ptr().add((blk + 1) * block));
        let mut acc1_hi = _mm256_loadu_ps(bias.as_ptr().add((blk + 1) * block + 8));
        for k in 0..in_dim {
            let x = _mm256_set1_ps(input[k]);
            let w0_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w0_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block + 8));
            let w1_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w1_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block + 8));
            acc0_lo = _mm256_fmadd_ps(w0_lo, x, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w0_hi, x, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w1_lo, x, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w1_hi, x, acc1_hi);
        }
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 0) * block), acc0_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 0) * block + 8), acc0_hi);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 1) * block), acc1_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 1) * block + 8), acc1_hi);
        blk += tile;
    }
    while blk < blocks {
        let base = blk * in_dim * block;
        let mut acc_lo = _mm256_loadu_ps(bias.as_ptr().add(blk * block));
        let mut acc_hi = _mm256_loadu_ps(bias.as_ptr().add(blk * block + 8));
        for k in 0..in_dim {
            let w_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
            let x = _mm256_set1_ps(input[k]);
            acc_lo = _mm256_fmadd_ps(w_lo, x, acc_lo);
            acc_hi = _mm256_fmadd_ps(w_hi, x, acc_hi);
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(blk * block), acc_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add(blk * block + 8), acc_hi);
        blk += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn matmul_packed16_avx2_m1(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let tile = 2usize;
    let mut blk = 0usize;
    while blk + tile <= blocks {
        let base0 = (blk + 0) * in_dim * block;
        let base1 = (blk + 1) * in_dim * block;
        let mut acc0_lo = _mm256_loadu_ps(bias.as_ptr().add((blk + 0) * block));
        let mut acc0_hi = _mm256_loadu_ps(bias.as_ptr().add((blk + 0) * block + 8));
        let mut acc1_lo = _mm256_loadu_ps(bias.as_ptr().add((blk + 1) * block));
        let mut acc1_hi = _mm256_loadu_ps(bias.as_ptr().add((blk + 1) * block + 8));
        let mut k = 0usize;
        while k + 2 <= in_dim {
            let x0 = _mm256_set1_ps(*input.get_unchecked(k));
            let x1 = _mm256_set1_ps(*input.get_unchecked(k + 1));
            let w00_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w00_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block + 8));
            let w01_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block));
            let w01_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block + 8));
            let w10_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w10_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block + 8));
            let w11_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block));
            let w11_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block + 8));
            acc0_lo = _mm256_fmadd_ps(w00_lo, x0, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w00_hi, x0, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w10_lo, x0, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w10_hi, x0, acc1_hi);
            acc0_lo = _mm256_fmadd_ps(w01_lo, x1, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w01_hi, x1, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w11_lo, x1, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w11_hi, x1, acc1_hi);
            k += 2;
        }
        while k < in_dim {
            let x = _mm256_set1_ps(*input.get_unchecked(k));
            let w0_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w0_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block + 8));
            let w1_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w1_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block + 8));
            acc0_lo = _mm256_fmadd_ps(w0_lo, x, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w0_hi, x, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w1_lo, x, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w1_hi, x, acc1_hi);
            k += 1;
        }
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 0) * block), acc0_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 0) * block + 8), acc0_hi);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 1) * block), acc1_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 1) * block + 8), acc1_hi);
        blk += tile;
    }
    while blk < blocks {
        let base = blk * in_dim * block;
        let mut acc_lo = _mm256_loadu_ps(bias.as_ptr().add(blk * block));
        let mut acc_hi = _mm256_loadu_ps(bias.as_ptr().add(blk * block + 8));
        let mut k = 0usize;
        while k + 2 <= in_dim {
            let x0 = _mm256_set1_ps(*input.get_unchecked(k));
            let x1 = _mm256_set1_ps(*input.get_unchecked(k + 1));
            let w0_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w0_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
            let w1_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block));
            let w1_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block + 8));
            acc_lo = _mm256_fmadd_ps(w0_lo, x0, acc_lo);
            acc_hi = _mm256_fmadd_ps(w0_hi, x0, acc_hi);
            acc_lo = _mm256_fmadd_ps(w1_lo, x1, acc_lo);
            acc_hi = _mm256_fmadd_ps(w1_hi, x1, acc_hi);
            k += 2;
        }
        while k < in_dim {
            let x = _mm256_set1_ps(*input.get_unchecked(k));
            let w_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
            acc_lo = _mm256_fmadd_ps(w_lo, x, acc_lo);
            acc_hi = _mm256_fmadd_ps(w_hi, x, acc_hi);
            k += 1;
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(blk * block), acc_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add(blk * block + 8), acc_hi);
        blk += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn matmul_packed16_avx512_m1(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let tile = 4usize;
    let mut blk = 0usize;
    while blk + tile <= blocks {
        let base0 = (blk + 0) * in_dim * block;
        let base1 = (blk + 1) * in_dim * block;
        let base2 = (blk + 2) * in_dim * block;
        let base3 = (blk + 3) * in_dim * block;
        let mut acc0 = _mm512_loadu_ps(bias.as_ptr().add((blk + 0) * block));
        let mut acc1 = _mm512_loadu_ps(bias.as_ptr().add((blk + 1) * block));
        let mut acc2 = _mm512_loadu_ps(bias.as_ptr().add((blk + 2) * block));
        let mut acc3 = _mm512_loadu_ps(bias.as_ptr().add((blk + 3) * block));
        let mut k = 0usize;
        while k + 2 <= in_dim {
            let x0 = _mm512_set1_ps(*input.get_unchecked(k));
            let x1 = _mm512_set1_ps(*input.get_unchecked(k + 1));
            let w00 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w01 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block));
            let w10 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w11 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block));
            let w20 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + k * block));
            let w21 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + (k + 1) * block));
            let w30 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + k * block));
            let w31 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + (k + 1) * block));
            acc0 = _mm512_fmadd_ps(w00, x0, acc0);
            acc1 = _mm512_fmadd_ps(w10, x0, acc1);
            acc2 = _mm512_fmadd_ps(w20, x0, acc2);
            acc3 = _mm512_fmadd_ps(w30, x0, acc3);
            acc0 = _mm512_fmadd_ps(w01, x1, acc0);
            acc1 = _mm512_fmadd_ps(w11, x1, acc1);
            acc2 = _mm512_fmadd_ps(w21, x1, acc2);
            acc3 = _mm512_fmadd_ps(w31, x1, acc3);
            k += 2;
        }
        while k < in_dim {
            let x = _mm512_set1_ps(*input.get_unchecked(k));
            let w0 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w1 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w2 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + k * block));
            let w3 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + k * block));
            acc0 = _mm512_fmadd_ps(w0, x, acc0);
            acc1 = _mm512_fmadd_ps(w1, x, acc1);
            acc2 = _mm512_fmadd_ps(w2, x, acc2);
            acc3 = _mm512_fmadd_ps(w3, x, acc3);
            k += 1;
        }
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 0) * block), acc0);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 1) * block), acc1);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 2) * block), acc2);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 3) * block), acc3);
        blk += tile;
    }
    while blk < blocks {
        let base = blk * in_dim * block;
        let mut acc = _mm512_loadu_ps(bias.as_ptr().add(blk * block));
        let mut k = 0usize;
        while k + 2 <= in_dim {
            let x0 = _mm512_set1_ps(*input.get_unchecked(k));
            let x1 = _mm512_set1_ps(*input.get_unchecked(k + 1));
            let w0 = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w1 = _mm512_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block));
            acc = _mm512_fmadd_ps(w0, x0, acc);
            acc = _mm512_fmadd_ps(w1, x1, acc);
            k += 2;
        }
        while k < in_dim {
            let x = _mm512_set1_ps(*input.get_unchecked(k));
            let w = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            acc = _mm512_fmadd_ps(w, x, acc);
            k += 1;
        }
        _mm512_storeu_ps(output.as_mut_ptr().add(blk * block), acc);
        blk += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn matmul_packed16_avx2_gemv_m1(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let tile = 2usize;
    let mut blk = 0usize;
    while blk + tile <= blocks {
        let base0 = (blk + 0) * in_dim * block;
        let base1 = (blk + 1) * in_dim * block;
        let mut acc0_lo = _mm256_loadu_ps(bias.as_ptr().add((blk + 0) * block));
        let mut acc0_hi = _mm256_loadu_ps(bias.as_ptr().add((blk + 0) * block + 8));
        let mut acc1_lo = _mm256_loadu_ps(bias.as_ptr().add((blk + 1) * block));
        let mut acc1_hi = _mm256_loadu_ps(bias.as_ptr().add((blk + 1) * block + 8));
        let mut k = 0usize;
        while k + 4 <= in_dim {
            let x0 = _mm256_set1_ps(*input.get_unchecked(k));
            let x1 = _mm256_set1_ps(*input.get_unchecked(k + 1));
            let x2 = _mm256_set1_ps(*input.get_unchecked(k + 2));
            let x3 = _mm256_set1_ps(*input.get_unchecked(k + 3));
            let w00_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w00_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block + 8));
            let w01_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block));
            let w01_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block + 8));
            let w02_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 2) * block));
            let w02_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 2) * block + 8));
            let w03_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 3) * block));
            let w03_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 3) * block + 8));
            let w10_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w10_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block + 8));
            let w11_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block));
            let w11_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block + 8));
            let w12_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 2) * block));
            let w12_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 2) * block + 8));
            let w13_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 3) * block));
            let w13_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 3) * block + 8));
            acc0_lo = _mm256_fmadd_ps(w00_lo, x0, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w00_hi, x0, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w10_lo, x0, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w10_hi, x0, acc1_hi);
            acc0_lo = _mm256_fmadd_ps(w01_lo, x1, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w01_hi, x1, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w11_lo, x1, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w11_hi, x1, acc1_hi);
            acc0_lo = _mm256_fmadd_ps(w02_lo, x2, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w02_hi, x2, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w12_lo, x2, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w12_hi, x2, acc1_hi);
            acc0_lo = _mm256_fmadd_ps(w03_lo, x3, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w03_hi, x3, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w13_lo, x3, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w13_hi, x3, acc1_hi);
            k += 4;
        }
        while k + 2 <= in_dim {
            let x0 = _mm256_set1_ps(*input.get_unchecked(k));
            let x1 = _mm256_set1_ps(*input.get_unchecked(k + 1));
            let w00_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w00_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block + 8));
            let w01_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block));
            let w01_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block + 8));
            let w10_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w10_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block + 8));
            let w11_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block));
            let w11_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block + 8));
            acc0_lo = _mm256_fmadd_ps(w00_lo, x0, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w00_hi, x0, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w10_lo, x0, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w10_hi, x0, acc1_hi);
            acc0_lo = _mm256_fmadd_ps(w01_lo, x1, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w01_hi, x1, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w11_lo, x1, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w11_hi, x1, acc1_hi);
            k += 2;
        }
        while k < in_dim {
            let x = _mm256_set1_ps(*input.get_unchecked(k));
            let w0_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w0_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base0 + k * block + 8));
            let w1_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w1_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base1 + k * block + 8));
            acc0_lo = _mm256_fmadd_ps(w0_lo, x, acc0_lo);
            acc0_hi = _mm256_fmadd_ps(w0_hi, x, acc0_hi);
            acc1_lo = _mm256_fmadd_ps(w1_lo, x, acc1_lo);
            acc1_hi = _mm256_fmadd_ps(w1_hi, x, acc1_hi);
            k += 1;
        }
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 0) * block), acc0_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 0) * block + 8), acc0_hi);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 1) * block), acc1_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add((blk + 1) * block + 8), acc1_hi);
        blk += tile;
    }
    while blk < blocks {
        let base = blk * in_dim * block;
        let mut acc_lo = _mm256_loadu_ps(bias.as_ptr().add(blk * block));
        let mut acc_hi = _mm256_loadu_ps(bias.as_ptr().add(blk * block + 8));
        let mut k = 0usize;
        while k + 4 <= in_dim {
            let x0 = _mm256_set1_ps(*input.get_unchecked(k));
            let x1 = _mm256_set1_ps(*input.get_unchecked(k + 1));
            let x2 = _mm256_set1_ps(*input.get_unchecked(k + 2));
            let x3 = _mm256_set1_ps(*input.get_unchecked(k + 3));
            let w0_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w0_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
            let w1_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block));
            let w1_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block + 8));
            let w2_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 2) * block));
            let w2_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 2) * block + 8));
            let w3_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 3) * block));
            let w3_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 3) * block + 8));
            acc_lo = _mm256_fmadd_ps(w0_lo, x0, acc_lo);
            acc_hi = _mm256_fmadd_ps(w0_hi, x0, acc_hi);
            acc_lo = _mm256_fmadd_ps(w1_lo, x1, acc_lo);
            acc_hi = _mm256_fmadd_ps(w1_hi, x1, acc_hi);
            acc_lo = _mm256_fmadd_ps(w2_lo, x2, acc_lo);
            acc_hi = _mm256_fmadd_ps(w2_hi, x2, acc_hi);
            acc_lo = _mm256_fmadd_ps(w3_lo, x3, acc_lo);
            acc_hi = _mm256_fmadd_ps(w3_hi, x3, acc_hi);
            k += 4;
        }
        while k + 2 <= in_dim {
            let x0 = _mm256_set1_ps(*input.get_unchecked(k));
            let x1 = _mm256_set1_ps(*input.get_unchecked(k + 1));
            let w0_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w0_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
            let w1_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block));
            let w1_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block + 8));
            acc_lo = _mm256_fmadd_ps(w0_lo, x0, acc_lo);
            acc_hi = _mm256_fmadd_ps(w0_hi, x0, acc_hi);
            acc_lo = _mm256_fmadd_ps(w1_lo, x1, acc_lo);
            acc_hi = _mm256_fmadd_ps(w1_hi, x1, acc_hi);
            k += 2;
        }
        while k < in_dim {
            let x = _mm256_set1_ps(*input.get_unchecked(k));
            let w_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
            acc_lo = _mm256_fmadd_ps(w_lo, x, acc_lo);
            acc_hi = _mm256_fmadd_ps(w_hi, x, acc_hi);
            k += 1;
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(blk * block), acc_lo);
        _mm256_storeu_ps(output.as_mut_ptr().add(blk * block + 8), acc_hi);
        blk += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn matmul_packed16_avx512_gemv_m1(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let tile = 4usize;
    let mut blk = 0usize;
    while blk + tile <= blocks {
        let base0 = (blk + 0) * in_dim * block;
        let base1 = (blk + 1) * in_dim * block;
        let base2 = (blk + 2) * in_dim * block;
        let base3 = (blk + 3) * in_dim * block;
        let mut acc0 = _mm512_loadu_ps(bias.as_ptr().add((blk + 0) * block));
        let mut acc1 = _mm512_loadu_ps(bias.as_ptr().add((blk + 1) * block));
        let mut acc2 = _mm512_loadu_ps(bias.as_ptr().add((blk + 2) * block));
        let mut acc3 = _mm512_loadu_ps(bias.as_ptr().add((blk + 3) * block));
        let mut k = 0usize;
        while k + 4 <= in_dim {
            let x0 = _mm512_set1_ps(*input.get_unchecked(k));
            let x1 = _mm512_set1_ps(*input.get_unchecked(k + 1));
            let x2 = _mm512_set1_ps(*input.get_unchecked(k + 2));
            let x3 = _mm512_set1_ps(*input.get_unchecked(k + 3));
            let w00 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w01 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block));
            let w02 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 2) * block));
            let w03 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 3) * block));
            let w10 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w11 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block));
            let w12 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 2) * block));
            let w13 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 3) * block));
            let w20 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + k * block));
            let w21 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + (k + 1) * block));
            let w22 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + (k + 2) * block));
            let w23 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + (k + 3) * block));
            let w30 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + k * block));
            let w31 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + (k + 1) * block));
            let w32 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + (k + 2) * block));
            let w33 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + (k + 3) * block));
            acc0 = _mm512_fmadd_ps(w00, x0, acc0);
            acc1 = _mm512_fmadd_ps(w10, x0, acc1);
            acc2 = _mm512_fmadd_ps(w20, x0, acc2);
            acc3 = _mm512_fmadd_ps(w30, x0, acc3);
            acc0 = _mm512_fmadd_ps(w01, x1, acc0);
            acc1 = _mm512_fmadd_ps(w11, x1, acc1);
            acc2 = _mm512_fmadd_ps(w21, x1, acc2);
            acc3 = _mm512_fmadd_ps(w31, x1, acc3);
            acc0 = _mm512_fmadd_ps(w02, x2, acc0);
            acc1 = _mm512_fmadd_ps(w12, x2, acc1);
            acc2 = _mm512_fmadd_ps(w22, x2, acc2);
            acc3 = _mm512_fmadd_ps(w32, x2, acc3);
            acc0 = _mm512_fmadd_ps(w03, x3, acc0);
            acc1 = _mm512_fmadd_ps(w13, x3, acc1);
            acc2 = _mm512_fmadd_ps(w23, x3, acc2);
            acc3 = _mm512_fmadd_ps(w33, x3, acc3);
            k += 4;
        }
        while k + 2 <= in_dim {
            let x0 = _mm512_set1_ps(*input.get_unchecked(k));
            let x1 = _mm512_set1_ps(*input.get_unchecked(k + 1));
            let w00 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w01 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + (k + 1) * block));
            let w10 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w11 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + (k + 1) * block));
            let w20 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + k * block));
            let w21 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + (k + 1) * block));
            let w30 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + k * block));
            let w31 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + (k + 1) * block));
            acc0 = _mm512_fmadd_ps(w00, x0, acc0);
            acc1 = _mm512_fmadd_ps(w10, x0, acc1);
            acc2 = _mm512_fmadd_ps(w20, x0, acc2);
            acc3 = _mm512_fmadd_ps(w30, x0, acc3);
            acc0 = _mm512_fmadd_ps(w01, x1, acc0);
            acc1 = _mm512_fmadd_ps(w11, x1, acc1);
            acc2 = _mm512_fmadd_ps(w21, x1, acc2);
            acc3 = _mm512_fmadd_ps(w31, x1, acc3);
            k += 2;
        }
        while k < in_dim {
            let x = _mm512_set1_ps(*input.get_unchecked(k));
            let w0 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w1 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w2 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + k * block));
            let w3 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + k * block));
            acc0 = _mm512_fmadd_ps(w0, x, acc0);
            acc1 = _mm512_fmadd_ps(w1, x, acc1);
            acc2 = _mm512_fmadd_ps(w2, x, acc2);
            acc3 = _mm512_fmadd_ps(w3, x, acc3);
            k += 1;
        }
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 0) * block), acc0);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 1) * block), acc1);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 2) * block), acc2);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 3) * block), acc3);
        blk += tile;
    }
    while blk < blocks {
        let base = blk * in_dim * block;
        let mut acc = _mm512_loadu_ps(bias.as_ptr().add(blk * block));
        let mut k = 0usize;
        while k + 4 <= in_dim {
            let x0 = _mm512_set1_ps(*input.get_unchecked(k));
            let x1 = _mm512_set1_ps(*input.get_unchecked(k + 1));
            let x2 = _mm512_set1_ps(*input.get_unchecked(k + 2));
            let x3 = _mm512_set1_ps(*input.get_unchecked(k + 3));
            let w0 = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w1 = _mm512_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block));
            let w2 = _mm512_loadu_ps(weight_packed.as_ptr().add(base + (k + 2) * block));
            let w3 = _mm512_loadu_ps(weight_packed.as_ptr().add(base + (k + 3) * block));
            acc = _mm512_fmadd_ps(w0, x0, acc);
            acc = _mm512_fmadd_ps(w1, x1, acc);
            acc = _mm512_fmadd_ps(w2, x2, acc);
            acc = _mm512_fmadd_ps(w3, x3, acc);
            k += 4;
        }
        while k + 2 <= in_dim {
            let x0 = _mm512_set1_ps(*input.get_unchecked(k));
            let x1 = _mm512_set1_ps(*input.get_unchecked(k + 1));
            let w0 = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let w1 = _mm512_loadu_ps(weight_packed.as_ptr().add(base + (k + 1) * block));
            acc = _mm512_fmadd_ps(w0, x0, acc);
            acc = _mm512_fmadd_ps(w1, x1, acc);
            k += 2;
        }
        while k < in_dim {
            let x = _mm512_set1_ps(*input.get_unchecked(k));
            let w = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            acc = _mm512_fmadd_ps(w, x, acc);
            k += 1;
        }
        _mm512_storeu_ps(output.as_mut_ptr().add(blk * block), acc);
        blk += 1;
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn matmul_packed16_batch_avx2(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    seq_len: usize,
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let nb = 4usize;
    let prefetch_dist = if in_dim >= 16 { 8 } else { 0 };
    for blk in 0..blocks {
        let bias_ptr = bias.as_ptr().add(blk * block);
        let base = blk * in_dim * block;
        let mut t = 0usize;
        while t + nb <= seq_len {
            let mut in0 = input.as_ptr().add((t + 0) * in_dim);
            let mut in1 = input.as_ptr().add((t + 1) * in_dim);
            let mut in2 = input.as_ptr().add((t + 2) * in_dim);
            let mut in3 = input.as_ptr().add((t + 3) * in_dim);
            let mut acc0_lo = _mm256_loadu_ps(bias_ptr);
            let mut acc0_hi = _mm256_loadu_ps(bias_ptr.add(8));
            let mut acc1_lo = acc0_lo;
            let mut acc1_hi = acc0_hi;
            let mut acc2_lo = acc0_lo;
            let mut acc2_hi = acc0_hi;
            let mut acc3_lo = acc0_lo;
            let mut acc3_hi = acc0_hi;
            for k in 0..in_dim {
                if prefetch_dist != 0 && k + prefetch_dist < in_dim {
                    let pf = base + (k + prefetch_dist) * block;
                    _mm_prefetch(weight_packed.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                }
                let w_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
                let w_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
                let x0 = _mm256_set1_ps(*in0);
                let x1 = _mm256_set1_ps(*in1);
                let x2 = _mm256_set1_ps(*in2);
                let x3 = _mm256_set1_ps(*in3);
                acc0_lo = _mm256_fmadd_ps(w_lo, x0, acc0_lo);
                acc0_hi = _mm256_fmadd_ps(w_hi, x0, acc0_hi);
                acc1_lo = _mm256_fmadd_ps(w_lo, x1, acc1_lo);
                acc1_hi = _mm256_fmadd_ps(w_hi, x1, acc1_hi);
                acc2_lo = _mm256_fmadd_ps(w_lo, x2, acc2_lo);
                acc2_hi = _mm256_fmadd_ps(w_hi, x2, acc2_hi);
                acc3_lo = _mm256_fmadd_ps(w_lo, x3, acc3_lo);
                acc3_hi = _mm256_fmadd_ps(w_hi, x3, acc3_hi);
                in0 = in0.add(1);
                in1 = in1.add(1);
                in2 = in2.add(1);
                in3 = in3.add(1);
            }
            let out0 = output.as_mut_ptr().add((t + 0) * out_dim + blk * block);
            let out1 = output.as_mut_ptr().add((t + 1) * out_dim + blk * block);
            let out2 = output.as_mut_ptr().add((t + 2) * out_dim + blk * block);
            let out3 = output.as_mut_ptr().add((t + 3) * out_dim + blk * block);
            _mm256_storeu_ps(out0, acc0_lo);
            _mm256_storeu_ps(out0.add(8), acc0_hi);
            _mm256_storeu_ps(out1, acc1_lo);
            _mm256_storeu_ps(out1.add(8), acc1_hi);
            _mm256_storeu_ps(out2, acc2_lo);
            _mm256_storeu_ps(out2.add(8), acc2_hi);
            _mm256_storeu_ps(out3, acc3_lo);
            _mm256_storeu_ps(out3.add(8), acc3_hi);
            t += nb;
        }
        while t < seq_len {
            let mut acc_lo = _mm256_loadu_ps(bias_ptr);
            let mut acc_hi = _mm256_loadu_ps(bias_ptr.add(8));
            let mut in0 = input.as_ptr().add(t * in_dim);
            for k in 0..in_dim {
                if prefetch_dist != 0 && k + prefetch_dist < in_dim {
                    let pf = base + (k + prefetch_dist) * block;
                    _mm_prefetch(weight_packed.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                }
                let w_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
                let w_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
                let x = _mm256_set1_ps(*in0);
                acc_lo = _mm256_fmadd_ps(w_lo, x, acc_lo);
                acc_hi = _mm256_fmadd_ps(w_hi, x, acc_hi);
                in0 = in0.add(1);
            }
            let out_ptr = output.as_mut_ptr().add(t * out_dim + blk * block);
            _mm256_storeu_ps(out_ptr, acc_lo);
            _mm256_storeu_ps(out_ptr.add(8), acc_hi);
            t += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum512_ps(v: std::arch::x86_64::__m512) -> f32 {
    let mut tmp = [0.0f32; 16];
    std::arch::x86_64::_mm512_storeu_ps(tmp.as_mut_ptr(), v);
    tmp.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn matmul_vec_avx512(
    weight: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    use std::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_setzero_ps};
    for o in 0..out_dim {
        let mut acc = match bias {
            Some(b) => b[o],
            None => 0.0,
        };
        let w_row = &weight[o * in_dim..][..in_dim];
        let mut v_acc = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= in_dim {
            let w = _mm512_loadu_ps(w_row.as_ptr().add(i));
            let x = _mm512_loadu_ps(input.as_ptr().add(i));
            v_acc = _mm512_fmadd_ps(w, x, v_acc);
            i += 16;
        }
        acc += hsum512_ps(v_acc);
        while i < in_dim {
            acc += w_row[i] * input[i];
            i += 1;
        }
        output[o] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn matmul_packed16_avx512(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let tile = 4usize;
    let mut blk = 0usize;
    while blk + tile <= blocks {
        let base0 = (blk + 0) * in_dim * block;
        let base1 = (blk + 1) * in_dim * block;
        let base2 = (blk + 2) * in_dim * block;
        let base3 = (blk + 3) * in_dim * block;
        let mut acc0 = _mm512_loadu_ps(bias.as_ptr().add((blk + 0) * block));
        let mut acc1 = _mm512_loadu_ps(bias.as_ptr().add((blk + 1) * block));
        let mut acc2 = _mm512_loadu_ps(bias.as_ptr().add((blk + 2) * block));
        let mut acc3 = _mm512_loadu_ps(bias.as_ptr().add((blk + 3) * block));
        for k in 0..in_dim {
            let x = _mm512_set1_ps(input[k]);
            let w0 = _mm512_loadu_ps(weight_packed.as_ptr().add(base0 + k * block));
            let w1 = _mm512_loadu_ps(weight_packed.as_ptr().add(base1 + k * block));
            let w2 = _mm512_loadu_ps(weight_packed.as_ptr().add(base2 + k * block));
            let w3 = _mm512_loadu_ps(weight_packed.as_ptr().add(base3 + k * block));
            acc0 = _mm512_fmadd_ps(w0, x, acc0);
            acc1 = _mm512_fmadd_ps(w1, x, acc1);
            acc2 = _mm512_fmadd_ps(w2, x, acc2);
            acc3 = _mm512_fmadd_ps(w3, x, acc3);
        }
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 0) * block), acc0);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 1) * block), acc1);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 2) * block), acc2);
        _mm512_storeu_ps(output.as_mut_ptr().add((blk + 3) * block), acc3);
        blk += tile;
    }
    while blk < blocks {
        let base = blk * in_dim * block;
        let mut acc = _mm512_loadu_ps(bias.as_ptr().add(blk * block));
        for k in 0..in_dim {
            let w = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
            let x = _mm512_set1_ps(input[k]);
            acc = _mm512_fmadd_ps(w, x, acc);
        }
        _mm512_storeu_ps(output.as_mut_ptr().add(blk * block), acc);
        blk += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn matmul_packed16_batch_avx512(
    weight_packed: &[f32],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    seq_len: usize,
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let prefetch_dist = if in_dim >= 16 { 8 } else { 0 };
    for blk in 0..blocks {
        let bias_ptr = bias.as_ptr().add(blk * block);
        let base = blk * in_dim * block;
        let mut t = 0usize;
        while t + 8 <= seq_len {
            let mut in0 = input.as_ptr().add((t + 0) * in_dim);
            let mut in1 = input.as_ptr().add((t + 1) * in_dim);
            let mut in2 = input.as_ptr().add((t + 2) * in_dim);
            let mut in3 = input.as_ptr().add((t + 3) * in_dim);
            let mut in4 = input.as_ptr().add((t + 4) * in_dim);
            let mut in5 = input.as_ptr().add((t + 5) * in_dim);
            let mut in6 = input.as_ptr().add((t + 6) * in_dim);
            let mut in7 = input.as_ptr().add((t + 7) * in_dim);
            let mut acc0 = _mm512_loadu_ps(bias_ptr);
            let mut acc1 = acc0;
            let mut acc2 = acc0;
            let mut acc3 = acc0;
            let mut acc4 = acc0;
            let mut acc5 = acc0;
            let mut acc6 = acc0;
            let mut acc7 = acc0;
            for k in 0..in_dim {
                if prefetch_dist != 0 && k + prefetch_dist < in_dim {
                    let pf = base + (k + prefetch_dist) * block;
                    _mm_prefetch(weight_packed.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                }
                let w = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
                let x0 = _mm512_set1_ps(*in0);
                let x1 = _mm512_set1_ps(*in1);
                let x2 = _mm512_set1_ps(*in2);
                let x3 = _mm512_set1_ps(*in3);
                let x4 = _mm512_set1_ps(*in4);
                let x5 = _mm512_set1_ps(*in5);
                let x6 = _mm512_set1_ps(*in6);
                let x7 = _mm512_set1_ps(*in7);
                acc0 = _mm512_fmadd_ps(w, x0, acc0);
                acc1 = _mm512_fmadd_ps(w, x1, acc1);
                acc2 = _mm512_fmadd_ps(w, x2, acc2);
                acc3 = _mm512_fmadd_ps(w, x3, acc3);
                acc4 = _mm512_fmadd_ps(w, x4, acc4);
                acc5 = _mm512_fmadd_ps(w, x5, acc5);
                acc6 = _mm512_fmadd_ps(w, x6, acc6);
                acc7 = _mm512_fmadd_ps(w, x7, acc7);
                in0 = in0.add(1);
                in1 = in1.add(1);
                in2 = in2.add(1);
                in3 = in3.add(1);
                in4 = in4.add(1);
                in5 = in5.add(1);
                in6 = in6.add(1);
                in7 = in7.add(1);
            }
            let out0 = output.as_mut_ptr().add((t + 0) * out_dim + blk * block);
            let out1 = output.as_mut_ptr().add((t + 1) * out_dim + blk * block);
            let out2 = output.as_mut_ptr().add((t + 2) * out_dim + blk * block);
            let out3 = output.as_mut_ptr().add((t + 3) * out_dim + blk * block);
            let out4 = output.as_mut_ptr().add((t + 4) * out_dim + blk * block);
            let out5 = output.as_mut_ptr().add((t + 5) * out_dim + blk * block);
            let out6 = output.as_mut_ptr().add((t + 6) * out_dim + blk * block);
            let out7 = output.as_mut_ptr().add((t + 7) * out_dim + blk * block);
            _mm512_storeu_ps(out0, acc0);
            _mm512_storeu_ps(out1, acc1);
            _mm512_storeu_ps(out2, acc2);
            _mm512_storeu_ps(out3, acc3);
            _mm512_storeu_ps(out4, acc4);
            _mm512_storeu_ps(out5, acc5);
            _mm512_storeu_ps(out6, acc6);
            _mm512_storeu_ps(out7, acc7);
            t += 8;
        }
        while t + 4 <= seq_len {
            let mut in0 = input.as_ptr().add((t + 0) * in_dim);
            let mut in1 = input.as_ptr().add((t + 1) * in_dim);
            let mut in2 = input.as_ptr().add((t + 2) * in_dim);
            let mut in3 = input.as_ptr().add((t + 3) * in_dim);
            let mut acc0 = _mm512_loadu_ps(bias_ptr);
            let mut acc1 = acc0;
            let mut acc2 = acc0;
            let mut acc3 = acc0;
            for k in 0..in_dim {
                if prefetch_dist != 0 && k + prefetch_dist < in_dim {
                    let pf = base + (k + prefetch_dist) * block;
                    _mm_prefetch(weight_packed.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                }
                let w = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
                let x0 = _mm512_set1_ps(*in0);
                let x1 = _mm512_set1_ps(*in1);
                let x2 = _mm512_set1_ps(*in2);
                let x3 = _mm512_set1_ps(*in3);
                acc0 = _mm512_fmadd_ps(w, x0, acc0);
                acc1 = _mm512_fmadd_ps(w, x1, acc1);
                acc2 = _mm512_fmadd_ps(w, x2, acc2);
                acc3 = _mm512_fmadd_ps(w, x3, acc3);
                in0 = in0.add(1);
                in1 = in1.add(1);
                in2 = in2.add(1);
                in3 = in3.add(1);
            }
            let out0 = output.as_mut_ptr().add((t + 0) * out_dim + blk * block);
            let out1 = output.as_mut_ptr().add((t + 1) * out_dim + blk * block);
            let out2 = output.as_mut_ptr().add((t + 2) * out_dim + blk * block);
            let out3 = output.as_mut_ptr().add((t + 3) * out_dim + blk * block);
            _mm512_storeu_ps(out0, acc0);
            _mm512_storeu_ps(out1, acc1);
            _mm512_storeu_ps(out2, acc2);
            _mm512_storeu_ps(out3, acc3);
            t += 4;
        }
        while t < seq_len {
            let mut acc = _mm512_loadu_ps(bias_ptr);
            let mut in0 = input.as_ptr().add(t * in_dim);
            for k in 0..in_dim {
                if prefetch_dist != 0 && k + prefetch_dist < in_dim {
                    let pf = base + (k + prefetch_dist) * block;
                    _mm_prefetch(weight_packed.as_ptr().add(pf) as *const i8, _MM_HINT_T0);
                }
                let w = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
                let x = _mm512_set1_ps(*in0);
                acc = _mm512_fmadd_ps(w, x, acc);
                in0 = in0.add(1);
            }
            let out_ptr = output.as_mut_ptr().add(t * out_dim + blk * block);
            _mm512_storeu_ps(out_ptr, acc);
            t += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn f32_to_bf16_bits_scalar(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn bf16_broadcast_pair(x0: f32, x1: f32) -> std::arch::x86_64::__m512bh {
    use std::arch::x86_64::*;
    let b0 = f32_to_bf16_bits_scalar(x0) as u32;
    let b1 = f32_to_bf16_bits_scalar(x1) as u32;
    let pair = ((b1 << 16) | b0) as i32;
    let v = _mm512_set1_epi32(pair);
    std::mem::transmute::<__m512i, __m512bh>(v)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn load_bf16x32(ptr: *const u16) -> std::arch::x86_64::__m512bh {
    use std::arch::x86_64::*;
    let v = _mm512_loadu_si512(ptr as *const __m512i);
    std::mem::transmute::<__m512i, __m512bh>(v)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bf16,avx512f")]
#[inline(never)]
pub(crate) unsafe fn matmul_packed16_batch_avx512_bf16(
    weight_packed: &[u16],
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    seq_len: usize,
    bias: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let blocks = out_dim / block;
    let pairs = (in_dim + 1) / 2;
    for t in 0..seq_len {
        let out_t = output.as_mut_ptr().add(t * out_dim);
        for blk in 0..blocks {
            let bias_ptr = bias.as_ptr().add(blk * block);
            _mm512_storeu_ps(out_t.add(blk * block), _mm512_loadu_ps(bias_ptr));
        }
    }
    for blk in 0..blocks {
        let base = blk * pairs * block * 2;
        for k in 0..pairs {
            let w_ptr = weight_packed.as_ptr().add(base + k * block * 2);
            let w_bf16 = load_bf16x32(w_ptr);
            let k0 = k * 2;
            let k1 = k0 + 1;
            for t in 0..seq_len {
                let x0 = *input.get_unchecked(t * in_dim + k0);
                let x1 = if k1 < in_dim {
                    *input.get_unchecked(t * in_dim + k1)
                } else {
                    0.0
                };
                let a_bf16 = bf16_broadcast_pair(x0, x1);
                let out_ptr = output.as_mut_ptr().add(t * out_dim + blk * block);
                let acc = _mm512_loadu_ps(out_ptr);
                let acc = _mm512_dpbf16_ps(acc, a_bf16, w_bf16);
                _mm512_storeu_ps(out_ptr, acc);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp256_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let exp_hi = _mm256_set1_ps(88.3762626647949f32);
    let exp_lo = _mm256_set1_ps(-88.3762626647949f32);
    let log2e = _mm256_set1_ps(1.44269504088896341f32);
    let ln2_hi = _mm256_set1_ps(0.693359375f32);
    let ln2_lo = _mm256_set1_ps(-2.12194440e-4f32);

    let c0 = _mm256_set1_ps(1.9875691500e-4f32);
    let c1 = _mm256_set1_ps(1.3981999507e-3f32);
    let c2 = _mm256_set1_ps(8.3334519073e-3f32);
    let c3 = _mm256_set1_ps(4.1665795894e-2f32);
    let c4 = _mm256_set1_ps(1.6666665459e-1f32);
    let c5 = _mm256_set1_ps(5.0000001201e-1f32);

    let one = _mm256_set1_ps(1.0f32);
    let half = _mm256_set1_ps(0.5f32);

    let mut x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    let fx = _mm256_fmadd_ps(x, log2e, half);
    let fx_floor = _mm256_floor_ps(fx);
    let emm0 = _mm256_cvttps_epi32(fx_floor);

    let tmp = _mm256_mul_ps(_mm256_cvtepi32_ps(emm0), ln2_hi);
    let z = _mm256_mul_ps(_mm256_cvtepi32_ps(emm0), ln2_lo);
    let mut y = _mm256_sub_ps(x, tmp);
    y = _mm256_sub_ps(y, z);

    let mut p = c0;
    p = _mm256_fmadd_ps(p, y, c1);
    p = _mm256_fmadd_ps(p, y, c2);
    p = _mm256_fmadd_ps(p, y, c3);
    p = _mm256_fmadd_ps(p, y, c4);
    p = _mm256_fmadd_ps(p, y, c5);
    p = _mm256_fmadd_ps(p, y, one);

    let emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    let emm0 = _mm256_slli_epi32(emm0, 23);
    let pow2n = _mm256_castsi256_ps(emm0);
    _mm256_mul_ps(p, pow2n)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub(crate) unsafe fn exp512_ps(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let exp_hi = _mm512_set1_ps(88.3762626647949f32);
    let exp_lo = _mm512_set1_ps(-88.3762626647949f32);
    let log2e = _mm512_set1_ps(1.44269504088896341f32);
    let ln2_hi = _mm512_set1_ps(0.693359375f32);
    let ln2_lo = _mm512_set1_ps(-2.12194440e-4f32);

    let c0 = _mm512_set1_ps(1.9875691500e-4f32);
    let c1 = _mm512_set1_ps(1.3981999507e-3f32);
    let c2 = _mm512_set1_ps(8.3334519073e-3f32);
    let c3 = _mm512_set1_ps(4.1665795894e-2f32);
    let c4 = _mm512_set1_ps(1.6666665459e-1f32);
    let c5 = _mm512_set1_ps(5.0000001201e-1f32);

    let one = _mm512_set1_ps(1.0f32);
    let half = _mm512_set1_ps(0.5f32);

    let mut x = _mm512_min_ps(x, exp_hi);
    x = _mm512_max_ps(x, exp_lo);

    let fx = _mm512_fmadd_ps(x, log2e, half);
    let fx_floor = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC }>(fx);
    let emm0 = _mm512_cvttps_epi32(fx_floor);

    let tmp = _mm512_mul_ps(_mm512_cvtepi32_ps(emm0), ln2_hi);
    let z = _mm512_mul_ps(_mm512_cvtepi32_ps(emm0), ln2_lo);
    let mut y = _mm512_sub_ps(x, tmp);
    y = _mm512_sub_ps(y, z);

    let mut p = c0;
    p = _mm512_fmadd_ps(p, y, c1);
    p = _mm512_fmadd_ps(p, y, c2);
    p = _mm512_fmadd_ps(p, y, c3);
    p = _mm512_fmadd_ps(p, y, c4);
    p = _mm512_fmadd_ps(p, y, c5);
    p = _mm512_fmadd_ps(p, y, one);

    let emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
    let emm0 = _mm512_slli_epi32(emm0, 23);
    let pow2n = _mm512_castsi512_ps(emm0);
    _mm512_mul_ps(p, pow2n)
}

pub fn validate_config(cfg: &ModelConfig) -> Result<(), crate::KernelError> {
    if cfg.d_state_pad < cfg.d_state {
        return Err(crate::KernelError::InvalidConfig("d_state_pad < d_state"));
    }
    if cfg.conv_kernel == 0 {
        return Err(crate::KernelError::InvalidConfig("conv_kernel must be > 0"));
    }
    if cfg.forward_kind == crate::ForwardKind::FullStateless {
        if cfg.seq_len == 0 {
            return Err(crate::KernelError::InvalidConfig("seq_len == 0"));
        }
        if cfg.feature_dim == 0 {
            return Err(crate::KernelError::InvalidConfig("feature_dim == 0"));
        }
        if cfg.static_dim_total == 0 {
            return Err(crate::KernelError::InvalidConfig("static_dim_total == 0"));
        }
    }
    Ok(())
}
