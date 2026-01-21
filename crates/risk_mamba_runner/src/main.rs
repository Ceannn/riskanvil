#!/usr/bin/env rust
use clap::Parser;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzReader;
use onnxruntime_sys as ort;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::ffi::{CStr, CString};
use std::fs::File;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Rust native ORT runner with metrics pack support")]
struct Args {
    #[arg(long, default_value = "onnx_out/model_stateful.onnx")]
    model: String,
    #[arg(long, default_value = "custom_op/build/libmamba_custom_ops.so")]
    custom_op: String,
    #[arg(long, default_value = "onnx_out/export_manifest.json")]
    manifest: String,
    #[arg(long, default_value = "outputs/rust_perf.json")]
    out: String,
    #[arg(long, default_value_t = 200)]
    iters: usize,
    #[arg(long, default_value_t = 50)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    repeat: usize,
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long, default_value_t = 128)]
    seq_len: usize,
    #[arg(long, default_value_t = 1)]
    feature_dim: usize,
    #[arg(long, default_value_t = 386)]
    static_dim: usize,
    #[arg(long, default_value_t = 4097)]
    vocab: i64,
    #[arg(long, default_value = "full", value_parser = ["full", "dispatch_only", "scan_only"])]
    bench_kind: String,
    #[arg(long, default_value_t = 1)]
    intra: i32,
    #[arg(long, default_value_t = 1)]
    inter: i32,
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    cpu_mem_arena: String,
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    mem_pattern: String,
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    mem_reuse: String,
    #[arg(long, default_value_t = false, conflicts_with = "metrics_only")]
    perf_only: bool,
    #[arg(long, default_value_t = false, conflicts_with = "perf_only")]
    metrics_only: bool,
    #[arg(long)]
    metrics_pack: Option<String>,
    #[arg(long, default_value = "confusion", value_parser = ["confusion", "fixed_fpr_sweep"])]
    metrics_mode: String,
    #[arg(long)]
    thr: Option<f64>,
    #[arg(long, default_value = "0.001,0.005,0.01")]
    fpr_levels: String,
    #[arg(long, default_value_t = 256)]
    metrics_batch: usize,
    #[arg(long, default_value_t = 0)]
    metrics_limit: usize,
    #[arg(long, default_value = "off", value_parser = ["on", "off"])]
    metrics_allow_nonfull: String,
    #[arg(long)]
    scores_out: Option<String>,
    #[arg(long, default_value = "off", value_parser = ["on", "off"])]
    profile: String,
    #[arg(long, default_value = "rust_profile.json")]
    profile_out: String,
    #[arg(long, default_value = "default", value_parser = ["default", "on", "off"])]
    intra_spinning: String,
    #[arg(long, default_value = "default", value_parser = ["default", "on", "off"])]
    inter_spinning: String,
    #[arg(long, default_value = "default", value_parser = ["default", "on", "off"])]
    force_spinning_stop: String,
    #[arg(long)]
    intra_affinity: Option<String>,
    #[arg(long)]
    cpu_affinity: Option<String>,
}

struct OrtApi {
    api: *const ort::OrtApi,
}

impl OrtApi {
    unsafe fn new() -> Self {
        let base = ort::OrtGetApiBase();
        let api = ((*base).GetApi.unwrap())(ort::ORT_API_VERSION);
        OrtApi { api }
    }
}

unsafe fn check_status(api: &OrtApi, status: *mut ort::OrtStatus) -> Result<(), String> {
    if status.is_null() {
        return Ok(());
    }
    let msg = CStr::from_ptr(((*api.api).GetErrorMessage.unwrap())(status))
        .to_string_lossy()
        .into_owned();
    ((*api.api).ReleaseStatus.unwrap())(status);
    Err(msg)
}

fn to_ort_bool(value: &str) -> Result<&'static str, String> {
    match value {
        "on" => Ok("1"),
        "off" => Ok("0"),
        _ => Err(format!("invalid on/off value: {}", value)),
    }
}

unsafe fn add_session_config(api: &OrtApi, so: *mut ort::OrtSessionOptions, key: &str, value: &str) -> Result<(), String> {
    let key_c = CString::new(key).unwrap();
    let val_c = CString::new(value).unwrap();
    let status = ((*api.api).AddSessionConfigEntry.unwrap())(so, key_c.as_ptr(), val_c.as_ptr());
    check_status(api, status)
}

fn parse_cpu_list(spec: &str) -> Result<Vec<usize>, String> {
    let mut cpus = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let s = start.trim().parse::<usize>().map_err(|_| format!("invalid cpu id: {}", start))?;
            let e = end.trim().parse::<usize>().map_err(|_| format!("invalid cpu id: {}", end))?;
            if s > e {
                return Err(format!("invalid cpu range: {}", part));
            }
            for cpu in s..=e {
                cpus.push(cpu);
            }
        } else {
            let cpu = part.parse::<usize>().map_err(|_| format!("invalid cpu id: {}", part))?;
            cpus.push(cpu);
        }
    }
    if cpus.is_empty() {
        return Err("cpu affinity list is empty".to_string());
    }
    cpus.sort_unstable();
    cpus.dedup();
    Ok(cpus)
}

#[cfg(target_os = "linux")]
fn apply_cpu_affinity(cpus: &[usize]) -> Result<(), String> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_ZERO(&mut set);
    }
    let max_cpu = libc::CPU_SETSIZE as usize;
    for &cpu in cpus {
        if cpu >= max_cpu {
            return Err(format!("cpu id {} exceeds CPU_SETSIZE {}", cpu, max_cpu));
        }
        unsafe {
            libc::CPU_SET(cpu, &mut set);
        }
    }
    let res = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    if res != 0 {
        return Err(std::io::Error::last_os_error().to_string());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn apply_cpu_affinity(_cpus: &[usize]) -> Result<(), String> {
    Err("cpu affinity not supported on this platform".to_string())
}

unsafe fn end_profiling(api: &OrtApi, session: *mut ort::OrtSession) -> Result<String, String> {
    let mut alloc: *mut ort::OrtAllocator = ptr::null_mut();
    check_status(api, ((*api.api).GetAllocatorWithDefaultOptions.unwrap())(&mut alloc))?;
    let mut out_path: *mut i8 = ptr::null_mut();
    let status = ((*api.api).SessionEndProfiling.unwrap())(session, alloc, &mut out_path);
    check_status(api, status)?;
    if out_path.is_null() {
        return Err("SessionEndProfiling returned null path".to_string());
    }
    let path = CStr::from_ptr(out_path).to_string_lossy().into_owned();
    ((*api.api).AllocatorFree.unwrap())(alloc, out_path as *mut c_void);
    Ok(path)
}

fn write_scores(path: &str, scores: &[f32]) -> Result<(), String> {
    let out_path = PathBuf::from(path);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    match out_path.extension().and_then(|s| s.to_str()) {
        Some("npy") => {
            let arr = Array1::from(scores.to_vec());
            ndarray_npy::write_npy(&out_path, &arr).map_err(|e| e.to_string())?;
        }
        _ => {
            let data = serde_json::to_string_pretty(scores).map_err(|e| e.to_string())?;
            std::fs::write(out_path, data).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

struct OrtEnv {
    ptr: *mut ort::OrtEnv,
    api: OrtApi,
}

impl OrtEnv {
    unsafe fn new(name: &str) -> Result<Self, String> {
        let api = OrtApi::new();
        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let cname = CString::new(name).unwrap();
        let status = ((*api.api).CreateEnv.unwrap())(ort::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, cname.as_ptr(), &mut env);
        check_status(&api, status)?;
        Ok(OrtEnv { ptr: env, api })
    }
}

impl Drop for OrtEnv {
    fn drop(&mut self) {
        unsafe {
            ((*self.api.api).ReleaseEnv.unwrap())(self.ptr);
        }
    }
}

struct OrtSession {
    ptr: *mut ort::OrtSession,
    api: OrtApi,
    mem_info: *mut ort::OrtMemoryInfo,
    input_names: Vec<String>,
    output_names: Vec<String>,
    custom_handle: *mut c_void,
}

impl OrtSession {
    unsafe fn new(env: &OrtEnv, args: &Args) -> Result<Self, String> {
        let api = OrtApi::new();
        let mut so: *mut ort::OrtSessionOptions = ptr::null_mut();
        check_status(&api, ((*api.api).CreateSessionOptions.unwrap())(&mut so))?;
        check_status(&api, ((*api.api).SetIntraOpNumThreads.unwrap())(so, args.intra))?;
        check_status(&api, ((*api.api).SetInterOpNumThreads.unwrap())(so, args.inter))?;

        if args.cpu_mem_arena == "off" {
            check_status(&api, ((*api.api).DisableCpuMemArena.unwrap())(so))?;
        }
        if args.mem_pattern == "off" {
            check_status(&api, ((*api.api).DisableMemPattern.unwrap())(so))?;
        }
        if args.intra_spinning != "default" {
            let val = to_ort_bool(&args.intra_spinning)?;
            add_session_config(&api, so, "session.intra_op.allow_spinning", val)?;
        }
        if args.inter_spinning != "default" {
            let val = to_ort_bool(&args.inter_spinning)?;
            add_session_config(&api, so, "session.inter_op.allow_spinning", val)?;
        }
        if args.force_spinning_stop != "default" {
            let val = to_ort_bool(&args.force_spinning_stop)?;
            add_session_config(&api, so, "session.force_spinning_stop", val)?;
        }
        if let Some(affinity) = args.intra_affinity.as_ref() {
            if !affinity.trim().is_empty() {
                add_session_config(&api, so, "session.intra_op_thread_affinities", affinity)?;
            }
        }
        if args.profile == "on" {
            let profile_path = Path::new(&args.profile_out);
            if let Some(parent) = profile_path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
            }
            let prefix = CString::new(args.profile_out.clone()).unwrap();
            check_status(&api, ((*api.api).EnableProfiling.unwrap())(so, prefix.as_ptr()))?;
        }
        let custom = CString::new(args.custom_op.clone()).unwrap();
        let mut custom_handle: *mut c_void = ptr::null_mut();
        check_status(
            &api,
            ((*api.api).RegisterCustomOpsLibrary.unwrap())(so, custom.as_ptr(), &mut custom_handle),
        )?;

        let model = CString::new(args.model.clone()).unwrap();
        let mut session: *mut ort::OrtSession = ptr::null_mut();
        check_status(&api, ((*api.api).CreateSession.unwrap())(env.ptr, model.as_ptr(), so, &mut session))?;
        ((*api.api).ReleaseSessionOptions.unwrap())(so);

        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        check_status(&api, ((*api.api).CreateCpuMemoryInfo.unwrap())(ort::OrtAllocatorType::OrtArenaAllocator, ort::OrtMemType::OrtMemTypeDefault, &mut mem_info))?;

        let input_names = get_io_names(&api, session, true)?;
        let output_names = get_io_names(&api, session, false)?;
        Ok(OrtSession {
            ptr: session,
            api,
            mem_info,
            input_names,
            output_names,
            custom_handle,
        })
    }
}

impl Drop for OrtSession {
    fn drop(&mut self) {
        unsafe {
            ((*self.api.api).ReleaseMemoryInfo.unwrap())(self.mem_info);
            ((*self.api.api).ReleaseSession.unwrap())(self.ptr);
            if !self.custom_handle.is_null() {
                #[cfg(unix)]
                {
                    libc::dlclose(self.custom_handle);
                }
            }
        }
    }
}

unsafe fn get_io_names(api: &OrtApi, sess: *mut ort::OrtSession, is_input: bool) -> Result<Vec<String>, String> {
    let mut count: usize = 0;
    if is_input {
        check_status(api, ((*api.api).SessionGetInputCount.unwrap())(sess, &mut count))?;
    } else {
        check_status(api, ((*api.api).SessionGetOutputCount.unwrap())(sess, &mut count))?;
    }
    let mut alloc: *mut ort::OrtAllocator = ptr::null_mut();
    check_status(api, ((*api.api).GetAllocatorWithDefaultOptions.unwrap())(&mut alloc))?;
    let mut names = Vec::with_capacity(count);
    for i in 0..count {
        let mut name_ptr: *mut i8 = ptr::null_mut();
        if is_input {
            check_status(api, ((*api.api).SessionGetInputName.unwrap())(sess, i as usize, alloc, &mut name_ptr))?;
        } else {
            check_status(api, ((*api.api).SessionGetOutputName.unwrap())(sess, i as usize, alloc, &mut name_ptr))?;
        }
        let name = CStr::from_ptr(name_ptr).to_string_lossy().into_owned();
        ((*api.api).AllocatorFree.unwrap())(alloc, name_ptr as *mut c_void);
        names.push(name);
    }
    Ok(names)
}

struct InputTensor {
    ort_value: *mut ort::OrtValue,
    _storage: InputStorage,
}

enum InputStorage {
    I64(Vec<i64>),
    F32(Vec<f32>),
}

unsafe fn make_tensor_i64(api: &OrtApi, mem_info: *mut ort::OrtMemoryInfo, data: Vec<i64>, shape: &[i64]) -> Result<InputTensor, String> {
    let mut ort_value: *mut ort::OrtValue = ptr::null_mut();
    let status = ((*api.api).CreateTensorWithDataAsOrtValue.unwrap())(
        mem_info,
        data.as_ptr() as *mut c_void,
        (data.len() * std::mem::size_of::<i64>()) as usize,
        shape.as_ptr(),
        shape.len() as usize,
        ort::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &mut ort_value,
    );
    check_status(api, status)?;
    Ok(InputTensor { ort_value, _storage: InputStorage::I64(data) })
}

unsafe fn make_tensor_f32(api: &OrtApi, mem_info: *mut ort::OrtMemoryInfo, data: Vec<f32>, shape: &[i64]) -> Result<InputTensor, String> {
    let mut ort_value: *mut ort::OrtValue = ptr::null_mut();
    let status = ((*api.api).CreateTensorWithDataAsOrtValue.unwrap())(
        mem_info,
        data.as_ptr() as *mut c_void,
        (data.len() * std::mem::size_of::<f32>()) as usize,
        shape.as_ptr(),
        shape.len() as usize,
        ort::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &mut ort_value,
    );
    check_status(api, status)?;
    Ok(InputTensor { ort_value, _storage: InputStorage::F32(data) })
}

unsafe fn run_session(
    sess: &OrtSession,
    input_names: &[*const i8],
    inputs: &[*mut ort::OrtValue],
    output_names: &[*const i8],
) -> Result<Vec<*mut ort::OrtValue>, String> {
    let mut outputs: Vec<*mut ort::OrtValue> = vec![ptr::null_mut(); output_names.len()];
    let input_ptrs: Vec<*const ort::OrtValue> = inputs.iter().map(|v| *v as *const ort::OrtValue).collect();
    let status = ((*sess.api.api).Run.unwrap())(
        sess.ptr,
        ptr::null(),
        input_names.as_ptr(),
        input_ptrs.as_ptr(),
        inputs.len(),
        output_names.as_ptr(),
        output_names.len(),
        outputs.as_mut_ptr(),
    );
    check_status(&sess.api, status)?;
    Ok(outputs)
}

unsafe fn extract_f32(api: &OrtApi, value: *mut ort::OrtValue) -> Result<Vec<f32>, String> {
    let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
    check_status(api, ((*api.api).GetTensorTypeAndShape.unwrap())(value, &mut info))?;
    let mut count: usize = 0;
    check_status(api, ((*api.api).GetTensorShapeElementCount.unwrap())(info, &mut count))?;
    ((*api.api).ReleaseTensorTypeAndShapeInfo.unwrap())(info);
    let mut data_ptr: *mut c_void = ptr::null_mut();
    check_status(api, ((*api.api).GetTensorMutableData.unwrap())(value, &mut data_ptr))?;
    let data = std::slice::from_raw_parts(data_ptr as *const f32, count).to_vec();
    Ok(data)
}

unsafe fn release_values(api: &OrtApi, values: &mut Vec<*mut ort::OrtValue>) {
    for v in values.iter_mut() {
        if !v.is_null() {
            ((*api.api).ReleaseValue.unwrap())(*v);
            *v = ptr::null_mut();
        }
    }
}

fn sha256(path: &str) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).ok()?;
    Some(format!("{:x}", hasher.finalize()))
}

fn read_cpu_model() -> Option<String> {
    let data = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in data.lines() {
        if line.to_lowercase().starts_with("model name") {
            return line.split(':').nth(1).map(|s| s.trim().to_string());
        }
    }
    None
}

fn is_wsl() -> bool {
    if let Ok(ver) = std::fs::read_to_string("/proc/version") {
        return ver.to_lowercase().contains("microsoft");
    }
    false
}

fn read_cpus_allowed_list() -> Option<String> {
    let data = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in data.lines() {
        if line.starts_with("Cpus_allowed_list:") {
            return Some(line.split(':').nth(1).unwrap_or("").trim().to_string());
        }
    }
    None
}

fn getrusage() -> Value {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        json!({
            "ru_utime_s": usage.ru_utime.tv_sec as f64 + (usage.ru_utime.tv_usec as f64) / 1e6,
            "ru_stime_s": usage.ru_stime.tv_sec as f64 + (usage.ru_stime.tv_usec as f64) / 1e6,
            "ru_maxrss_kb": usage.ru_maxrss as i64,
            "ru_minflt": usage.ru_minflt as i64,
            "ru_majflt": usage.ru_majflt as i64,
            "ru_nvcsw": usage.ru_nvcsw as i64,
            "ru_nivcsw": usage.ru_nivcsw as i64,
        })
    }
}

fn summarize_latency(runs_ms: &[f64]) -> Value {
    if runs_ms.is_empty() {
        return json!({"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "var": 0.0, "cv": 0.0});
    }
    let mean = runs_ms.iter().sum::<f64>() / runs_ms.len() as f64;
    let var = runs_ms.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / runs_ms.len() as f64;
    let std = var.sqrt();
    let mut sorted = runs_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[idx]
    };
    json!({
        "mean": mean,
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "min": *sorted.first().unwrap(),
        "max": *sorted.last().unwrap(),
        "std": std,
        "var": var,
        "cv": if mean > 0.0 { std / mean } else { 0.0 },
    })
}

fn outlier_count(runs_ms: &[f64]) -> i64 {
    if runs_ms.is_empty() {
        return 0;
    }
    let mut sorted = runs_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    let mut abs_dev: Vec<f64> = sorted.iter().map(|x| (x - med).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = abs_dev[abs_dev.len() / 2];
    if mad <= 0.0 {
        return 0;
    }
    let thresh = med + 6.0 * mad;
    runs_ms.iter().filter(|x| **x > thresh).count() as i64
}

fn load_manifest(path: &str) -> Value {
    if let Ok(data) = std::fs::read_to_string(path) {
        if let Ok(v) = serde_json::from_str::<Value>(&data) {
            return v;
        }
    }
    json!({})
}

fn load_npz(path: &str) -> Result<(Value, Value, String, MetricsPack), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut npz = NpzReader::new(file).map_err(|e| e.to_string())?;

    let input_ids: Array3<i64> = npz.by_name("input_ids.npy").map_err(|e| e.to_string())?;
    let lengths: Array1<i64> = npz.by_name("lengths.npy").map_err(|e| e.to_string())?;
    let labels: Array1<i64> = npz.by_name("labels.npy").map_err(|e| e.to_string())?;

    let dt: Option<Array2<f32>> = npz.by_name("dt.npy").ok();
    let uid: Option<Array1<i64>> = npz.by_name("uid.npy").ok();
    let static_feat: Option<Array2<f32>> = npz.by_name("static.npy").ok();

    let meta_path = Path::new(path).with_extension("meta.json");
    let meta = if meta_path.exists() {
        let s = std::fs::read_to_string(&meta_path).unwrap_or_else(|_| "{}".to_string());
        serde_json::from_str(&s).unwrap_or_else(|_| json!({}))
    } else {
        json!({})
    };

    let pack_sha = sha256(path).unwrap_or_else(|| "UNKNOWN".to_string());
    let info = json!({
        "path": path,
        "sha256": pack_sha,
        "n_samples": labels.len(),
        "label_pos": labels.iter().filter(|x| **x == 1).count(),
        "label_neg": labels.iter().filter(|x| **x == 0).count(),
        "meta": meta,
    });

    Ok((info.clone(), meta, pack_sha, MetricsPack { input_ids, lengths, labels, dt, uid, static_feat }))
}

struct MetricsPack {
    input_ids: Array3<i64>,
    lengths: Array1<i64>,
    labels: Array1<i64>,
    dt: Option<Array2<f32>>,
    uid: Option<Array1<i64>>,
    static_feat: Option<Array2<f32>>,
}

fn confusion_metrics(y_true: &[i64], scores: &[f32], thr: f64) -> Value {
    let mut tp = 0i64;
    let mut fp = 0i64;
    let mut tn = 0i64;
    let mut fnv = 0i64;
    for (y, s) in y_true.iter().zip(scores.iter()) {
        let pred = (*s as f64) >= thr;
        match (*y, pred) {
            (1, true) => tp += 1,
            (0, true) => fp += 1,
            (0, false) => tn += 1,
            (1, false) => fnv += 1,
            _ => {}
        }
    }
    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fnv > 0 { tp as f64 / (tp + fnv) as f64 } else { 0.0 };
    let fpr = if fp + tn > 0 { fp as f64 / (fp + tn) as f64 } else { 0.0 };
    json!({
        "thr": thr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fnv,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": recall,
    })
}

fn fixed_fpr_sweep(y_true: &[i64], scores: &[f32], levels: &[f64]) -> Value {
    let mut neg_scores: Vec<f32> = y_true
        .iter()
        .zip(scores.iter())
        .filter(|(y, _)| **y == 0)
        .map(|(_, s)| *s)
        .collect();
    if neg_scores.is_empty() {
        return json!({"fpr_levels": levels, "sweep": []});
    }
    neg_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sweep = Vec::new();
    for fpr in levels {
        let q = (1.0 - *fpr).clamp(0.0, 1.0);
        let idx = ((neg_scores.len() - 1) as f64 * q).round() as usize;
        let thr = neg_scores[idx] as f64;
        let mut row = confusion_metrics(y_true, scores, thr);
        if let Some(obj) = row.as_object_mut() {
            obj.insert("fpr_target".to_string(), json!(*fpr));
        }
        sweep.push(row);
    }
    json!({
        "fpr_levels": levels,
        "sweep": sweep,
    })
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let metrics_enabled = args.metrics_only;
    let perf_enabled = !metrics_enabled;
    let perf_only_explicit = args.perf_only;
    let run_mode = if metrics_enabled { "metrics-only" } else { "perf-only" };
    if metrics_enabled && args.metrics_pack.is_none() {
        return Err("--metrics-pack is required for --metrics-only".to_string());
    }
    if args.scores_out.is_some() && !metrics_enabled {
        return Err("--scores-out requires --metrics-only".to_string());
    }
    if args.profile == "on" && !perf_enabled {
        return Err("--profile is only supported in perf mode".to_string());
    }

    let cpu_affinity_list = if let Some(spec) = args.cpu_affinity.as_ref() {
        let cpus = parse_cpu_list(spec)?;
        apply_cpu_affinity(&cpus)?;
        Some(cpus)
    } else {
        None
    };
    let manifest = load_manifest(&args.manifest);

    let env_info = json!({
        "cpu_model": read_cpu_model(),
        "cpu_count": num_cpus::get(),
        "uname_r": std::fs::read_to_string("/proc/sys/kernel/osrelease").ok().map(|s| s.trim().to_string()),
        "proc_version": std::fs::read_to_string("/proc/version").ok(),
        "is_wsl": is_wsl(),
        "wsl_distro": std::env::var("WSL_DISTRO_NAME").ok(),
        "python_version": std::env::var("PYTHON_VERSION").ok(),
        "affinity": read_cpus_allowed_list(),
        "ld_preload": std::env::var("LD_PRELOAD").ok(),
        "contamination_warning": std::env::var("LD_PRELOAD").ok().map(|s| !s.is_empty()).unwrap_or(false),
        "env": {
            "OMP_NUM_THREADS": std::env::var("OMP_NUM_THREADS").ok(),
            "ORT_DISABLE_ARENA": std::env::var("ORT_DISABLE_ARENA").ok(),
            "ORT_ENABLE_MEM_PATTERN": std::env::var("ORT_ENABLE_MEM_PATTERN").ok(),
            "MAMBA_BDL_DBLOCK": std::env::var("MAMBA_BDL_DBLOCK").ok(),
            "MAMBA_BDL_PREFETCH": std::env::var("MAMBA_BDL_PREFETCH").ok(),
        },
    });

    let mut d_inner = None;
    let mut d_state = None;
    if let Some(m) = manifest.get("mamba") {
        let d_model = m.get("d_model").and_then(|v| v.as_i64()).unwrap_or(0);
        let expand = m.get("mamba_expand").and_then(|v| v.as_i64()).unwrap_or(1);
        let d_st = m.get("mamba_d_state").and_then(|v| v.as_i64()).unwrap_or(0);
        if d_model > 0 && d_st > 0 {
            d_inner = Some((d_model * expand) as usize);
            d_state = Some(d_st as usize);
        }
    }

    let ort_session = unsafe { OrtSession::new(&OrtEnv::new("risk_mamba")?, &args)? };
    let input_names = ort_session.input_names.clone();
    let output_names = ort_session.output_names.clone();

    let input_name_c: Vec<CString> = input_names.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();
    let output_name_c: Vec<CString> = output_names.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();
    // Keep CString storage alive for name pointers used by ORT.
    let input_name_ptrs: Vec<*const i8> = input_name_c.iter().map(|s| s.as_ptr()).collect();
    let output_name_ptrs: Vec<*const i8> = output_name_c.iter().map(|s| s.as_ptr()).collect();

    let mut runs_ms: Vec<f64> = Vec::new();
    let mut repeat_means: Vec<f64> = Vec::new();
    let mut repeat_p99s: Vec<f64> = Vec::new();
    let mut latency_summary = summarize_latency(&runs_ms);
    let mut outlier_cnt = 0i64;
    let mut repeat_cv = 0.0;
    let mut perf_skipped_reason: Option<&str> = None;
    if perf_enabled {
        // Build random feed for perf.
        let mut inputs: Vec<InputTensor> = Vec::new();
        for name in &input_names {
            if name == "input_ids" {
                let mut data = Vec::with_capacity(args.batch * args.seq_len * args.feature_dim);
                for _ in 0..data.capacity() {
                    data.push(1 + (rand::random::<u32>() as i64 % args.vocab));
                }
                let shape = vec![args.batch as i64, args.seq_len as i64, args.feature_dim as i64];
                inputs.push(unsafe { make_tensor_i64(&ort_session.api, ort_session.mem_info, data, &shape)? });
            } else if name == "lengths" {
                let data = vec![args.seq_len as i64; args.batch];
                let shape = vec![args.batch as i64];
                inputs.push(unsafe { make_tensor_i64(&ort_session.api, ort_session.mem_info, data, &shape)? });
            } else if name == "static" {
                let data = vec![0.0f32; args.batch * args.static_dim];
                let shape = vec![args.batch as i64, args.static_dim as i64];
                inputs.push(unsafe { make_tensor_f32(&ort_session.api, ort_session.mem_info, data, &shape)? });
            } else if name == "dt" {
                let data = vec![0.0f32; args.batch * args.seq_len];
                let shape = vec![args.batch as i64, args.seq_len as i64];
                inputs.push(unsafe { make_tensor_f32(&ort_session.api, ort_session.mem_info, data, &shape)? });
            } else if name == "uid" {
                let data = vec![0i64; args.batch];
                let shape = vec![args.batch as i64];
                inputs.push(unsafe { make_tensor_i64(&ort_session.api, ort_session.mem_info, data, &shape)? });
            } else if name.starts_with("prev_state_") {
                let (di, ds) = (d_inner.unwrap_or(1), d_state.unwrap_or(1));
                let data = vec![0.0f32; args.batch * di * ds];
                let shape = vec![args.batch as i64, di as i64, ds as i64];
                inputs.push(unsafe { make_tensor_f32(&ort_session.api, ort_session.mem_info, data, &shape)? });
            } else {
                return Err(format!("unexpected input name: {}", name));
            }
        }

        // Warmup
        for _ in 0..args.warmup {
            let in_ptrs: Vec<*mut ort::OrtValue> = inputs.iter().map(|x| x.ort_value).collect();
            let mut outputs = unsafe { run_session(&ort_session, &input_name_ptrs, &in_ptrs, &output_name_ptrs)? };
            unsafe { release_values(&ort_session.api, &mut outputs) };
        }

        for _ in 0..args.repeat {
            let mut per_iter = Vec::with_capacity(args.iters);
            for _ in 0..args.iters {
                let in_ptrs: Vec<*mut ort::OrtValue> = inputs.iter().map(|x| x.ort_value).collect();
                let t0 = Instant::now();
                let mut outputs = unsafe { run_session(&ort_session, &input_name_ptrs, &in_ptrs, &output_name_ptrs)? };
                let dt = t0.elapsed().as_secs_f64() * 1000.0;
                per_iter.push(dt);
                unsafe { release_values(&ort_session.api, &mut outputs) };
            }
            let mean = per_iter.iter().sum::<f64>() / per_iter.len() as f64;
            let mut sorted = per_iter.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p99 = sorted[((sorted.len() - 1) as f64 * 0.99).round() as usize];
            repeat_means.push(mean);
            repeat_p99s.push(p99);
            runs_ms.extend(per_iter);
        }

        latency_summary = summarize_latency(&runs_ms);
        outlier_cnt = outlier_count(&runs_ms);
        repeat_cv = if repeat_means.is_empty() {
            0.0
        } else {
            let mean = repeat_means.iter().sum::<f64>() / repeat_means.len() as f64;
            let var = repeat_means.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / repeat_means.len() as f64;
            if mean > 0.0 { var.sqrt() / mean } else { 0.0 }
        };
    } else {
        perf_skipped_reason = Some("metrics-only");
    }

    let mut metrics_pack_info = None;
    let mut task_metrics = None;
    let mut metrics_skipped_reason: Option<&str> = None;
    let mut scores_out_path: Option<String> = None;
    if metrics_enabled {
        if args.bench_kind != "full" && args.metrics_allow_nonfull != "on" {
            metrics_skipped_reason = Some("bench_kind != full");
        } else {
            let pack_path = args.metrics_pack.as_ref().unwrap();
            let (info, _meta, _sha, pack) = load_npz(pack_path)?;
            metrics_pack_info = Some(info);

            let total = if args.metrics_limit > 0 {
                args.metrics_limit.min(pack.labels.len())
            } else {
                pack.labels.len()
            };
            let mut scores = vec![0.0f32; total];
            let y_true: Vec<i64> = pack.labels.iter().take(total).copied().collect();

            for start in (0..total).step_by(args.metrics_batch) {
                let end = (start + args.metrics_batch).min(total);
                let ids = pack.input_ids.slice(ndarray::s![start..end, .., ..]).to_owned();
                let lengths = pack.lengths.slice(ndarray::s![start..end]).to_owned();
                let static_feat = pack.static_feat.as_ref().map(|s| s.slice(ndarray::s![start..end, ..]).to_owned());
                let dt = pack.dt.as_ref().map(|d| d.slice(ndarray::s![start..end, ..]).to_owned());
                let uid = pack.uid.as_ref().map(|u| u.slice(ndarray::s![start..end]).to_owned());

                let mut batch_inputs: Vec<InputTensor> = Vec::new();
                for name in &input_names {
                    if name == "input_ids" {
                        let shape = vec![ids.shape()[0] as i64, ids.shape()[1] as i64, ids.shape()[2] as i64];
                        batch_inputs.push(unsafe { make_tensor_i64(&ort_session.api, ort_session.mem_info, ids.iter().copied().collect(), &shape)? });
                    } else if name == "lengths" {
                        let shape = vec![lengths.shape()[0] as i64];
                        batch_inputs.push(unsafe { make_tensor_i64(&ort_session.api, ort_session.mem_info, lengths.iter().copied().collect(), &shape)? });
                    } else if name == "static" {
                        let data = static_feat
                            .as_ref()
                            .map(|s| s.iter().copied().collect())
                            .unwrap_or_else(|| vec![0.0f32; (end - start) * args.static_dim]);
                        let shape = vec![end as i64 - start as i64, args.static_dim as i64];
                        batch_inputs.push(unsafe { make_tensor_f32(&ort_session.api, ort_session.mem_info, data, &shape)? });
                    } else if name == "dt" {
                        let data = dt
                            .as_ref()
                            .map(|d| d.iter().copied().collect())
                            .unwrap_or_else(|| vec![0.0f32; (end - start) * args.seq_len]);
                        let shape = vec![end as i64 - start as i64, args.seq_len as i64];
                        batch_inputs.push(unsafe { make_tensor_f32(&ort_session.api, ort_session.mem_info, data, &shape)? });
                    } else if name == "uid" {
                        let data = uid
                            .as_ref()
                            .map(|u| u.iter().copied().collect())
                            .unwrap_or_else(|| vec![0i64; end - start]);
                        let shape = vec![end as i64 - start as i64];
                        batch_inputs.push(unsafe { make_tensor_i64(&ort_session.api, ort_session.mem_info, data, &shape)? });
                    } else if name.starts_with("prev_state_") {
                        let (di, ds) = (d_inner.unwrap_or(1), d_state.unwrap_or(1));
                        let data = vec![0.0f32; (end - start) * di * ds];
                        let shape = vec![end as i64 - start as i64, di as i64, ds as i64];
                        batch_inputs.push(unsafe { make_tensor_f32(&ort_session.api, ort_session.mem_info, data, &shape)? });
                    } else {
                        return Err(format!("unexpected metrics input: {}", name));
                    }
                }

                let in_ptrs: Vec<*mut ort::OrtValue> = batch_inputs.iter().map(|x| x.ort_value).collect();
                let mut outputs = unsafe { run_session(&ort_session, &input_name_ptrs, &in_ptrs, &output_name_ptrs)? };
                let logits = unsafe { extract_f32(&ort_session.api, outputs[0])? };
                unsafe { release_values(&ort_session.api, &mut outputs) };
                let mut idx = 0;
                for i in start..end {
                    scores[i] = logits[idx + 1];
                    idx += 2;
                }
            }

            if let Some(out_path) = args.scores_out.as_ref() {
                write_scores(out_path, &scores)?;
                scores_out_path = Some(out_path.clone());
            }

            if args.metrics_mode == "confusion" {
                let thr = args.thr.ok_or_else(|| "--thr is required for metrics-mode=confusion".to_string())?;
                task_metrics = Some(confusion_metrics(&y_true, &scores, thr));
            } else {
                let levels: Vec<f64> = args
                    .fpr_levels
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .collect();
                task_metrics = Some(fixed_fpr_sweep(&y_true, &scores, &levels));
            }
        }
    } else {
        metrics_skipped_reason = Some("perf-only");
    }

    let mut profile_path: Option<String> = None;
    if args.profile == "on" {
        let actual = unsafe { end_profiling(&ort_session.api, ort_session.ptr)? };
        if actual != args.profile_out {
            let desired = PathBuf::from(&args.profile_out);
            if let Some(parent) = desired.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
            }
            std::fs::rename(&actual, &desired).map_err(|e| e.to_string())?;
        }
        profile_path = Some(args.profile_out.clone());
    }

    let metrics = json!({
        "bench": {
            "iters": args.iters,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "intra": args.intra,
            "inter": args.inter,
            "batch": args.batch,
            "seq_len": args.seq_len,
            "feature_dim": args.feature_dim,
            "static_dim": args.static_dim,
            "bench_kind": args.bench_kind,
            "metrics_pack": args.metrics_pack,
            "metrics_mode": args.metrics_mode,
            "metrics_batch": args.metrics_batch,
            "metrics_limit": args.metrics_limit,
            "thr": args.thr,
            "fpr_levels": args.fpr_levels,
            "run_mode": run_mode,
            "perf_only": perf_enabled,
            "perf_only_explicit": perf_only_explicit,
            "metrics_only": metrics_enabled,
            "cpu_affinity": args.cpu_affinity,
        },
        "model": {
            "path": args.model,
            "is_ort_format": args.model.ends_with(".ort"),
            "sha256": sha256(&args.model),
        },
        "custom_op": {
            "path": args.custom_op,
            "sha256": sha256(&args.custom_op),
        },
        "ort_session": {
            "ort_version": format!("ORT_API_VERSION_{}", ort::ORT_API_VERSION),
            "providers": ["CPUExecutionProvider"],
            "cpu_mem_arena": args.cpu_mem_arena,
            "mem_pattern": args.mem_pattern,
            "mem_reuse": args.mem_reuse,
            "graph_optimization_level": "DEFAULT",
            "io_binding": "none",
            "intra_spinning": args.intra_spinning,
            "inter_spinning": args.inter_spinning,
            "force_spinning_stop": args.force_spinning_stop,
            "intra_thread_affinities": args.intra_affinity,
            "cpu_affinity_list": cpu_affinity_list,
            "profile": args.profile,
            "profile_out": args.profile_out,
        },
        "perf": if perf_enabled {
            json!({
                "enabled": true,
                "latency": {
                    "runs_ms": runs_ms,
                    "summary": latency_summary,
                    "repeat_stats": {
                        "repeat_means": repeat_means,
                        "repeat_p99s": repeat_p99s,
                        "between_repeat_cv": repeat_cv,
                    },
                    "outlier_count": outlier_cnt,
                },
                "resource": getrusage(),
                "ort_profile": {
                    "enabled": args.profile == "on",
                    "trace_path": profile_path,
                },
            })
        } else {
            json!({
                "enabled": false,
                "skipped": true,
                "reason": perf_skipped_reason,
            })
        },
        "env_info": env_info,
        "metrics_eval": if metrics_enabled {
            if let Some(reason) = metrics_skipped_reason {
                json!({
                    "enabled": false,
                    "skipped": true,
                    "reason": reason,
                })
            } else {
                json!({
                    "enabled": true,
                    "metrics_pack": metrics_pack_info,
                    "task_metrics": task_metrics,
                    "scores_out": scores_out_path,
                })
            }
        } else {
            json!({
                "enabled": false,
                "skipped": true,
                "reason": metrics_skipped_reason,
            })
        },
    });

    let report = json!({
        "model": args.model,
        "custom_op": args.custom_op,
        "manifest": args.manifest,
        "cmd": std::env::args().collect::<Vec<_>>().join(" "),
        "git_commit": manifest.get("git_commit").cloned().unwrap_or(json!("UNKNOWN")),
        "pt_filename": manifest.get("pt_filename").cloned().unwrap_or(json!("UNKNOWN")),
        "iters": args.iters,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "feature_dim": args.feature_dim,
        "static_dim": args.static_dim,
        "intra": args.intra,
        "inter": args.inter,
        "io_binding": "none",
        "d_inner": d_inner,
        "d_state": d_state,
        "ort_memory": {
            "cpu_mem_arena": args.cpu_mem_arena,
            "mem_pattern": args.mem_pattern,
            "mem_reuse": args.mem_reuse,
        },
        "env_info": metrics.get("env_info").cloned().unwrap_or(json!({})),
        "maxrss_kb": metrics
            .get("perf")
            .and_then(|v| v.get("resource"))
            .and_then(|v| v.get("ru_maxrss_kb"))
            .cloned()
            .unwrap_or(json!(0)),
        "mean_ms": metrics.get("perf").and_then(|v| v.get("latency")).and_then(|v| v.get("summary")).and_then(|v| v.get("mean")).cloned().unwrap_or(json!(0.0)),
        "p50_ms": metrics.get("perf").and_then(|v| v.get("latency")).and_then(|v| v.get("summary")).and_then(|v| v.get("p50")).cloned().unwrap_or(json!(0.0)),
        "p95_ms": metrics.get("perf").and_then(|v| v.get("latency")).and_then(|v| v.get("summary")).and_then(|v| v.get("p95")).cloned().unwrap_or(json!(0.0)),
        "p99_ms": metrics.get("perf").and_then(|v| v.get("latency")).and_then(|v| v.get("summary")).and_then(|v| v.get("p99")).cloned().unwrap_or(json!(0.0)),
        "min_ms": metrics.get("perf").and_then(|v| v.get("latency")).and_then(|v| v.get("summary")).and_then(|v| v.get("min")).cloned().unwrap_or(json!(0.0)),
        "max_ms": metrics.get("perf").and_then(|v| v.get("latency")).and_then(|v| v.get("summary")).and_then(|v| v.get("max")).cloned().unwrap_or(json!(0.0)),
        "runs_ms": metrics.get("perf").and_then(|v| v.get("latency")).and_then(|v| v.get("runs_ms")).cloned().unwrap_or(json!([])),
        "repeat_means": metrics
            .get("perf")
            .and_then(|v| v.get("latency"))
            .and_then(|v| v.get("repeat_stats"))
            .and_then(|v| v.get("repeat_means"))
            .cloned()
            .unwrap_or(json!([])),
        "between_repeat_cv": metrics
            .get("perf")
            .and_then(|v| v.get("latency"))
            .and_then(|v| v.get("repeat_stats"))
            .and_then(|v| v.get("between_repeat_cv"))
            .cloned()
            .unwrap_or(json!(0.0)),
        "outliers": metrics
            .get("perf")
            .and_then(|v| v.get("latency"))
            .and_then(|v| v.get("outlier_count"))
            .cloned()
            .unwrap_or(json!(0)),
        "metrics": metrics,
    });

    let out_path = PathBuf::from(&args.out);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    println!("OK: {}", args.out);
    Ok(())
}
