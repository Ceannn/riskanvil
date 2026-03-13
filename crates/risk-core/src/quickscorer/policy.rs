use crate::quickscorer::types::QuickRouteMeta;
use anyhow::{anyhow, bail, Context};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum L2FeatureSource {
    FromL1(usize),
    L1Score,
}

#[derive(Debug, Clone)]
pub struct L2Policy {
    pub dim: usize,
    pub feature_sources: Vec<L2FeatureSource>,
    pub seg_enabled: bool,
    pub seg_cols: Vec<String>,
    pub seg_indices: Vec<usize>,
    pub tau_map_by_fold: HashMap<i32, HashMap<Vec<u8>, f32>>,
    pub tau_global_by_fold: BTreeMap<i32, f32>,
    pub default_fold: i32,
    pub gb_target: String,
}

#[derive(Debug, Clone)]
pub struct QuickPolicy {
    pub l1_dim: usize,
    pub l1_threshold: f32,
    pub l2: Option<L2Policy>,
}

#[derive(Debug, Clone, Copy)]
pub struct ResolvedL2Threshold {
    pub fold: i32,
    pub tau: f32,
}

impl QuickPolicy {
    pub fn load_bundle(bundle_dir: &Path) -> anyhow::Result<Self> {
        let l1_model_dir =
            bundle_dir.join("quickscorer_l1l2_single_l512_cascade_20260304/models/quickscorer_l1");
        let l2_model_dir =
            bundle_dir.join("quickscorer_l1l2_single_l512_cascade_20260304/models/quickscorer_l2");
        Self::load(&l1_model_dir, Some(&l2_model_dir))
    }

    pub fn load(l1_model_dir: &Path, l2_model_dir: Option<&Path>) -> anyhow::Result<Self> {
        let l1_feature_names = load_feature_names(l1_model_dir)?;
        let l1_dim = l1_feature_names.len();
        let l1_threshold = load_l1_threshold(l1_model_dir)?;

        let l2 = if let Some(l2_dir) = l2_model_dir {
            Some(load_l2_policy(
                l1_model_dir,
                l2_dir,
                &l1_feature_names,
                l1_dim,
            )?)
        } else {
            None
        };

        Ok(Self {
            l1_dim,
            l1_threshold,
            l2,
        })
    }
}

impl L2Policy {
    pub fn resolve_threshold(&self, row_l2: &[f32]) -> anyhow::Result<f32> {
        let mut seg_key_buf = Vec::new();
        Ok(self
            .resolve_threshold_with_route_meta(row_l2, None, &mut seg_key_buf)?
            .tau)
    }

    pub fn resolve_threshold_with_buf(
        &self,
        row_l2: &[f32],
        seg_key_buf: &mut Vec<u8>,
    ) -> anyhow::Result<f32> {
        Ok(self
            .resolve_threshold_with_route_meta(row_l2, None, seg_key_buf)?
            .tau)
    }

    pub fn resolve_threshold_with_route_meta(
        &self,
        row_l2: &[f32],
        route_meta: Option<&QuickRouteMeta>,
        seg_key_buf: &mut Vec<u8>,
    ) -> anyhow::Result<ResolvedL2Threshold> {
        if let Some(tau) = route_meta.and_then(|x| x.l2_tau_used) {
            let fold = self.select_fold(route_meta);
            return Ok(ResolvedL2Threshold { fold, tau });
        }
        let fold = self.select_fold(route_meta);
        if self.seg_enabled {
            self.build_seg_key_bytes(row_l2, route_meta, seg_key_buf);
            if let Some(by_key) = self.tau_map_by_fold.get(&fold) {
                if let Some(v) = by_key.get(seg_key_buf.as_slice()) {
                    return Ok(ResolvedL2Threshold { fold, tau: *v });
                }
            }
        }

        if let Some(v) = self.tau_global_by_fold.get(&fold) {
            return Ok(ResolvedL2Threshold { fold, tau: *v });
        }
        if let Some((_, v)) = self.tau_global_by_fold.first_key_value() {
            return Ok(ResolvedL2Threshold { fold, tau: *v });
        }
        bail!("l2 threshold unavailable (no global tau and no segmented hit)")
    }

    fn select_fold(&self, route_meta: Option<&QuickRouteMeta>) -> i32 {
        let requested = route_meta.map(|x| x.fold_id).unwrap_or(self.default_fold);
        if self.tau_global_by_fold.contains_key(&requested) {
            requested
        } else {
            self.tau_global_by_fold
                .first_key_value()
                .map(|(k, _)| *k)
                .unwrap_or(requested)
        }
    }

    fn build_seg_key_bytes(
        &self,
        row_l2: &[f32],
        route_meta: Option<&QuickRouteMeta>,
        out: &mut Vec<u8>,
    ) {
        if let Some(meta) = route_meta {
            out.clear();
            let mut buf = itoa::Buffer::new();
            out.extend_from_slice(buf.format(meta.seg_prod_amtbin).as_bytes());
            return;
        }
        out.clear();
        for (pos, idx) in self.seg_indices.iter().enumerate() {
            if pos != 0 {
                out.push(b'|');
            }
            let v = row_l2.get(*idx).copied().unwrap_or(f32::NAN);
            write_canonical_seg_component_from_f32(v, out);
        }
    }
}

fn load_l1_threshold(l1_model_dir: &Path) -> anyhow::Result<f32> {
    let policy_path = l1_model_dir.join("policy.json");
    let text = fs::read_to_string(&policy_path)
        .with_context(|| format!("read l1 policy: {}", policy_path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse l1 policy: {}", policy_path.display()))?;
    let thr = v
        .pointer("/decision/threshold")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| anyhow!("l1 policy missing /decision/threshold"))?;
    Ok(thr as f32)
}

fn load_l2_policy(
    _l1_model_dir: &Path,
    l2_model_dir: &Path,
    l1_feature_names: &[String],
    _l1_dim: usize,
) -> anyhow::Result<L2Policy> {
    let l2_feature_names = load_feature_names(l2_model_dir)?;
    let l2_dim = l2_feature_names.len();

    let mut l1_idx: HashMap<&str, usize> = HashMap::with_capacity(l1_feature_names.len());
    for (i, n) in l1_feature_names.iter().enumerate() {
        l1_idx.insert(n.as_str(), i);
    }

    let mut feature_sources = Vec::with_capacity(l2_dim);
    for n in &l2_feature_names {
        if is_l1_score_feature(n) {
            feature_sources.push(L2FeatureSource::L1Score);
            continue;
        }
        if let Some(i) = l1_idx.get(n.as_str()) {
            feature_sources.push(L2FeatureSource::FromL1(*i));
            continue;
        }
        bail!(
            "l2 feature '{}' cannot be sourced from l1 row and is not a supported derived feature",
            n
        );
    }

    let policy_path = l2_model_dir.join("policy.json");
    let text = fs::read_to_string(&policy_path)
        .with_context(|| format!("read l2 policy: {}", policy_path.display()))?;
    let v: Value = serde_json::from_str(&text)
        .with_context(|| format!("parse l2 policy: {}", policy_path.display()))?;

    let gb_target = std::env::var("QS_L2_GB_TARGET").unwrap_or_else(|_| {
        v.pointer("/thresholds/default_gb_target")
            .and_then(|x| x.as_str())
            .unwrap_or("0.0005")
            .to_string()
    });
    let gb_target_f64 = gb_target.parse::<f64>().unwrap_or(0.0005);

    let mut tau_global_by_fold: BTreeMap<i32, f32> = BTreeMap::new();
    if let Some(obj) = v
        .pointer(&format!("/thresholds/tau_global_by_fold/{}", gb_target))
        .and_then(|x| x.as_object())
    {
        for (k, vv) in obj {
            if let (Ok(fold), Some(tau)) = (k.parse::<i32>(), vv.as_f64()) {
                tau_global_by_fold.insert(fold, tau as f32);
            }
        }
    }
    if tau_global_by_fold.is_empty() {
        bail!(
            "l2 policy missing thresholds.tau_global_by_fold['{}']",
            gb_target
        );
    }

    let default_fold = std::env::var("QS_FOLD_ID")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1);

    let seg_enabled = v
        .pointer("/segmented_threshold/enabled")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let seg_cols: Vec<String> = v
        .pointer("/segmented_threshold/seg_cols")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(ToOwned::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let tau_table_file = v
        .pointer("/segmented_threshold/tau_table_file")
        .and_then(|x| x.as_str())
        .unwrap_or("seg_tau.tsv")
        .to_string();
    let tau_field = v
        .pointer("/segmented_threshold/tau_field")
        .and_then(|x| x.as_str())
        .unwrap_or("tau_shrink")
        .to_string();

    let mut seg_indices = Vec::with_capacity(seg_cols.len());
    for c in &seg_cols {
        let idx = l2_feature_names
            .iter()
            .position(|x| x == c)
            .ok_or_else(|| anyhow!("seg col '{}' not found in l2 feature_names", c))?;
        seg_indices.push(idx);
    }

    let tau_map_by_fold = if seg_enabled {
        load_seg_tau_map(
            &l2_model_dir.join(tau_table_file),
            gb_target_f64,
            &tau_field,
        )?
    } else {
        HashMap::new()
    };

    Ok(L2Policy {
        dim: l2_dim,
        feature_sources,
        seg_enabled,
        seg_cols,
        seg_indices,
        tau_map_by_fold,
        tau_global_by_fold,
        default_fold,
        gb_target,
    })
}

fn load_seg_tau_map(
    path: &Path,
    gb_target: f64,
    tau_field: &str,
) -> anyhow::Result<HashMap<i32, HashMap<Vec<u8>, f32>>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read seg tau table: {}", path.display()))?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| anyhow!("empty seg tau table: {}", path.display()))?;
    let cols: Vec<&str> = header.split('\t').collect();
    let idx_fold = find_col(&cols, "fold")?;
    let idx_gb = find_col(&cols, "gb_target")?;
    let idx_seg = find_col(&cols, "seg_key")?;
    let idx_tau = find_col(&cols, tau_field)?;

    let mut out: HashMap<i32, HashMap<Vec<u8>, f32>> = HashMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() <= idx_tau || parts.len() <= idx_seg || parts.len() <= idx_fold {
            continue;
        }

        let Ok(gb) = parts[idx_gb].trim().parse::<f64>() else {
            continue;
        };
        if !float_eq(gb, gb_target) {
            continue;
        }

        let Ok(fold) = parts[idx_fold].trim().parse::<i32>() else {
            continue;
        };
        let Ok(tau) = parts[idx_tau].trim().parse::<f32>() else {
            continue;
        };
        let seg_key = canonical_seg_component_from_text(parts[idx_seg]).into_bytes();
        out.entry(fold).or_default().insert(seg_key, tau);
    }

    Ok(out)
}

fn load_feature_names(model_dir: &Path) -> anyhow::Result<Vec<String>> {
    let p = model_dir.join("feature_names.json");
    let text =
        fs::read_to_string(&p).with_context(|| format!("read feature_names: {}", p.display()))?;
    let names: Vec<String> = serde_json::from_str(&text)
        .with_context(|| format!("parse feature_names: {}", p.display()))?;
    if names.is_empty() {
        bail!("empty feature_names: {}", p.display());
    }
    Ok(names)
}

fn is_l1_score_feature(name: &str) -> bool {
    matches!(
        name,
        "l1_1_oof_score" | "l1_oof_score" | "l1_score" | "l1_margin"
    )
}

fn find_col(cols: &[&str], name: &str) -> anyhow::Result<usize> {
    cols.iter()
        .position(|x| *x == name)
        .ok_or_else(|| anyhow!("missing column '{}' in seg tau header", name))
}

fn float_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

pub fn canonical_seg_component_from_f32(v: f32) -> String {
    if !v.is_finite() {
        return "NA".to_string();
    }
    let fv = v as f64;
    if fv.fract().abs() <= 1e-9 {
        return format!("{}", fv as i64);
    }
    trim_float_string(format!("{:.15}", fv))
}

fn write_canonical_seg_component_from_f32(v: f32, out: &mut Vec<u8>) {
    if !v.is_finite() {
        out.extend_from_slice(b"NA");
        return;
    }
    let fv = v as f64;
    if fv.fract().abs() <= 1e-9 {
        let mut buf = itoa::Buffer::new();
        out.extend_from_slice(buf.format(fv as i64).as_bytes());
        return;
    }
    let mut buf = ryu::Buffer::new();
    let s = buf.format_finite(fv);
    let bytes = s.as_bytes();
    let has_dot = bytes.contains(&b'.');
    let mut end = bytes.len();
    while has_dot && end > 0 && bytes[end - 1] == b'0' {
        end -= 1;
    }
    if end > 0 && bytes[end - 1] == b'.' {
        end -= 1;
    }
    if end == 0 {
        out.push(b'0');
    } else {
        out.extend_from_slice(&bytes[..end]);
    }
}

pub fn canonical_seg_component_from_text(s: &str) -> String {
    let t = s.trim();
    if t.is_empty() {
        return "NA".to_string();
    }
    if let Ok(v) = t.parse::<f64>() {
        if v.is_finite() && v.fract().abs() <= 1e-12 {
            return format!("{}", v as i64);
        }
    }
    t.to_string()
}

fn trim_float_string(mut s: String) -> String {
    while s.contains('.') && s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    if s.is_empty() {
        "0".to_string()
    } else {
        s
    }
}
