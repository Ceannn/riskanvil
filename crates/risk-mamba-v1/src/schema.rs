use crate::error::MicroStateError;
use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct MicroStateSchema {
    pub schema_name: String,
    pub schema_version: u32,
    pub dtype: String,
    pub endianness: String,
    pub dim: usize,
    pub byte_len: usize,
    pub alignment_bytes: usize,
    pub composition: Composition,
    pub entity_key: EntityKey,
    pub time: TimeSpec,
    pub ttl_policy: TtlPolicy,
    pub constants: Constants,
    pub ema_halflife_sec: EmaHalflife,
    pub fields: Vec<FieldSpec>,
    pub aux_state_store: AuxStateStore,
    pub update_contract: UpdateContract,
}

#[derive(Debug, Deserialize)]
pub struct Composition {
    pub static_v1_1: StaticV1_1,
}

#[derive(Debug, Deserialize)]
pub struct StaticV1_1 {
    pub static_base_dim: usize,
    pub micro_state_dim: usize,
    pub concat_order: Vec<String>,
    pub static_total_dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct EntityKey {
    pub key_kind: String,
    pub key_priority: Vec<String>,
    pub hash: String,
}

#[derive(Debug, Deserialize)]
pub struct TimeSpec {
    pub time_unit: String,
    pub event_time_field: String,
    pub dt_clamp_sec: u64,
}

#[derive(Debug, Deserialize)]
pub struct TtlPolicy {
    pub reset_on_inactive_sec: u64,
    pub hard_expire_sec: u64,
    pub reset_action: String,
    pub reset_sets_last_ts_to: u64,
}

#[derive(Debug, Deserialize)]
pub struct Constants {
    pub near_band_default: f32,
    pub route_code: RouteCodeSpec,
    pub amount_transform: String,
    pub time_since_transform: String,
    pub time_since_clip_sec: u64,
}

#[derive(Debug, Deserialize)]
pub struct RouteCodeSpec {
    pub pass: f32,
    pub refer: f32,
    pub reject: f32,
    pub unknown: f32,
}

#[derive(Debug, Deserialize)]
pub struct EmaHalflife {
    pub h_5m: u64,
    pub h_10m: u64,
    pub h_1h: u64,
    pub h_24h: u64,
    pub h_1d: u64,
}

#[derive(Debug, Deserialize)]
pub struct FieldSpec {
    pub i: usize,
    pub name: String,
    pub dtype: String,
    pub desc: String,
    pub update: String,
}

#[derive(Debug, Deserialize)]
pub struct AuxStateStore {
    pub not_model_inputs: bool,
    pub alignment_bytes: usize,
    pub fields: Vec<AuxFieldSpec>,
}

#[derive(Debug, Deserialize)]
pub struct AuxFieldSpec {
    pub name: String,
    pub dtype: String,
    pub desc: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateContract {
    pub call_order: Vec<String>,
    pub ema_formula: String,
}

impl MicroStateSchema {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, MicroStateError> {
        let mut file = File::open(path)?;
        let mut buf = String::new();
        file.read_to_string(&mut buf)?;
        let schema: MicroStateSchema = serde_json::from_str(&buf)?;
        schema.validate()?;
        Ok(schema)
    }

    pub fn validate(&self) -> Result<(), MicroStateError> {
        if self.schema_version != 1 {
            return Err(MicroStateError::BadSchema(format!(
                "schema_version {}",
                self.schema_version
            )));
        }
        if self.dtype != "f32" {
            return Err(MicroStateError::BadSchema(format!(
                "dtype {}",
                self.dtype
            )));
        }
        if self.endianness != "little" {
            return Err(MicroStateError::BadSchema(format!(
                "endianness {}",
                self.endianness
            )));
        }
        if self.dim != 32 || self.byte_len != 128 {
            return Err(MicroStateError::BadSchema(format!(
                "dim/byte_len {}/{}",
                self.dim, self.byte_len
            )));
        }
        if self.composition.static_v1_1.static_base_dim != 386 {
            return Err(MicroStateError::BadSchema(format!(
                "static_base_dim {}",
                self.composition.static_v1_1.static_base_dim
            )));
        }
        if self.composition.static_v1_1.micro_state_dim != 32 {
            return Err(MicroStateError::BadSchema(format!(
                "micro_state_dim {}",
                self.composition.static_v1_1.micro_state_dim
            )));
        }
        if self.composition.static_v1_1.static_total_dim != 418 {
            return Err(MicroStateError::BadSchema(format!(
                "static_total_dim {}",
                self.composition.static_v1_1.static_total_dim
            )));
        }
        if self.composition.static_v1_1.concat_order != ["static_base", "micro_state"] {
            return Err(MicroStateError::BadSchema("concat_order".to_string()));
        }
        Ok(())
    }
}
