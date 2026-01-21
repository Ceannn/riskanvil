use libm::{log1pf, powf};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use xxhash_rust::xxh3::xxh3_64;

const DIM: usize = 32;
const RING: usize = 4;
const MERCHANT_RING: usize = 8;

const RESET_INACTIVE_SEC: u64 = 1_209_600; // 14d
const HARD_TTL_SEC: u64 = 2_592_000; // 30d
const DT_CLAMP_SEC: u64 = 2_592_000; // 30d
const TIME_LOG1P_MAX: f32 = 30.0;
const NEAR_BAND: f32 = 0.05;
const PRIOR_ALPHA: f32 = 0.02;
const EPS_RATIO: f32 = 1e-6;

const HL_5M: f32 = 300.0;
const HL_10M: f32 = 600.0;
const HL_1H: f32 = 3600.0;
const HL_1D: f32 = 86_400.0;
const HL_24H: f32 = 86_400.0;

const IDX_L1: usize = 0; // 0..4
const IDX_L2: usize = 4; // 4..8
const IDX_FINAL: usize = 8; // 8..12
const IDX_ROUTE: usize = 12; // 12..16
const IDX_NEAR_10M: usize = 16;
const IDX_NEAR_24H: usize = 17;
const IDX_EMA_TXN_5M: usize = 18;
const IDX_EMA_TXN_1H: usize = 19;
const IDX_EMA_TXN_1D: usize = 20;
const IDX_EMA_AMT_5M: usize = 21;
const IDX_EMA_AMT_1H: usize = 22;
const IDX_EMA_AMT_1D: usize = 23;
const IDX_TSL_TXN: usize = 24;
const IDX_TSL_REJECT: usize = 25;
const IDX_EMA_NEW_MERCH_1H: usize = 26;
const IDX_EMA_NEW_MERCH_1D: usize = 27;
const IDX_EMA_REJECT_ADDR_1H: usize = 28;
const IDX_EMA_REJECT_PROD_1H: usize = 29;
const IDX_NEW_NEIGHBOR_RATIO: usize = 30;
const IDX_RESERVED: usize = 31;

#[derive(Debug, Clone, Copy)]
pub enum Route {
    Pass,
    Refer,
    Reject,
}

impl Route {
    fn code(self) -> f32 {
        match self {
            Route::Pass => 0.0,
            Route::Refer => 1.0,
            Route::Reject => 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MicroEvent {
    pub event_ts_sec: u64,
    pub l1_margin: f32,
    pub l2_margin: f32,
    pub delta_margin: f32,
    pub route: Route,
    pub amount_log1p: f32,
    pub thr01: f32,
    pub merchant_id_hash: Option<u64>,
    pub addr_hash: Option<u64>,
    pub prod_hash: Option<u64>,
}

#[derive(Debug, Clone)]
struct AuxState {
    last_ts_sec: u64,
    last_txn_ts_sec: u64,
    last_reject_ts_sec: u64,
    last_reject_addr_hash: u64,
    last_reject_prod_hash: u64,
    recent_merchants: [u64; MERCHANT_RING],
    recent_merchant_len: u8,
    recent_merchant_pos: u8,
}

impl AuxState {
    fn new() -> Self {
        Self {
            last_ts_sec: 0,
            last_txn_ts_sec: 0,
            last_reject_ts_sec: 0,
            last_reject_addr_hash: 0,
            last_reject_prod_hash: 0,
            recent_merchants: [0; MERCHANT_RING],
            recent_merchant_len: 0,
            recent_merchant_pos: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MicroState {
    vec: [f32; DIM],
    aux: AuxState,
}

impl MicroState {
    pub fn new() -> Self {
        Self {
            vec: [0.0; DIM],
            aux: AuxState::new(),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.vec
    }

    pub fn reset(&mut self) {
        self.vec = [0.0; DIM];
        self.aux = AuxState::new();
    }

    pub fn prepare_for_event(&mut self, event_ts_sec: u64) {
        let dt_since_last = event_ts_sec.saturating_sub(self.aux.last_ts_sec);
        if self.aux.last_ts_sec != 0
            && (dt_since_last > RESET_INACTIVE_SEC || dt_since_last > HARD_TTL_SEC)
        {
            self.reset();
        }

        let tsl_txn = if self.aux.last_txn_ts_sec == 0 {
            0.0
        } else {
            let dt = event_ts_sec.saturating_sub(self.aux.last_txn_ts_sec).min(DT_CLAMP_SEC);
            log1pf(dt as f32).min(TIME_LOG1P_MAX)
        };
        let tsl_reject = if self.aux.last_reject_ts_sec == 0 {
            0.0
        } else {
            let dt = event_ts_sec
                .saturating_sub(self.aux.last_reject_ts_sec)
                .min(DT_CLAMP_SEC);
            log1pf(dt as f32).min(TIME_LOG1P_MAX)
        };
        self.vec[IDX_TSL_TXN] = tsl_txn;
        self.vec[IDX_TSL_REJECT] = tsl_reject;
    }

    pub fn update_for_event(&mut self, event: &MicroEvent) {
        let dt_raw = event.event_ts_sec.saturating_sub(self.aux.last_ts_sec);
        if self.aux.last_ts_sec != 0 && (dt_raw > RESET_INACTIVE_SEC || dt_raw > HARD_TTL_SEC) {
            self.reset();
        }

        let dt = dt_raw.min(DT_CLAMP_SEC) as f32;
        let decay_5m = decay(dt, HL_5M);
        let decay_10m = decay(dt, HL_10M);
        let decay_1h = decay(dt, HL_1H);
        let decay_1d = decay(dt, HL_1D);
        let decay_24h = decay(dt, HL_24H);

        let final_margin = event.l2_margin + PRIOR_ALPHA * event.delta_margin;
        shift4(&mut self.vec, IDX_L1, event.l1_margin);
        shift4(&mut self.vec, IDX_L2, event.l2_margin);
        shift4(&mut self.vec, IDX_FINAL, final_margin);
        shift4(&mut self.vec, IDX_ROUTE, event.route.code());

        let is_near = (event.l2_margin - event.thr01).abs() <= NEAR_BAND;
        let near_val = if is_near { 1.0 } else { 0.0 };

        self.vec[IDX_NEAR_10M] = decayed_sum(self.vec[IDX_NEAR_10M], near_val, decay_10m, 1_000_000.0);
        self.vec[IDX_NEAR_24H] = decayed_sum(self.vec[IDX_NEAR_24H], near_val, decay_24h, 1_000_000.0);

        self.vec[IDX_EMA_TXN_5M] = decayed_sum(self.vec[IDX_EMA_TXN_5M], 1.0, decay_5m, 1_000_000.0);
        self.vec[IDX_EMA_TXN_1H] = decayed_sum(self.vec[IDX_EMA_TXN_1H], 1.0, decay_1h, 1_000_000.0);
        self.vec[IDX_EMA_TXN_1D] = decayed_sum(self.vec[IDX_EMA_TXN_1D], 1.0, decay_1d, 1_000_000.0);

        self.vec[IDX_EMA_AMT_5M] = decayed_sum(self.vec[IDX_EMA_AMT_5M], event.amount_log1p, decay_5m, 1_000_000_000.0);
        self.vec[IDX_EMA_AMT_1H] = decayed_sum(self.vec[IDX_EMA_AMT_1H], event.amount_log1p, decay_1h, 1_000_000_000.0);
        self.vec[IDX_EMA_AMT_1D] = decayed_sum(self.vec[IDX_EMA_AMT_1D], event.amount_log1p, decay_1d, 1_000_000_000.0);

        let (is_new_merch, updated_ring) = update_recent_merchants(
            &mut self.aux,
            event.merchant_id_hash.unwrap_or(0),
        );
        let new_merch_val = if is_new_merch { 1.0 } else { 0.0 };
        self.vec[IDX_EMA_NEW_MERCH_1H] = decayed_sum(self.vec[IDX_EMA_NEW_MERCH_1H], new_merch_val, decay_1h, 1_000_000.0);
        self.vec[IDX_EMA_NEW_MERCH_1D] = decayed_sum(self.vec[IDX_EMA_NEW_MERCH_1D], new_merch_val, decay_1d, 1_000_000.0);
        let _ = updated_ring;

        let prev_addr = self.aux.last_reject_addr_hash;
        let prev_prod = self.aux.last_reject_prod_hash;
        let addr_hash = event.addr_hash.unwrap_or(0);
        let prod_hash = event.prod_hash.unwrap_or(0);
        let is_reject = matches!(event.route, Route::Reject);
        let same_addr = is_reject && addr_hash != 0 && addr_hash == prev_addr;
        let same_prod = is_reject && prod_hash != 0 && prod_hash == prev_prod;

        self.vec[IDX_EMA_REJECT_ADDR_1H] = decayed_sum(
            self.vec[IDX_EMA_REJECT_ADDR_1H],
            if same_addr { 1.0 } else { 0.0 },
            decay_1h,
            1_000_000.0,
        );
        self.vec[IDX_EMA_REJECT_PROD_1H] = decayed_sum(
            self.vec[IDX_EMA_REJECT_PROD_1H],
            if same_prod { 1.0 } else { 0.0 },
            decay_1h,
            1_000_000.0,
        );

        let denom = (self.vec[IDX_EMA_TXN_1D]).max(EPS_RATIO);
        let ratio = (self.vec[IDX_EMA_NEW_MERCH_1D] / denom).clamp(0.0, 1.0);
        self.vec[IDX_NEW_NEIGHBOR_RATIO] = ratio;

        self.vec[IDX_RESERVED] = 0.0;

        self.aux.last_ts_sec = event.event_ts_sec;
        self.aux.last_txn_ts_sec = event.event_ts_sec;
        if is_reject {
            self.aux.last_reject_ts_sec = event.event_ts_sec;
            self.aux.last_reject_addr_hash = addr_hash;
            self.aux.last_reject_prod_hash = prod_hash;
        }

        // time_since values reset to 0 right after the event
        self.vec[IDX_TSL_TXN] = 0.0;
        if is_reject {
            self.vec[IDX_TSL_REJECT] = 0.0;
        }
        self.vec[IDX_RESERVED] = 0.0;
    }
}

fn shift4(vec: &mut [f32; DIM], base: usize, new_val: f32) {
    for i in (1..RING).rev() {
        vec[base + i] = vec[base + i - 1];
    }
    vec[base] = new_val;
}

fn decay(dt: f32, half_life: f32) -> f32 {
    if half_life <= 0.0 {
        0.0
    } else {
        powf(0.5, dt / half_life)
    }
}

fn decayed_sum(prev: f32, x: f32, decay: f32, max_val: f32) -> f32 {
    let mut v = prev * decay + x;
    if v > max_val {
        v = max_val;
    }
    v
}

fn update_recent_merchants(aux: &mut AuxState, merchant_hash: u64) -> (bool, bool) {
    if merchant_hash == 0 {
        return (false, false);
    }
    let len = aux.recent_merchant_len as usize;
    for i in 0..len {
        if aux.recent_merchants[i] == merchant_hash {
            return (false, false);
        }
    }
    let pos = aux.recent_merchant_pos as usize % MERCHANT_RING;
    aux.recent_merchants[pos] = merchant_hash;
    aux.recent_merchant_pos = (pos as u8 + 1) % MERCHANT_RING as u8;
    if aux.recent_merchant_len < MERCHANT_RING as u8 {
        aux.recent_merchant_len += 1;
    }
    (true, true)
}

pub fn hash_entity_key(card_id: Option<u64>, uid: Option<u64>) -> Option<u64> {
    if let Some(card) = card_id {
        let bytes = card.to_le_bytes();
        return Some(xxh3_64(&bytes));
    }
    if let Some(uid) = uid {
        let bytes = uid.to_le_bytes();
        return Some(xxh3_64(&bytes));
    }
    None
}

pub struct MicroStateCache {
    shards: Vec<Mutex<HashMap<u64, MicroState>>>,
    mask: usize,
}

impl MicroStateCache {
    pub fn new(shards: usize) -> Self {
        let shards = shards.next_power_of_two().max(1);
        let mut vec = Vec::with_capacity(shards);
        for _ in 0..shards {
            vec.push(Mutex::new(HashMap::new()));
        }
        Self {
            shards: vec,
            mask: shards - 1,
        }
    }

    fn shard(&self, key: u64) -> MutexGuard<'_, HashMap<u64, MicroState>> {
        let idx = (key as usize) & self.mask;
        self.shards[idx].lock().unwrap()
    }

    pub fn read_for_event(&self, key: u64, event_ts_sec: u64) -> [f32; DIM] {
        let mut shard = self.shard(key);
        let state = shard.entry(key).or_insert_with(MicroState::new);
        state.prepare_for_event(event_ts_sec);
        state.vec
    }

    pub fn read_for_event_optional(&self, key: Option<u64>, event_ts_sec: u64) -> [f32; DIM] {
        if let Some(k) = key {
            self.read_for_event(k, event_ts_sec)
        } else {
            [0.0; DIM]
        }
    }

    pub fn update_for_event(&self, key: u64, event: &MicroEvent) {
        let mut shard = self.shard(key);
        let state = shard.entry(key).or_insert_with(MicroState::new);
        state.update_for_event(event);
    }

    pub fn update_for_event_optional(&self, key: Option<u64>, event: &MicroEvent) {
        if let Some(k) = key {
            self.update_for_event(k, event);
        }
    }
}
