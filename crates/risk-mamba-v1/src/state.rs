use crate::error::MicroStateError;

pub const MICRO_STATE_DIM: usize = 32;
pub const STATIC_BASE_DIM: usize = 386;
pub const STATIC_TOTAL_DIM: usize = 418;

pub const IDX_L2_MARGIN_T0: usize = 0;
pub const IDX_FINAL_MARGIN_T0: usize = 4;
pub const IDX_ROUTE_CODE_T0: usize = 8;

pub const IDX_EMA_TXN_CNT_5M: usize = 12;
pub const IDX_EMA_TXN_CNT_1H: usize = 13;
pub const IDX_EMA_TXN_CNT_1D: usize = 14;

pub const IDX_EMA_AMT_5M: usize = 15;
pub const IDX_EMA_AMT_1H: usize = 16;
pub const IDX_EMA_AMT_1D: usize = 17;

pub const IDX_EMA_REFER_5M: usize = 18;
pub const IDX_EMA_REFER_1H: usize = 19;
pub const IDX_EMA_REFER_1D: usize = 20;

pub const IDX_EMA_REJECT_5M: usize = 21;
pub const IDX_EMA_REJECT_1H: usize = 22;
pub const IDX_EMA_REJECT_1D: usize = 23;

pub const IDX_TSL_TXN: usize = 24;
pub const IDX_TSL_REFER: usize = 25;
pub const IDX_TSL_REJECT: usize = 26;

pub const IDX_EMA_NEAR_THR_10M: usize = 27;
pub const IDX_EMA_NEAR_THR_24H: usize = 28;

pub const IDX_EMA_SWITCH_MERCHANT_1H: usize = 29;
pub const IDX_EMA_SWITCH_DEVICE_1H: usize = 30;

pub const IDX_NEAR_THR_STREAK: usize = 31;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct MicroState {
    pub v: [f32; MICRO_STATE_DIM],
}

impl Default for MicroState {
    fn default() -> Self {
        Self { v: [0.0; MICRO_STATE_DIM] }
    }
}

impl MicroState {
    pub fn reset(&mut self) {
        self.v = [0.0; MICRO_STATE_DIM];
    }

    pub fn apply_event(
        &mut self,
        aux: &mut MicroStateAux,
        params: &MicroStateParams,
        event: &MicroStateEvent,
    ) -> Result<(), MicroStateError> {
        if aux.last_ts_sec != 0 {
            let gap = event
                .event_ts_sec
                .saturating_sub(aux.last_ts_sec);
            if gap > params.reset_on_inactive_sec || gap > params.hard_expire_sec {
                self.reset();
                aux.reset();
            }
        }

        let dt_sec = if aux.last_ts_sec == 0 {
            0
        } else {
            let raw = event.event_ts_sec.saturating_sub(aux.last_ts_sec);
            raw.min(params.dt_clamp_sec)
        };
        let dt = dt_sec as f32;

        let is_refer = event.is_refer;
        let is_reject = event.is_reject;
        let is_near_thr = (event.final_margin - event.thr_final).abs() <= params.near_band;
        let is_merchant_switch =
            aux.last_merchant_hash != 0
                && event.merchant_hash != 0
                && aux.last_merchant_hash != event.merchant_hash;
        let is_device_switch =
            aux.last_device_hash != 0
                && event.device_hash != 0
                && aux.last_device_hash != event.device_hash;

        shift4(&mut self.v, IDX_L2_MARGIN_T0, event.l2_margin);
        shift4(&mut self.v, IDX_FINAL_MARGIN_T0, event.final_margin);
        shift4(&mut self.v, IDX_ROUTE_CODE_T0, event.route_code.as_f32());

        self.v[IDX_EMA_TXN_CNT_5M] = ema_update(
            self.v[IDX_EMA_TXN_CNT_5M],
            1.0,
            dt,
            params.halflife_5m,
        );
        self.v[IDX_EMA_TXN_CNT_1H] = ema_update(
            self.v[IDX_EMA_TXN_CNT_1H],
            1.0,
            dt,
            params.halflife_1h,
        );
        self.v[IDX_EMA_TXN_CNT_1D] = ema_update(
            self.v[IDX_EMA_TXN_CNT_1D],
            1.0,
            dt,
            params.halflife_1d,
        );

        let amt = if event.amount < 0.0 { 0.0 } else { event.amount };
        let amt_log1p = amt.ln_1p();
        self.v[IDX_EMA_AMT_5M] = ema_update(
            self.v[IDX_EMA_AMT_5M],
            amt_log1p,
            dt,
            params.halflife_5m,
        );
        self.v[IDX_EMA_AMT_1H] = ema_update(
            self.v[IDX_EMA_AMT_1H],
            amt_log1p,
            dt,
            params.halflife_1h,
        );
        self.v[IDX_EMA_AMT_1D] = ema_update(
            self.v[IDX_EMA_AMT_1D],
            amt_log1p,
            dt,
            params.halflife_1d,
        );

        self.v[IDX_EMA_REFER_5M] = ema_update(
            self.v[IDX_EMA_REFER_5M],
            bool_to_f32(is_refer),
            dt,
            params.halflife_5m,
        );
        self.v[IDX_EMA_REFER_1H] = ema_update(
            self.v[IDX_EMA_REFER_1H],
            bool_to_f32(is_refer),
            dt,
            params.halflife_1h,
        );
        self.v[IDX_EMA_REFER_1D] = ema_update(
            self.v[IDX_EMA_REFER_1D],
            bool_to_f32(is_refer),
            dt,
            params.halflife_1d,
        );

        self.v[IDX_EMA_REJECT_5M] = ema_update(
            self.v[IDX_EMA_REJECT_5M],
            bool_to_f32(is_reject),
            dt,
            params.halflife_5m,
        );
        self.v[IDX_EMA_REJECT_1H] = ema_update(
            self.v[IDX_EMA_REJECT_1H],
            bool_to_f32(is_reject),
            dt,
            params.halflife_1h,
        );
        self.v[IDX_EMA_REJECT_1D] = ema_update(
            self.v[IDX_EMA_REJECT_1D],
            bool_to_f32(is_reject),
            dt,
            params.halflife_1d,
        );

        self.v[IDX_EMA_NEAR_THR_10M] = ema_update(
            self.v[IDX_EMA_NEAR_THR_10M],
            bool_to_f32(is_near_thr),
            dt,
            params.halflife_10m,
        );
        self.v[IDX_EMA_NEAR_THR_24H] = ema_update(
            self.v[IDX_EMA_NEAR_THR_24H],
            bool_to_f32(is_near_thr),
            dt,
            params.halflife_24h,
        );

        self.v[IDX_EMA_SWITCH_MERCHANT_1H] = ema_update(
            self.v[IDX_EMA_SWITCH_MERCHANT_1H],
            bool_to_f32(is_merchant_switch),
            dt,
            params.halflife_1h,
        );
        self.v[IDX_EMA_SWITCH_DEVICE_1H] = ema_update(
            self.v[IDX_EMA_SWITCH_DEVICE_1H],
            bool_to_f32(is_device_switch),
            dt,
            params.halflife_1h,
        );

        self.v[IDX_TSL_TXN] =
            time_since_log1p(event.event_ts_sec, aux.last_txn_ts_sec, params.time_since_clip_sec);
        self.v[IDX_TSL_REFER] =
            time_since_log1p(event.event_ts_sec, aux.last_refer_ts_sec, params.time_since_clip_sec);
        self.v[IDX_TSL_REJECT] = time_since_log1p(
            event.event_ts_sec,
            aux.last_reject_ts_sec,
            params.time_since_clip_sec,
        );

        self.v[IDX_NEAR_THR_STREAK] = update_streak(
            self.v[IDX_NEAR_THR_STREAK],
            is_near_thr,
            params.streak_cap,
        );

        aux.last_ts_sec = event.event_ts_sec;
        if event.is_txn {
            aux.last_txn_ts_sec = event.event_ts_sec;
        }
        if is_refer {
            aux.last_refer_ts_sec = event.event_ts_sec;
        }
        if is_reject {
            aux.last_reject_ts_sec = event.event_ts_sec;
        }
        aux.last_merchant_hash = event.merchant_hash;
        aux.last_device_hash = event.device_hash;

        Ok(())
    }
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct MicroStateAux {
    pub last_ts_sec: u64,
    pub last_txn_ts_sec: u64,
    pub last_refer_ts_sec: u64,
    pub last_reject_ts_sec: u64,
    pub last_merchant_hash: u64,
    pub last_device_hash: u64,
}

impl Default for MicroStateAux {
    fn default() -> Self {
        Self {
            last_ts_sec: 0,
            last_txn_ts_sec: 0,
            last_refer_ts_sec: 0,
            last_reject_ts_sec: 0,
            last_merchant_hash: 0,
            last_device_hash: 0,
        }
    }
}

impl MicroStateAux {
    pub fn reset(&mut self) {
        *self = MicroStateAux::default();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MicroStateParams {
    pub near_band: f32,
    pub reset_on_inactive_sec: u64,
    pub hard_expire_sec: u64,
    pub dt_clamp_sec: u64,
    pub time_since_clip_sec: u64,
    pub halflife_5m: f32,
    pub halflife_10m: f32,
    pub halflife_1h: f32,
    pub halflife_24h: f32,
    pub halflife_1d: f32,
    pub streak_cap: f32,
}

impl Default for MicroStateParams {
    fn default() -> Self {
        Self {
            near_band: 0.20,
            reset_on_inactive_sec: 1_209_600,
            hard_expire_sec: 7_776_000,
            dt_clamp_sec: 7_776_000,
            time_since_clip_sec: 7_776_000,
            halflife_5m: 300.0,
            halflife_10m: 600.0,
            halflife_1h: 3600.0,
            halflife_24h: 86400.0,
            halflife_1d: 86400.0,
            streak_cap: 16.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RouteCode {
    Pass,
    Refer,
    Reject,
    Unknown,
}

impl RouteCode {
    pub fn as_f32(self) -> f32 {
        match self {
            RouteCode::Pass => -1.0,
            RouteCode::Refer => 0.0,
            RouteCode::Reject => 1.0,
            RouteCode::Unknown => 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MicroStateEvent {
    pub event_ts_sec: u64,
    pub l2_margin: f32,
    pub final_margin: f32,
    pub route_code: RouteCode,
    pub amount: f32,
    pub is_txn: bool,
    pub is_refer: bool,
    pub is_reject: bool,
    pub merchant_hash: u64,
    pub device_hash: u64,
    pub thr_final: f32,
}

fn bool_to_f32(v: bool) -> f32 {
    if v { 1.0 } else { 0.0 }
}

fn shift4(v: &mut [f32], base: usize, new_value: f32) {
    v[base + 3] = v[base + 2];
    v[base + 2] = v[base + 1];
    v[base + 1] = v[base + 0];
    v[base + 0] = new_value;
}

fn ema_update(ema: f32, x: f32, dt: f32, halflife: f32) -> f32 {
    if halflife <= 0.0 {
        return x;
    }
    if dt <= 0.0 {
        return ema;
    }
    let alpha = 1.0 - (-std::f32::consts::LN_2 * dt / halflife).exp();
    (1.0 - alpha) * ema + alpha * x
}

fn time_since_log1p(now: u64, last: u64, clip: u64) -> f32 {
    if last == 0 || now <= last {
        return 0.0;
    }
    let dt = now - last;
    let dt = dt.min(clip) as f32;
    dt.ln_1p()
}

fn update_streak(prev: f32, is_hit: bool, cap: f32) -> f32 {
    if is_hit {
        let next = prev + 1.0;
        if next > cap { cap } else { next }
    } else {
        0.0
    }
}
