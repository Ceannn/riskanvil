use crate::error::MicroStateError;
use crate::state::{MicroState, MICRO_STATE_DIM, STATIC_BASE_DIM, STATIC_TOTAL_DIM};

pub const INPUT_IDS_LEN: usize = 128;

pub struct ModelInputV1_1<'a> {
    pub input_ids: &'a [i64],
    pub static_base: &'a [f32],
    pub micro_state: &'a MicroState,
}

pub fn concat_static(
    static_base: &[f32],
    micro_state: &MicroState,
    out: &mut [f32],
) -> Result<(), MicroStateError> {
    if static_base.len() != STATIC_BASE_DIM {
        return Err(MicroStateError::BadLen("static_base"));
    }
    if out.len() != STATIC_TOTAL_DIM {
        return Err(MicroStateError::BadLen("static_out"));
    }
    out[..STATIC_BASE_DIM].copy_from_slice(static_base);
    out[STATIC_BASE_DIM..STATIC_BASE_DIM + MICRO_STATE_DIM]
        .copy_from_slice(&micro_state.v);
    Ok(())
}
