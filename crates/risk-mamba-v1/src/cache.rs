use crate::error::MicroStateError;
use crate::state::{MicroState, MicroStateAux, MicroStateEvent, MicroStateParams};
use std::collections::HashMap;
use std::sync::Mutex;
use xxhash_rust::xxh3::xxh3_64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityKey(u64);

impl EntityKey {
    pub fn from_ids(card_id: Option<u64>, uid: Option<u64>) -> Result<Self, MicroStateError> {
        if let Some(card) = card_id {
            return Ok(EntityKey(hash_id(b'c', card)));
        }
        if let Some(uid) = uid {
            return Ok(EntityKey(hash_id(b'u', uid)));
        }
        Err(MicroStateError::MissingKey)
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

fn hash_id(tag: u8, id: u64) -> u64 {
    let mut buf = [0u8; 9];
    buf[0] = tag;
    buf[1..].copy_from_slice(&id.to_le_bytes());
    xxh3_64(&buf)
}

#[derive(Debug, Clone, Copy)]
struct StateEntry {
    state: MicroState,
    aux: MicroStateAux,
}

impl Default for StateEntry {
    fn default() -> Self {
        Self {
            state: MicroState::default(),
            aux: MicroStateAux::default(),
        }
    }
}

pub struct MicroStateCache {
    shards: Vec<Mutex<HashMap<EntityKey, StateEntry>>>,
    params: MicroStateParams,
}

impl MicroStateCache {
    pub fn new(shards: usize, params: MicroStateParams) -> Self {
        let shard_count = if shards == 0 { 1 } else { shards };
        let mut vec = Vec::with_capacity(shard_count);
        for _ in 0..shard_count {
            vec.push(Mutex::new(HashMap::new()));
        }
        Self {
            shards: vec,
            params,
        }
    }

    pub fn params(&self) -> &MicroStateParams {
        &self.params
    }

    pub fn get(&self, key: EntityKey) -> Option<MicroState> {
        let idx = self.shard_index(key);
        let guard = self.shards[idx].lock().ok()?;
        guard.get(&key).map(|entry| entry.state)
    }

    pub fn update(&self, key: EntityKey, event: &MicroStateEvent) -> Result<MicroState, MicroStateError> {
        let idx = self.shard_index(key);
        let mut guard = self.shards[idx].lock().map_err(|_| MicroStateError::Io("lock poisoned".to_string()))?;
        let entry = guard.entry(key).or_insert_with(StateEntry::default);
        entry.state.apply_event(&mut entry.aux, &self.params, event)?;
        Ok(entry.state)
    }

    fn shard_index(&self, key: EntityKey) -> usize {
        if self.shards.len() == 1 {
            return 0;
        }
        (key.value() as usize) % self.shards.len()
    }
}
