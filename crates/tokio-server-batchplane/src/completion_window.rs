use std::collections::VecDeque;

pub struct CompletionWindow<T> {
    next_seq: u64,
    slots: VecDeque<Option<T>>,
}

impl<T> CompletionWindow<T> {
    pub fn new() -> Self {
        Self {
            next_seq: 0,
            slots: VecDeque::new(),
        }
    }

    pub fn insert(&mut self, seq: u64, value: T) {
        if seq < self.next_seq {
            return;
        }
        let idx = (seq - self.next_seq) as usize;
        while self.slots.len() <= idx {
            self.slots.push_back(None);
        }
        self.slots[idx] = Some(value);
    }

    pub fn pop_ready(&mut self) -> Option<(u64, T)> {
        match self.slots.front() {
            Some(Some(_)) => {}
            _ => return None,
        }
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let value = self
            .slots
            .pop_front()
            .and_then(|slot| slot)
            .expect("completion window front must contain a value");
        Some((seq, value))
    }
}
