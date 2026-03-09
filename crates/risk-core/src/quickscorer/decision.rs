use crate::schema::Decision;

#[inline]
pub fn l1_is_pass(score: f32, threshold: f32) -> bool {
    score >= threshold
}

#[inline]
pub fn l2_is_reject(score: f32, threshold: f32) -> bool {
    score >= threshold
}

#[inline]
pub fn final_decision(l1_pass: bool, l2_reject: Option<bool>) -> Decision {
    if l1_pass {
        return Decision::Allow;
    }
    match l2_reject {
        Some(true) => Decision::Deny,
        Some(false) => Decision::ManualReview,
        None => Decision::ManualReview,
    }
}
