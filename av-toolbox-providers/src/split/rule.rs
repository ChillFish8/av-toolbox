use crate::scene::SceneScore;

/// Cuts one scene into pieces no longer than `limit` frames, returning each piece's length in order.
///
/// `scores[k]` scores a cut just before frame `k` of the scene. A scene nobody scored is cut
/// evenly. Otherwise each cut favours a strong score near an even split, following Av1an's scene
/// splitting.
pub(crate) fn split_lengths(scores: &[Option<SceneScore>], limit: usize) -> Vec<usize> {
    assert!(limit > 0, "a split limit is at least one frame");

    let length = scores.len();
    if length == 0 {
        return Vec::new();
    }

    if length <= limit {
        return vec![length];
    }

    let scored = scores.iter().any(Option::is_some);
    if !scored {
        return even_lengths(length, limit);
    }

    scored_lengths(scores, limit)
}

/// Cuts `length` frames into `length / limit + 1` pieces as even as whole frames allow.
fn even_lengths(length: usize, limit: usize) -> Vec<usize> {
    let count = length / limit + 1;

    let mut lengths = Vec::with_capacity(count);
    let mut previous_end = 0;
    for piece in 1..count {
        let end = length * piece / count;
        if end > previous_end {
            lengths.push(end - previous_end);
            previous_end = end;
        }
    }

    lengths.push(length - previous_end);

    lengths
}

fn scored_lengths(scores: &[Option<SceneScore>], limit: usize) -> Vec<usize> {
    let mut lengths = Vec::new();
    let mut start = 0;
    let mut remaining = scores.len();
    while remaining > limit {
        let piece = next_piece(scores, start, remaining, limit);
        lengths.push(piece);
        start += piece;
        remaining -= piece;
    }

    lengths.push(remaining);

    lengths
}

/// The length of the piece cut from the front of the `remaining` frames starting at `start`.
///
/// It searches around the length an even split would give, valuing each scored cut by its score
/// weighted down the further it sits from that length.
fn next_piece(scores: &[Option<SceneScore>], start: usize, remaining: usize, limit: usize) -> usize {
    let needed = remaining / limit;
    let ideal = remaining / (needed + 1);
    let half = ideal / 2;
    let low = half.max(1);
    let high = limit.min(ideal + half);
    let span = high.saturating_sub(low);

    let mut best: Option<(i64, usize)> = None;
    for length in low..=high {
        let Some(score) = scores[start + length] else {
            continue;
        };

        let value = candidate_value(score, ideal, length, span);
        let beats_best = best.is_none_or(|(best_value, _)| value >= best_value);
        if beats_best {
            best = Some((value, length));
        }
    }

    match best {
        Some((_, length)) => length,
        None => ideal.max(1),
    }
}

/// A candidate's weighted score, rounded to four decimal places so near-equal values tie.
fn candidate_value(score: SceneScore, ideal: usize, length: usize, span: usize) -> i64 {
    let distance = ideal.abs_diff(length) as f64;
    let weight = if span == 0 {
        1.0
    } else {
        1.0 - distance / span as f64
    };
    let raw = score.get();
    let value = f64::from(raw) * weight;
    let scaled = value * 10_000.0;

    scaled.round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unscored(length: usize) -> Vec<Option<SceneScore>> {
        vec![None; length]
    }

    fn flat(length: usize, value: f32) -> Vec<Option<SceneScore>> {
        let score = SceneScore::new(value);

        vec![Some(score); length]
    }

    fn only(length: usize, scored: &[(usize, f32)]) -> Vec<Option<SceneScore>> {
        let mut scores = unscored(length);
        for &(position, value) in scored {
            scores[position] = Some(SceneScore::new(value));
        }

        scores
    }

    fn with_peaks(length: usize, base: f32, peaks: &[(usize, f32)]) -> Vec<Option<SceneScore>> {
        let mut scores = flat(length, base);
        for &(position, value) in peaks {
            scores[position] = Some(SceneScore::new(value));
        }

        scores
    }

    fn varied(length: usize) -> Vec<Option<SceneScore>> {
        (0..length)
            .map(|position| {
                if position % 5 == 0 {
                    return None;
                }

                let value = (position * 7919 % 97) as f32 / 10.0;
                Some(SceneScore::new(value))
            })
            .collect()
    }

    #[test]
    fn an_empty_scene_has_no_pieces() {
        let lengths = split_lengths(&[], 10);

        assert!(lengths.is_empty());
    }

    #[test]
    fn a_scene_within_the_limit_is_one_piece() {
        let scores = flat(10, 1.0);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![10]);
    }

    #[test]
    fn an_unscored_scene_splits_evenly() {
        let scores = unscored(25);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![8, 8, 9]);
    }

    #[test]
    fn an_unscored_scene_twice_the_limit_splits_in_three() {
        let scores = unscored(20);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![6, 7, 7]);
    }

    #[test]
    fn an_unscored_scene_with_a_limit_of_one_frame_has_no_empty_pieces() {
        let scores = unscored(3);

        let lengths = split_lengths(&scores, 1);

        assert_eq!(lengths, vec![1, 1, 1]);
    }

    #[test]
    fn equal_scores_split_at_the_even_length() {
        let scores = flat(25, 1.0);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![8, 8, 9]);
    }

    #[test]
    fn a_long_flat_scene_splits_into_equal_pieces() {
        let scores = flat(100, 1.0);

        let lengths = split_lengths(&scores, 30);

        assert_eq!(lengths, vec![25, 25, 25, 25]);
    }

    #[test]
    fn a_strong_score_off_the_even_length_beats_a_weaker_one_on_it() {
        let scores = with_peaks(25, 1.0, &[(10, 3.0)]);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![10, 7, 8]);
    }

    #[test]
    fn a_strong_score_outside_the_search_range_is_ignored() {
        let scores = with_peaks(25, 1.0, &[(3, 50.0), (11, 50.0)]);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![8, 8, 9]);
    }

    #[test]
    fn no_score_in_the_search_range_cuts_at_the_even_length() {
        let scores = only(25, &[(0, 5.0), (24, 5.0)]);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![8, 8, 9]);
    }

    #[test]
    fn ties_go_to_the_longer_piece() {
        let scores = only(25, &[(6, 3.0), (10, 3.0)]);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![10, 7, 8]);
    }

    #[test]
    fn values_equal_to_four_decimal_places_tie() {
        let scores = only(25, &[(6, 3.00001), (10, 3.0)]);

        let lengths = split_lengths(&scores, 10);

        assert_eq!(lengths, vec![10, 7, 8]);
    }

    #[test]
    fn pieces_always_cover_the_scene_within_the_limit() {
        for length in 1..=120 {
            for limit in 1..=30 {
                let patterns = [unscored(length), flat(length, 1.0), varied(length)];
                for scores in patterns {
                    let lengths = split_lengths(&scores, limit);
                    let total: usize = lengths.iter().sum();

                    assert_eq!(total, length, "length {length}, limit {limit}");
                    assert!(
                        lengths.iter().all(|&piece| piece >= 1 && piece <= limit),
                        "length {length}, limit {limit}, pieces {lengths:?}",
                    );
                }
            }
        }
    }
}
