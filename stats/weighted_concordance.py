import numpy as np


def _make_assertions(event_times, predicted_scores, event_observed, weights=None):
    event_times = np.asarray(event_times)
    predicted_scores = np.asarray(predicted_scores)
    event_observed = np.asarray(event_observed, dtype=bool)

    if weights is None:
        weights = np.ones(len(event_times))
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if len(weights) != len(event_times):
            raise ValueError("weights must have same length as event_times")
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("weights must be between 0 and 1")

    n = len(event_times)
    if n < 2:
        return np.nan

    return event_times, predicted_scores, event_observed, weights, n


def weighted_concordance_index(event_times, predicted_scores, event_observed, weights=None):
    event_times, predicted_scores, event_observed, weights, n = _make_assertions(event_times,
                                                                                 predicted_scores,
                                                                                 event_observed,
                                                                                 weights)

    # Create matrices for vectorized comparison
    times_i, times_j = np.meshgrid(event_times, event_times, indexing='ij')
    observed_i, observed_j = np.meshgrid(event_observed, event_observed, indexing='ij')
    scores_i, scores_j = np.meshgrid(predicted_scores, predicted_scores, indexing='ij')
    weights_i, weights_j = np.meshgrid(weights, weights, indexing='ij')

    # Only consider upper triangle (i < j)
    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Valid pair conditions
    both_observed = observed_i & observed_j
    i_obs_j_cens = observed_i & ~observed_j & (times_i <= times_j)
    i_cens_j_obs = ~observed_i & observed_j & (times_i >= times_j)

    valid_pairs = upper_triangle & (both_observed | i_obs_j_cens | i_cens_j_obs)

    if not np.any(valid_pairs):
        return np.nan

    # Calculate pair weights
    pair_weights = weights_i * weights_j
    valid_pair_weights = pair_weights[valid_pairs]
    total_weighted_pairs = np.sum(valid_pair_weights)

    # Get valid comparisons
    valid_times_i = times_i[valid_pairs]
    valid_times_j = times_j[valid_pairs]
    valid_scores_i = scores_i[valid_pairs]
    valid_scores_j = scores_j[valid_pairs]

    # Determine expected ordering and check concordance
    i_earlier = valid_times_i < valid_times_j
    j_earlier = valid_times_i > valid_times_j
    same_time = valid_times_i == valid_times_j

    # For i_earlier: i should have higher score (higher risk)
    # For j_earlier: j should have higher score (higher risk)
    concordant_i_earlier = i_earlier & (valid_scores_i > valid_scores_j)
    concordant_j_earlier = j_earlier & (valid_scores_i < valid_scores_j)
    concordant = concordant_i_earlier | concordant_j_earlier

    # Tied predictions
    tied_predictions = (valid_scores_i == valid_scores_j)
    tied = same_time | tied_predictions

    # Calculate weighted sums
    concordant_weighted_pairs = np.sum(valid_pair_weights[concordant])
    tied_weighted_pairs = np.sum(valid_pair_weights[tied])

    # Calculate weighted concordance index
    weighted_concordance = (concordant_weighted_pairs + 0.5 * tied_weighted_pairs) / total_weighted_pairs

    return weighted_concordance
