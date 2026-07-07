"""
FICO Score Bucketing (Quantization) for Categorical Rating Map
================================================================
Charlie's model needs FICO scores (integers, ~300-850) converted into a
small number of categorical "rating" buckets, so the model can use them as
categorical input labels. This module finds the *optimal* bucket boundaries
-- rather than naive equal-width or equal-frequency bins -- using dynamic
programming (DP), and exposes a `RatingMap` that converts any raw FICO
score into its bucket rating.

Convention: rating 1 = best credit quality (highest FICO scores), and
higher rating numbers = worse credit quality (lower FICO scores) -- as
specified in the task ("a lower rating signifies a better credit score").

Two optimization objectives are implemented, both solved exactly via DP:

1. Mean Squared Error (MSE) quantization
   ------------------------------------
   Treats this as a 1-D approximation problem: every FICO score in a bucket
   is represented by a single value (the bucket mean). We choose boundaries
   to minimize the total squared error between each score and its bucket's
   mean. This only looks at the distribution of scores themselves -- it
   ignores whether the borrower defaulted.

2. Log-Likelihood (LL) quantization
   ---------------------------------
   Models each bucket i as its own Bernoulli/binomial default-rate parameter
   p_i = k_i / n_i (k_i defaults out of n_i borrowers in the bucket), and
   picks boundaries that MAXIMIZE the total log-likelihood of observing the
   actual default outcomes under this piecewise-constant PD model:

       LL = sum_i [ k_i * log(p_i) + (n_i - k_i) * log(1 - p_i) ]

   This is the objective quoted in the task, and directly optimizes buckets
   for their usefulness in predicting default -- two buckets get separated
   only if doing so meaningfully improves how well we can describe the
   observed default pattern.

Both problems are solved with the same DP structure: precompute the cost/
log-likelihood of merging any contiguous range of *unique* FICO scores into
a single bucket (using prefix sums so this is O(1) per range), then run a
standard "partition into B contiguous segments to optimize sum of segment
costs" DP, which is O(B * V^2) where V = number of unique score values
(here V=374, so this runs in well under a second).
"""

import math
import numpy as np
import pandas as pd

CSV_PATH = "Task 3 and 4_Loan_Data.csv"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_fico_data(csv_path: str = CSV_PATH):
    """
    Returns (scores_sorted, n_per_score, k_per_score):
      scores_sorted -- sorted array of unique FICO scores
      n_per_score   -- number of borrowers at each unique score
      k_per_score   -- number of defaults at each unique score
    Aggregating to unique scores keeps the DP fast regardless of how many
    total loan records (rows) there are.
    """
    df = pd.read_csv(csv_path)
    grouped = df.groupby("fico_score")["default"].agg(["count", "sum"]).reset_index()
    grouped = grouped.sort_values("fico_score")
    scores = grouped["fico_score"].to_numpy(dtype=float)
    n = grouped["count"].to_numpy(dtype=float)
    k = grouped["sum"].to_numpy(dtype=float)
    return scores, n, k


# ---------------------------------------------------------------------------
# Generic DP: partition V ordered points into B contiguous buckets to
# optimize the sum of a pluggable per-bucket cost/score function.
# ---------------------------------------------------------------------------

def _optimal_partition(V: int, n_buckets: int, segment_value, maximize: bool):
    """
    segment_value(i, j) -> value of making bucket == points[i:j] (0-indexed,
    j exclusive). Points are implicit (we only need count V of them).

    Returns list of boundary indices [0, b_1, b_2, ..., V] (length n_buckets+1)
    describing bucket k as points[boundaries[k]:boundaries[k+1]].
    """
    NEG = -math.inf if maximize else math.inf
    better = (lambda a, b: a > b) if maximize else (lambda a, b: a < b)

    # dp[b][j] = best total value of splitting points[0:j] into b buckets
    dp = [[NEG] * (V + 1) for _ in range(n_buckets + 1)]
    back = [[-1] * (V + 1) for _ in range(n_buckets + 1)]
    dp[0][0] = 0.0

    for b in range(1, n_buckets + 1):
        for j in range(b, V + 1):
            best_val = NEG
            best_i = -1
            # bucket b covers points[i:j]; need at least 1 point per bucket
            for i in range(b - 1, j):
                if dp[b - 1][i] == NEG:
                    continue
                val = dp[b - 1][i] + segment_value(i, j)
                if better(val, best_val):
                    best_val = val
                    best_i = i
            dp[b][j] = best_val
            back[b][j] = best_i

    # Reconstruct boundaries
    boundaries = [V]
    b, j = n_buckets, V
    while b > 0:
        i = back[b][j]
        boundaries.append(i)
        j = i
        b -= 1
    boundaries.reverse()
    return boundaries, dp[n_buckets][V]


# ---------------------------------------------------------------------------
# 1. MSE-optimal quantization
# ---------------------------------------------------------------------------

def mse_quantization(scores: np.ndarray, n: np.ndarray, n_buckets: int):
    """
    Finds bucket boundaries minimizing total squared error when every score
    in a bucket is represented by the bucket's (count-weighted) mean.
    Returns (boundaries_as_fico_values, total_sse).
    """
    V = len(scores)
    # Weighted prefix sums for O(1) segment cost.
    cum_n = np.concatenate([[0.0], np.cumsum(n)])
    cum_nx = np.concatenate([[0.0], np.cumsum(n * scores)])
    cum_nx2 = np.concatenate([[0.0], np.cumsum(n * scores ** 2)])

    def segment_value(i, j):
        seg_n = cum_n[j] - cum_n[i]
        if seg_n <= 0:
            return 0.0
        seg_sum = cum_nx[j] - cum_nx[i]
        seg_sumsq = cum_nx2[j] - cum_nx2[i]
        mean = seg_sum / seg_n
        # SSE = sum(n_v * (x_v - mean)^2) = sum(n_v x_v^2) - 2*mean*sum(n_v x_v) + mean^2 * sum(n_v)
        sse = seg_sumsq - 2 * mean * seg_sum + mean ** 2 * seg_n
        return -sse  # negate so we can "maximize" (i.e., minimize SSE)

    idx_boundaries, best_val = _optimal_partition(V, n_buckets, segment_value, maximize=True)
    fico_boundaries = [scores[0]] + [scores[i] for i in idx_boundaries[1:-1]] + [scores[-1]]
    return fico_boundaries, -best_val


# ---------------------------------------------------------------------------
# 2. Log-likelihood-optimal quantization
# ---------------------------------------------------------------------------

def _bucket_log_likelihood(n_seg: float, k_seg: float) -> float:
    """LL contribution of one bucket with n_seg borrowers, k_seg defaults."""
    if n_seg <= 0:
        return 0.0
    p = k_seg / n_seg
    ll = 0.0
    if k_seg > 0:
        ll += k_seg * math.log(p)
    if n_seg - k_seg > 0:
        ll += (n_seg - k_seg) * math.log(1 - p)
    return ll


def log_likelihood_quantization(scores: np.ndarray, n: np.ndarray, k: np.ndarray, n_buckets: int):
    """
    Finds bucket boundaries maximizing the total log-likelihood of the
    observed default outcomes under a piecewise-constant PD model (one PD
    per bucket). Returns (boundaries_as_fico_values, total_log_likelihood).
    """
    V = len(scores)
    cum_n = np.concatenate([[0.0], np.cumsum(n)])
    cum_k = np.concatenate([[0.0], np.cumsum(k)])

    def segment_value(i, j):
        seg_n = cum_n[j] - cum_n[i]
        seg_k = cum_k[j] - cum_k[i]
        return _bucket_log_likelihood(seg_n, seg_k)

    idx_boundaries, best_val = _optimal_partition(V, n_buckets, segment_value, maximize=True)
    fico_boundaries = [scores[0]] + [scores[i] for i in idx_boundaries[1:-1]] + [scores[-1]]
    return fico_boundaries, best_val


# ---------------------------------------------------------------------------
# Rating map: convert boundaries into a usable scoring function
# ---------------------------------------------------------------------------

class RatingMap:
    """
    Maps a raw FICO score to an integer rating bucket, where rating 1 =
    best credit quality (highest FICO) and rating n_buckets = worst.
    """

    def __init__(self, boundaries):
        # boundaries: sorted list of n_buckets+1 values, e.g.
        # [408, 580, 620, 660, 700, 850] for 5 buckets.
        self.boundaries = sorted(boundaries)
        self.n_buckets = len(self.boundaries) - 1

    def rate(self, fico_score: float) -> int:
        b = self.boundaries
        # Find which bucket [b[i], b[i+1]] the score falls into
        # (inclusive of the upper edge on the last bucket).
        bucket_idx = None
        for i in range(self.n_buckets):
            lower, upper = b[i], b[i + 1]
            is_last = (i == self.n_buckets - 1)
            if (lower <= fico_score < upper) or (is_last and fico_score <= upper):
                bucket_idx = i
                break
        if bucket_idx is None:
            # Score outside observed range -> clip to nearest edge bucket.
            bucket_idx = 0 if fico_score < b[0] else self.n_buckets - 1

        # bucket_idx=0 is the LOWEST-score bucket -> worst credit -> highest
        # rating number. bucket_idx = n_buckets-1 (highest scores) -> rating 1.
        rating = self.n_buckets - bucket_idx
        return rating

    def describe(self) -> str:
        lines = []
        for i in range(self.n_buckets):
            rating = self.n_buckets - i
            lines.append(f"  Rating {rating}: FICO [{self.boundaries[i]:.0f}, {self.boundaries[i+1]:.0f}]"
                          f"{' (best)' if rating == 1 else ''}{' (worst)' if rating == self.n_buckets else ''}")
        return "\n".join(lines)


def bucket_default_stats(scores, n, k, boundaries):
    """Compute (count, defaults, default_rate) for each bucket given boundaries."""
    boundaries = sorted(boundaries)
    stats = []
    for i in range(len(boundaries) - 1):
        lower, upper = boundaries[i], boundaries[i + 1]
        is_last = (i == len(boundaries) - 2)
        if is_last:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        seg_n = n[mask].sum()
        seg_k = k[mask].sum()
        rate = seg_k / seg_n if seg_n > 0 else float("nan")
        stats.append((seg_n, seg_k, rate))
    return stats


if __name__ == "__main__":
    scores, n, k = load_fico_data()
    print(f"Loaded {len(scores)} unique FICO scores from {int(n.sum())} borrower records.")
    print(f"Overall default rate: {k.sum() / n.sum():.2%}\n")

    N_BUCKETS = 5

    # --- MSE-based quantization ---
    mse_boundaries, sse = mse_quantization(scores, n, N_BUCKETS)
    print(f"=== MSE-optimal quantization ({N_BUCKETS} buckets) ===")
    print(f"Boundaries: {[round(b) for b in mse_boundaries]}")
    print(f"Total SSE: {sse:,.0f}")
    mse_stats = bucket_default_stats(scores, n, k, mse_boundaries)
    rating_map_mse = RatingMap(mse_boundaries)
    print(rating_map_mse.describe())
    for i, (seg_n, seg_k, rate) in enumerate(mse_stats):
        rating = N_BUCKETS - i
        print(f"  Rating {rating}: n={int(seg_n)}, defaults={int(seg_k)}, default_rate={rate:.2%}")
    print()

    # --- Log-likelihood-based quantization ---
    ll_boundaries, ll = log_likelihood_quantization(scores, n, k, N_BUCKETS)
    print(f"=== Log-likelihood-optimal quantization ({N_BUCKETS} buckets) ===")
    print(f"Boundaries: {[round(b) for b in ll_boundaries]}")
    print(f"Total log-likelihood: {ll:,.2f}")
    ll_stats = bucket_default_stats(scores, n, k, ll_boundaries)
    rating_map_ll = RatingMap(ll_boundaries)
    print(rating_map_ll.describe())
    for i, (seg_n, seg_k, rate) in enumerate(ll_stats):
        rating = N_BUCKETS - i
        print(f"  Rating {rating}: n={int(seg_n)}, defaults={int(seg_k)}, default_rate={rate:.2%}")
    print()

    # --- Example usage of the final rating map ---
    print("=== Example: rating individual borrowers (using log-likelihood map) ===")
    for test_score in [410, 580, 620, 660, 700, 780, 850]:
        print(f"  FICO {test_score} -> rating {rating_map_ll.rate(test_score)}")
