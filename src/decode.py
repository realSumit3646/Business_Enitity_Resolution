"""Metric-aware decoding: per S1 entity, link the set of records that maximises EXPECTED F0.5 instead of
thresholding every record at the same t.

The score is macro F0.5 per S1 with a singleton bonus (an empty prediction scores 1 on an entity with no
matches), so the first link to an S1 is riskier than a 3rd link to an S1 that already has confident matches.
Each record first picks its best S1 (as before). For every S1, its records are sorted by probability and the
prefix length k in 0..K is chosen that maximises E[F0.5], estimated by Monte Carlo with independent Bernoulli
outcomes (calibrated stage-2 probabilities). MISS adds the expected number of true matches that no record
proposes (blocking misses). Pure numpy, vectorised per K.
"""
import numpy as np
import polars as pl

KMAX, M, PFLOOR = 12, 400, 0.05


def expected_best_k(P: np.ndarray, miss: float, rng) -> np.ndarray:
    """P: [S, K] probabilities sorted descending (0-padded). Returns the best k per row (0..K)."""
    S, K = P.shape
    Z = rng.random((S, K, M)) < P[:, :, None]                     # sampled truth of each proposed record
    extra = rng.poisson(miss, (S, 1, M)) if miss > 0 else 0       # true matches no record proposes
    n_true = Z.sum(1, keepdims=True) + extra                       # [S,1,M]
    cum = np.cumsum(Z, axis=1)                                     # tp if the first k are linked, k=1..K
    best = np.zeros((S, K + 1))
    best[:, 0] = (n_true[:, 0, :] == 0).mean(1)                    # k=0: 1 only when there is no true match
    for k in range(1, K + 1):
        tp = cum[:, k - 1, :]
        nt = n_true[:, 0, :]
        prec, rec = tp / k, np.where(nt > 0, tp / np.maximum(nt, 1), 0)
        f = np.where(tp > 0, 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12), 0.0)
        best[:, k] = f.mean(1)
    valid = np.concatenate([np.ones((S, 1), bool), P > 0], axis=1)  # can't link padding
    best[~valid] = -1
    return best.argmax(1)


def decode(pred: pl.DataFrame, miss: float = 0.0, pfloor: float = PFLOOR, seed: int = 0) -> pl.DataFrame:
    """pred: ridx, sidx, p (all candidate pairs). Returns the chosen links (sidx, ridx)."""
    rng = np.random.default_rng(seed)
    best = (pred.group_by("ridx").agg(pl.col("sidx").get(pl.col("p").arg_max()), pl.col("p").max())
                .filter(pl.col("p") >= pfloor))
    g = (best.sort("p", descending=True).group_by("sidx")
             .agg(pl.col("ridx").head(KMAX), pl.col("p").head(KMAX)).with_columns(pl.col("p").list.len().alias("K")))
    out = []
    for K, sub in g.group_by("K"):
        K = K[0]
        P = np.array(sub["p"].to_list(), dtype=np.float64).reshape(-1, K)
        for lo in range(0, len(P), 20000):                         # bounded memory: S*K*M booleans per block
            kb = expected_best_k(P[lo:lo + 20000], miss, rng)
            blk = sub.slice(lo, 20000).with_columns(pl.Series("k", kb))
            out.append(blk.select("sidx", pl.col("ridx").list.head(pl.col("k"))).explode("ridx").drop_nulls())
    return pl.concat(out) if out else pl.DataFrame({"sidx": [], "ridx": []})
