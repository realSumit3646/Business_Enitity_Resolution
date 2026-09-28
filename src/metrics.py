"""Macro F0.5 exactly as the challenge defines it (per Source-1 entity, singletons included)."""
import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series, beta: float = 0.5) -> dict:
    """pred / truth: long frames (s1, rid). s1_ids: every S1 entity in the evaluation set.
    Singleton rule: empty truth & empty pred -> 1.0; empty truth & non-empty pred -> 0.0;
    non-empty truth & empty pred -> 0.0."""
    b2 = beta * beta
    base = pl.DataFrame({"s1": s1_ids})
    tp = pred.join(truth, on=["s1", "rid"], how="semi").group_by("s1").len().rename({"len": "tp"})
    np_ = pred.group_by("s1").len().rename({"len": "n_pred"})
    nt = truth.group_by("s1").len().rename({"len": "n_true"})
    d = (base.join(tp, on="s1", how="left").join(np_, on="s1", how="left").join(nt, on="s1", how="left")
             .fill_null(0))
    d = d.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
          .when((pl.col("n_true") == 0) | (pl.col("n_pred") == 0) | (pl.col("tp") == 0)).then(0.0)
          .otherwise(
              (1 + b2) * (pl.col("tp") / pl.col("n_pred")) * (pl.col("tp") / pl.col("n_true"))
              / (b2 * pl.col("tp") / pl.col("n_pred") + pl.col("tp") / pl.col("n_true"))
          ).alias("f"))
    single = d["n_true"] == 0
    return {
        "f05": d["f"].mean(),
        "f05_singletons": d.filter(single)["f"].mean() if single.any() else None,
        "f05_matched": d.filter(~single)["f"].mean(),
        "precision_micro": tp["tp"].sum() / max(pred.height, 1),
        "recall_micro": tp["tp"].sum() / max(truth.height, 1),
        "n_s1": d.height,
    }


if __name__ == "__main__":
    # the worked example from the problem statement: P=2/3, R=1 -> 0.714
    p = pl.DataFrame({"s1": ["S1-1"] * 3, "rid": ["S2-47", "S2-193", "S3-812"]})
    t = pl.DataFrame({"s1": ["S1-1"] * 2, "rid": ["S2-47", "S3-812"]})
    print(macro_f05(p, t, pl.Series(["S1-1"])))
    print(macro_f05(p.head(0), t.head(0), pl.Series(["S1-1"])))  # correct singleton -> 1.0
