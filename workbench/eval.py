from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np
import numpy.typing as npt
import lightgbm as lgb


def calc_ndgc_by_model(
    model: lgb.Booster,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    q: npt.NDArray[np.int64],
    k: int = 10,
):
    score = model.predict(X)
    return calc_ndgc(score, y, q, k)


def calc_ndgc(
    score: npt.NDArray[np.float64],
    label: npt.NDArray[np.float64],
    query: npt.NDArray[np.int64],
    k: int = 10,
):
    pred_df = pd.DataFrame({"score": score, "label": label, "qid": query})
    pred_df["rank"] = pred_df.sort_values(by="score", ascending=False).groupby("qid").cumcount() + 1

    label_df = pred_df.pivot(index="qid", columns="rank", values="label").fillna(0)
    # NOTE: score fill value should be less than min score
    score_fill_value = pred_df["score"].min() - 1
    score_df = pred_df.pivot(index="qid", columns="rank", values="score").fillna(score_fill_value)

    ndcg_value = ndcg_score(label_df, score_df, k=k, ignore_ties=True)
    return ndcg_value
