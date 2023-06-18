import pathlib
from collections import Counter

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.datasets import load_svmlight_file
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit


def sparsity(X: npt.NDArray[np.float64]) -> float:
    """Check sparsity of a matrix: X

    Args:
        X: matrix to check

    Returns:
        sparsity of X
    """
    number_of_nan = np.count_nonzero(np.isnan(X))
    number_of_zeros = np.count_nonzero(np.abs(X) < 1e-6)
    return (number_of_nan + number_of_zeros) / float(X.shape[0] * X.shape[1]) * 100.0


def print_dataset_statistics(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    queries: npt.NDArray[np.int64],
    name,
):
    """Print dataset statistics

    Args:
        X: features
        y: labels
        queries: query ids
        name: dataset name
    """
    print("----------------------------------")
    print("Characteristics of dataset " + name)
    print("rows x columns " + str(X.shape))
    print("sparsity: " + str(sparsity(X)))
    print("y distribution")
    print(Counter(y))
    print("num samples in queries: minimum, median, maximum")
    num_queries = list(Counter(queries).values())
    print(np.min(num_queries), np.median(num_queries), np.max(num_queries))
    print("----------------------------------")


def process_libsvm_file(
    path: pathlib.Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Load libsvm file and convert it to numpy

    Args:
        path: libsvm file path

    Returns:
        data, labels, query ids
    """
    # NOTE: The detail of svmlight format is the following:
    # https://blog.argcv.com/articles/5371.c#:~:text=SVMlight%20is%20an%20implementation,a%20lot%20of%20other%20programs.
    X, y, queries = load_svmlight_file(str(path), query_id=True)
    return X.todense(), y, queries


def dump_to_file(
    path: pathlib.Path,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    queries: npt.NDArray[np.int64],
):
    """Dump data to file

    Args:
        path: destination path
        X: features
        y: labels
        queries: query ids
    """
    columns = ["label", "qid"] + [f"f{i}" for i in range(1, X.shape[1] + 1)]
    dtype_dict = {c: int if c == "qid" else float for c in columns}
    all_data = np.hstack((y.reshape(-1, 1), queries.reshape(-1, 1), X))
    df = pd.DataFrame(all_data, columns=columns).astype(dtype_dict).sort_values(by="qid")
    df.to_csv(path, sep="\t", index=False)


def read_dataset(
    path: pathlib.Path,
) -> tuple[pd.DataFrame, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    assert path.exists(), f"{path} does not exist"
    df = pd.read_csv(path, sep="\t")

    y = df["label"].values.astype(np.float64)
    queries = df["qid"].values.astype(np.int64)
    X = df.drop(columns=["label", "qid"]).values.astype(np.float64)

    return df, X, y, queries


def prepare_lightgbm_dataset(
    dst_path: pathlib.Path,
) -> tuple[
    tuple[
        lgb.Dataset,
        pd.DataFrame,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
    ],
    tuple[
        lgb.Dataset,
        pd.DataFrame,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
    ],
    tuple[
        lgb.Dataset,
        pd.DataFrame,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
    ],
]:
    df, X, y, q = read_dataset(dst_path / "train.tsv")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1026)
    train_idx, val_idx = next(gss.split(X, y, groups=q))

    df_train, X_train, y_train, q_train = (
        df.iloc[train_idx],
        X[train_idx],
        y[train_idx],
        q[train_idx],
    )
    df_val, X_val, y_val, q_val = df.iloc[val_idx], X[val_idx], y[val_idx], q[val_idx]

    group = list(Counter(q_train).values())
    lgb_dataset_train = lgb.Dataset(X_train, y_train, group=group)

    group = list(Counter(q_val).values())
    lgb_dataset_val = lgb.Dataset(X_val, y_val, group=group)

    df_test, X_test, y_test, q_test = read_dataset(dst_path / "test.tsv")
    group = list(Counter(q_test).values())
    lgb_dataset_test = lgb.Dataset(X_test, y_test, group=group)

    return (
        (lgb_dataset_train, df_train, X_train, y_train, q_train),
        (lgb_dataset_val, df_val, X_val, y_val, q_val),
        (lgb_dataset_test, df_test, X_test, y_test, q_test),
    )


def mq2008(src_path: pathlib.Path, dst_path: pathlib.Path):
    """
    0 - label, 1 - qid, ...features...
    ----------------------------------
    Characteristics of dataset mq2008 train
    rows x columns (9630, 46)
    sparsity: 47.2267370987
    y distribution
    Counter({0.0: 7820, 1.0: 1223, 2.0: 587})
    num samples in queries: minimum, median, maximum
    (5, '8.0', 121)
    ----------------------------------
    ----------------------------------
    Characteristics of dataset mq2008 test
    rows x columns (2874, 46)
    sparsity: 46.1128256331
    y distribution
    Counter({0.0: 2319, 1.0: 378, 2.0: 177})
    num samples in queries: minimum, median, maximum
    (6, '14.5', 119)
    ----------------------------------
    """
    train_file = src_path / "train.txt"
    test_file = src_path / "test.txt"

    train_out_file = dst_path / "train.tsv"
    test_out_file = dst_path / "test.tsv"

    X, y, queries = process_libsvm_file(train_file)
    print_dataset_statistics(X, y, queries, "mq2008 train")
    dump_to_file(train_out_file, X, y, queries)

    X, y, queries = process_libsvm_file(test_file)
    print_dataset_statistics(X, y, queries, "mq2008 test")
    dump_to_file(test_out_file, X, y, queries)
