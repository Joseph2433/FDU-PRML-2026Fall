import os
import numpy as np
from typing import List, Tuple
from data_generate import load_prepared_dataset
from viz_knn import plot_k_curve, plot_decision_boundary_multi

# 输出目录
OUT_DIR = "./output"
DATA_DIR = "./input_knn"

# ============ TODO 1：pairwise_dist ============
def pairwise_dist(X_test, X_train, metric, mode):
    """
    Compute pairwise distances between X_test (Nte,D) and X_train (Ntr,D).

    Required:
      - L2 distance 'l2' with modes:
          * 'two_loops'  
          * 'no_loops' 
      - 'cosine' distance (distance = 1 - cosine_similarity)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    Nte, D  = X_test.shape
    Ntr, D2 = X_train.shape
    assert D == D2, "Dim mismatch between test and train."

    if metric == "l2":
        if mode == "two_loops":
            # =============== TODO (students, REQUIRED) ===============
            dists = np.empty((Nte, Ntr), dtype=np.float64)
            for i in range(Nte):
                for j in range(Ntr):
                    diff = X_test[i] - X_train[j]
                    dists[i, j] = np.sqrt(np.sum(diff * diff))
            return dists
            # =========================================================

        elif mode == "no_loops":
            # =============== TODO (students, REQUIRED) ===============
            test_sq = np.sum(X_test * X_test, axis=1, keepdims=True)
            train_sq = np.sum(X_train * X_train, axis=1)[None, :]
            squared_dists = test_sq + train_sq - 2.0 * (X_test @ X_train.T)
            # Roundoff can make a theoretically nonnegative value negative.
            return np.sqrt(np.maximum(squared_dists, 0.0))
            # =========================================================

        else:
            raise ValueError("Unknown mode for L2.")

    elif metric == "cosine":
        # =============== TODO (students, REQUIRED) ===============
        test_norm = np.linalg.norm(X_test, axis=1, keepdims=True)
        train_norm = np.linalg.norm(X_train, axis=1)[None, :]
        denominators = test_norm * train_norm
        # A zero vector has no direction; use similarity 0 (distance 1).
        similarities = np.divide(
            X_test @ X_train.T, denominators,
            out=np.zeros((Nte, Ntr), dtype=np.float64),
            where=denominators > 0.0,
        )
        return 1.0 - np.clip(similarities, -1.0, 1.0)
        # ================================================
    else:
        raise ValueError("metric must be 'l2' or 'cosine'.")


# ============ TODO 2：knn_predict（多数表决） ============
def knn_predict(X_test, X_train, y_train, k, metric, mode):
    """
    kNN prediction.
    Required: majority vote with L2 distance.

    Returns
    -------
    y_pred : (Nte,) int
    """
    dists = pairwise_dist(X_test, X_train, metric=metric, mode=mode)
    y_train = np.asarray(y_train).reshape(-1).astype(int)
    Nte = dists.shape[0]
    y_pred = np.zeros(Nte, dtype=int)

    for i in range(Nte):
        idx = np.argsort(dists[i])[:k]
        neighbors = y_train[idx]

        # =============== TODO (students, REQUIRED) ===============
        labels, counts = np.unique(neighbors, return_counts=True)
        # np.unique sorts labels, so the first maximum is the smallest tie.
        y_pred[i] = labels[np.argmax(counts)]
        # ===========================================

    return y_pred


# ============ TODO 3：select_k_by_validation ============
def select_k_by_validation(X_train, y_train, X_val, y_val, ks: List[int], metric, mode) -> Tuple[int, List[float]]:
    """
    Grid-search K on validation set.

    Returns
    -------
    best_k : int
    accs   : list of validation accuracies aligned with ks
    """
    # =============== TODO (students, REQUIRED) ===============
    if len(ks) == 0:
        raise ValueError("ks must contain at least one candidate.")
    y_val = np.asarray(y_val).reshape(-1)
    accs = []
    for k in ks:
        y_pred = knn_predict(X_val, X_train, y_train, k, metric, mode)
        accs.append(float(np.mean(y_pred == y_val)))
    # Keep accs in input order, but resolve accuracy ties by smallest k.
    best_index = max(range(len(ks)), key=lambda i: (accs[i], -ks[i]))
    return int(ks[best_index]), accs
    # =========================================================


def run_with_visualization(metric="l2", mode="no_loops"):
    X_train, y_train, X_val, y_val, X_test, y_test = load_prepared_dataset(DATA_DIR)

    ks = [1, 3, 5, 7, 9, 11, 13]

    best_k, accs = select_k_by_validation(X_train, y_train, X_val, y_val,
                                          ks, metric=metric, mode=mode)
    print(f"[ModelSelect] metric={metric} best k={best_k} (val acc={max(accs):.4f})")
    plot_k_curve(ks, accs, os.path.join(OUT_DIR, f"knn_k_curve_{metric}.png"))

    X_trv = np.vstack([X_train, X_val]); y_trv = np.hstack([y_train, y_val])
    def predict_fn_for_k(k):
        return lambda Xq: knn_predict(Xq, X_trv, y_trv, k, metric=metric, mode=mode)

    ks_panel = sorted(set(ks + [best_k]))
    plot_decision_boundary_multi(predict_fn_for_k, X_train, y_train, X_test, y_test,
                                 ks=ks_panel,
                                 out_path=os.path.join(OUT_DIR, f"knn_boundary_grid_{metric}.png"),
                                 grid_n=200, batch_size=4096)


if __name__ == "__main__":
    # metric: ["l2", "cosine"];  mode: ["two_loops", "no_loops"]
    for metric in ["l2", "cosine"]:
        run_with_visualization(metric=metric, mode="no_loops")
