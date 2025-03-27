import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_DIR = "../data"
PLOTS_DIR = "../plots"


def load_split(split_type: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    df_train = pd.read_csv(os.path.join(DATA_DIR, f"{split_type}_train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, f"{split_type}_test.csv"))

    smiles_train = df_train["SMILES"].tolist()
    smiles_test = df_test["SMILES"].tolist()

    y_train = df_train["label"].values
    y_test = df_test["label"].values

    return smiles_train, smiles_test, y_train, y_test


def evaluate_sklearn_classifier(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    metrics = defaultdict(list)
    for seed in range(50):
        model.set_params(random_state=seed)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics["MCC"].append(matthews_corrcoef(y_test, y_pred))
        metrics["AUROC"].append(roc_auc_score(y_test, y_pred_proba))
        metrics["precision"].append(precision_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))

    print("MCC\tAUROC\tprecision\trecall")

    for metric_name in ["MCC", "AUROC", "precision", "recall"]:
        values = metrics[metric_name]
        if metric_name == "MCC":
            print(f"{np.mean(values):.2f} +- {np.std(values):.2f}", end="\t")
        else:
            print(f"{np.mean(values):.2%} +- {np.std(values):.2%}", end="\t")

    print()
