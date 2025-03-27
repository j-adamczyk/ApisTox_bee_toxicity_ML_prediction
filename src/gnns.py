from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.gnns.models import ClassicGNN
from src.gnns.trainer import predict_proba, train_gnn, tune_hyperparameters
from src.gnns.utils import create_pyg_dataset
from src.utils import load_split


def train_classic_GNN(
    smiles_train: list[str],
    smiles_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    conv_type: str,
) -> None:
    hyperparams = tune_hyperparameters(conv_type, smiles_train, y_train)

    dataset_train = create_pyg_dataset(smiles_train, y_train)
    dataset_test = create_pyg_dataset(smiles_test, y_test)

    metrics = defaultdict(list)
    for seed in tqdm(range(50)):
        model = ClassicGNN(conv_type=conv_type, **hyperparams)
        model = train_gnn(
            model,
            dataset_train,
            dataset_valid=None,
            num_epochs=100,
            learning_rate=hyperparams["learning_rate"],
            eval_every_n_epochs=10,
            random_state=seed,
        )

        y_pred_proba = predict_proba(model, dataset_test)
        y_pred = y_pred_proba.argmax(axis=1)
        y_pred_proba = y_pred_proba[:, 1]

        metrics["AUROC"].append(roc_auc_score(y_test, y_pred_proba))
        metrics["balanced acc"].append(balanced_accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["MCC"].append(matthews_corrcoef(y_test, y_pred))

    print("AUROC\tbalanced acc\tprecision\trecall\tMCC")
    for metric_name in ["AUROC", "balanced acc", "precision", "recall"]:
        values = metrics[metric_name]
        print(f"{np.mean(values):.2%} +- {np.std(values):.2%}", end="\t")

    values = metrics["MCC"]
    print(f"{np.mean(values):.2f} +- {np.std(values):.2f}")
    print(f"Hyperparameters: {hyperparams}")


if __name__ == "__main__":
    for split_type in ["time", "maxmin"]:
        print("SPLIT TYPE", split_type)
        smiles_train, smiles_test, y_train, y_test = load_split(split_type)
        for conv_type in ["GCN", "GraphSAGE", "GIN", "GAT", "AttentiveFP"]:
            print(conv_type, flush=True)
            train_classic_GNN(smiles_train, smiles_test, y_train, y_test, conv_type)
            print()
