import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.utils import compute_class_weight
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss, _Loss
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .models import ClassicGNN
from .utils import DEVICE, GraphDataset, create_pyg_dataset, set_determinism


def train_gnn(
    model: nn.Module,
    dataset_train: GraphDataset,
    dataset_valid: GraphDataset | None,
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    batch_size: int = 64,
    eval_every_n_epochs: int = 10,
    random_state: int = 0,
) -> nn.Module:
    set_determinism(random_state)
    model = model.to(DEVICE)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=(DEVICE == "cuda"),
    )

    optimizer, criterion = _get_optimizer_and_loss(
        model, learning_rate, dataset_train.y
    )

    best_valid_auroc = 0
    best_model = None
    for epoch in range(num_epochs):
        model.train()

        for inputs in train_loader:
            inputs = inputs.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(
                x=inputs.x,
                edge_index=inputs.edge_index,
                edge_attr=inputs.edge_attr,
                batch=inputs.batch,
            )
            loss = criterion(outputs, inputs.y.long())
            loss.backward()
            optimizer.step()

        if (
            dataset_valid is not None
            and epoch > 0
            and (epoch % eval_every_n_epochs == 0 or epoch == num_epochs - 1)
        ):
            model.eval()
            valid_auroc = eval_auroc(model, dataset_valid)

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                best_model = copy.deepcopy(model.state_dict())

    # for the final model, we don't use validation set,
    # but just train for given number of epochs
    if best_model is None:
        best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model


def _get_optimizer_and_loss(
    model: nn.Module, learning_rate: float, y_train: np.ndarray
) -> tuple[Optimizer, _Loss]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.ravel(),
    )
    class_weights = torch.from_numpy(class_weights)
    class_weights = class_weights.float().to(DEVICE)

    criterion = CrossEntropyLoss(weight=class_weights)
    return optimizer, criterion


def eval_auroc(model: nn.Module, dataset: GraphDataset) -> float:
    y_true = dataset.y
    y_pred_proba = predict_proba(model, dataset)[:, 1]
    auroc = 100 * roc_auc_score(y_true, y_pred_proba)
    return auroc


@torch.no_grad()
def predict_proba(model: nn.Module, dataset: GraphDataset) -> np.ndarray:
    model.eval()
    predictions = []
    data_loader = DataLoader(dataset, batch_size=1000)
    for inputs in data_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(
            x=inputs.x,
            edge_index=inputs.edge_index,
            edge_attr=inputs.edge_attr,
            batch=inputs.batch,
        )
        y_pred_proba = torch.softmax(outputs, dim=1).cpu()
        predictions.append(y_pred_proba)

    return torch.cat(predictions).numpy()


def tune_hyperparameters(
    conv_type: str,
    smiles_list: list[str],
    labels: np.ndarray,
) -> dict:
    param_grid = {
        "num_layers": [2, 3],
        "num_channels": [32, 64],
        "dropout": [0.25, 0.5],
        "learning_rate": [1e-3, 1e-2, 1e-1],
    }

    smiles_list = np.array(smiles_list)
    idxs = list(range(len(smiles_list)))
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    best_score = 0
    best_params = None
    for params in tqdm(ParameterGrid(param_grid)):
        model = ClassicGNN(conv_type=conv_type, **params)
        valid_scores = []
        for train_idxs, valid_idxs in kfold.split(idxs, labels):
            smiles_train = smiles_list[train_idxs]
            y_train = labels[train_idxs]
            dataset_train = create_pyg_dataset(smiles_train, y_train)

            smiles_valid = smiles_list[valid_idxs]
            y_valid = labels[valid_idxs]
            dataset_valid = create_pyg_dataset(smiles_valid, y_valid)

            model = train_gnn(
                model,
                dataset_train,
                dataset_valid,
                num_epochs=100,
                learning_rate=params["learning_rate"],
            )
            fold_score = eval_auroc(model, dataset_valid)
            valid_scores.append(fold_score)

        valid_score = np.mean(valid_scores)
        if valid_score > best_score:
            best_score = valid_score
            best_params = params

    return best_params
