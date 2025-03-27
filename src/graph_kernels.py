import joblib
import networkx as nx
import numpy as np
from grakel import (
    Propagation,
    ShortestPath,
    WeisfeilerLehman,
    WeisfeilerLehmanOptimalAssignment,
    graph_from_networkx,
)
from skfp.bases import BasePreprocessor
from skfp.utils import ensure_mols
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.utils import load_split


class SmilesToGrakelGraphTransformer(BasePreprocessor):
    def _transform_batch(self, X):
        # other types are marked as 0
        bond_types = {
            "SINGLE": 1,
            "AROMATIC": 1,
            "DOUBLE": 2,
            "TRIPLE": 3,
        }

        mols = ensure_mols(X)
        graphs = []
        for mol in mols:
            graph = nx.Graph()

            for atom in mol.GetAtoms():
                graph.add_node(atom.GetIdx(), atom_type=atom.GetAtomicNum())

            for bond in mol.GetBonds():
                bond_type = bond_types.get(bond.GetBondType(), 0)
                graph.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond_type,
                )

            graphs.append(graph)

        graphs = list(
            graph_from_networkx(
                graphs, node_labels_tag="atom_type", edge_labels_tag="bond_type"
            )
        )
        return np.array(graphs)


def get_kernel_pipeline(
    kernel_name: str, kernel_hyperparam_tuning: bool
) -> GridSearchCV:
    n_cores = joblib.cpu_count(only_physical_cores=True)
    if kernel_name == "propagation":
        kernel = Propagation(normalize=True, n_jobs=n_cores)
        kernel_params_grid = {"t_max": [1, 2, 3, 4, 5]}
        parallel_kernel = True
    elif kernel_name == "SP":
        kernel = ShortestPath()
        kernel_params_grid = {}
        parallel_kernel = False
    elif kernel_name == "WL":
        kernel = WeisfeilerLehman(normalize=True, n_jobs=n_cores)
        kernel_params_grid = {"n_iter": [1, 2, 3, 4, 5]}
        parallel_kernel = True
    elif kernel_name == "WL-OA":
        kernel = WeisfeilerLehmanOptimalAssignment(normalize=True, n_jobs=n_cores)
        kernel_params_grid = {"n_iter": [1, 2, 3, 4, 5]}
        parallel_kernel = True
    else:
        raise ValueError(f"Kernel type '{kernel_name}' not recognized")

    svm = SVC(
        kernel="precomputed",
        probability=True,
        class_weight="balanced",
        cache_size=1024,
        random_state=0,
    )
    svm_params_grid = {"C": [1e-2, 1e-1, 1, 1e1, 1e2]}

    # create parameters grid with prefixes required by GridSearchCV with Pipeline
    params_grid = {}
    if kernel_hyperparam_tuning:
        for name, values in kernel_params_grid.items():
            params_grid[f"kernel__{name}"] = values

    # we always tune SVM hyperparameters
    for name, values in svm_params_grid.items():
        params_grid[f"svm__{name}"] = values

    pipeline = Pipeline(
        [
            ("smiles_to_graph", SmilesToGrakelGraphTransformer()),
            ("kernel", kernel),
            ("svm", svm),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=(1 if parallel_kernel else n_cores),
    )

    return grid_search


def train_graph_kernel_SVM(
    smiles_train: list[str],
    smiles_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernel_name: str,
    kernel_hyperparam_tuning: bool,
) -> None:
    grid_search = get_kernel_pipeline(kernel_name, kernel_hyperparam_tuning)
    grid_search.fit(smiles_train, y_train)

    y_pred_proba = grid_search.predict_proba(smiles_test)[:, 1]
    y_pred = grid_search.predict(smiles_test)

    mcc = matthews_corrcoef(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Selected hyperparameters:", grid_search.best_params_)
    print(f"MCC: {mcc:.2}")
    print(f"AUROC: {auroc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")


if __name__ == "__main__":
    for split_type in ["time", "maxmin"]:
        print("SPLIT TYPE", split_type)
        smiles_train, smiles_test, y_train, y_test = load_split(split_type)

        for kernel_name in [
            "propagation",
            "SP",
            "WL",
            "WL-OA",
        ]:
            for kernel_hyperparam_tuning in [False, True]:
                if kernel_name == "SP" and kernel_hyperparam_tuning:
                    print(
                        "Shortest paths kernel has no hyperparameters to tune, skipping"
                    )
                    continue

                print(f"Kernel {kernel_name}")
                train_graph_kernel_SVM(
                    smiles_train,
                    smiles_test,
                    y_train,
                    y_test,
                    kernel_name,
                    kernel_hyperparam_tuning,
                )
                print()
