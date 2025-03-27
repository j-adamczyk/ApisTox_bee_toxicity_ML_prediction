import networkit
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from networkit.centrality import Betweenness
from networkit.linkprediction import AdjustedRandIndex
from networkit.sparsification import SCANStructuralSimilarityScore, TriangleEdgeScore
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.data import Data
from torch_geometric.nn.aggr.fused import FusedAggregation

from src.utils import create_pyg_dataset, evaluate_sklearn_classifier


def extract_moltop_features(data: Data, n_bins: int) -> np.array:
    row, col = data.edge_index
    num_nodes = data.num_nodes

    deg = torch_geometric.utils.degree(row, num_nodes, dtype=torch.float)
    deg = deg.view(-1, 1)
    deg_col = deg[col]

    aggr = FusedAggregation(["min", "max", "mean", "std"])
    ldp_features = [deg] + aggr(deg_col, row, dim_size=num_nodes)
    ldp_features = [feature.numpy().ravel() for feature in ldp_features]

    graph = torch_geometric.utils.to_networkit(
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        num_nodes=data.num_nodes,
        directed=False,
    )
    graph.indexEdges()

    topological_descriptors = [
        calculate_edge_betweenness(graph),
        calculate_adjusted_rand_index(graph),
        calculate_scan_structural_similarity_score(graph),
    ]

    molecular_features = []

    # MoleculeNet OGB featurization - atomic number is the first feature
    atom_features = data.x[:, 0]
    atom_features = F.one_hot(atom_features, 120).float()
    atom_features = atom_features[:, :89]

    atom_types_mean = torch.mean(atom_features, dim=0).numpy()
    atom_types_std = torch.std(atom_features, dim=0).numpy()
    atom_types_sum = torch.sum(atom_features, dim=0).numpy()

    # in case of all-zero features standard deviation is NaN, we fill it with zeros
    atom_types_std[np.isnan(atom_types_std)] = 0

    atom_type_features = np.concatenate(
        (
            atom_types_mean,
            atom_types_std,
            atom_types_sum,
        )
    )
    molecular_features.append(atom_type_features)

    # MoleculeNet OGB featurization - bond type is the first feature
    bond_features = data.edge_attr[:, 0]
    bond_features = F.one_hot(bond_features, 5).float()

    # there are a few graphs without edge features, we use all zeros
    if bond_features.shape[0] == 0:
        bond_features = np.zeros(15, dtype=float)
    else:
        bond_features_mean = torch.mean(bond_features, dim=0).numpy()
        bond_features_std = torch.std(bond_features, dim=0).numpy()
        bond_features_sum = torch.sum(bond_features, dim=0).numpy()

        # in case of all-zero features standard deviation is NaN, we fill it with zeros
        bond_features_std[np.isnan(bond_features_std)] = 0

        bond_features = np.concatenate(
            (
                bond_features_mean,
                bond_features_std,
                bond_features_sum,
            )
        )

    molecular_features.append(bond_features)

    # aggregate all features with histograms
    topological_features = []

    for feature in ldp_features[:3]:
        values = np.bincount(feature.astype(int), minlength=11)[:11]
        values = values.astype(float)
        topological_features.append(values)

    for feature in ldp_features[3:]:
        values, _ = np.histogram(feature, bins=n_bins)
        topological_features.append(values)

    for feature in topological_descriptors:
        values, _ = np.histogram(feature, bins=n_bins)
        topological_features.append(values)

    features = np.concatenate(topological_features + molecular_features)
    return features


def calculate_edge_betweenness(graph: networkit.Graph) -> np.ndarray:
    betweeness = Betweenness(graph, normalized=True, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    scores = np.array(scores, dtype=float)
    return scores


def calculate_adjusted_rand_index(graph: networkit.Graph) -> np.ndarray:
    index = AdjustedRandIndex(graph)
    scores = [index.run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=float)
    return scores


def calculate_scan_structural_similarity_score(graph: networkit.Graph) -> np.ndarray:
    triangles = TriangleEdgeScore(graph)
    triangles.run()
    triangles = triangles.scores()

    score = SCANStructuralSimilarityScore(graph, triangles)
    score.run()
    scores = score.scores()
    scores = np.array(scores, dtype=float)
    return scores


def train_moltop(
    smiles_train: list[str],
    smiles_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    graphs_train = create_pyg_dataset(smiles_train, y_train)
    graphs_test = create_pyg_dataset(smiles_test, y_test)

    n_bins = int(np.median([data.num_nodes for data in graphs_train]))

    X_train = np.stack(
        [extract_moltop_features(graph, n_bins) for graph in graphs_train]
    )
    X_test = np.stack([extract_moltop_features(graph, n_bins) for graph in graphs_test])

    rf_clf = RandomForestClassifier(
        n_estimators=1000,
        criterion="entropy",
        min_samples_split=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
    )
    rf_clf.fit(X_train, y_train)

    evaluate_sklearn_classifier(
        model=rf_clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
