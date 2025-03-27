import networkit
import numpy as np
import torch
from networkit.centrality import Betweenness
from networkit.linkprediction import JaccardIndex
from networkit.nxadapter import nx2nk
from networkit.sparsification import LocalDegreeScore
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.data import Data
from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.utils import degree, to_networkx

from src.utils import create_pyg_dataset, evaluate_sklearn_classifier


def extract_ltp_features(graph: Data) -> np.ndarray:
    """
    Extracts topological LTP features from graph:
    - LDP features, i.e. degree statistics of node and its neighbors
    - additional LTP features: edge betweenness centrality, Jaccard index and
      Local Degree Score

    We use default hyperparameters proposed in LTP paper - simple histogram
    aggregation with 50 bins.
    """
    row, col = graph.edge_index
    N = graph.num_nodes

    # compute degree for each node, using adjacency matrix
    deg = degree(row, N, dtype=torch.float)
    deg = deg.view(-1, 1)

    # compute degree statistics
    aggregation = FusedAggregation(["min", "max", "mean", "std"])
    neighbors_statistics = aggregation(deg[col], row, dim_size=N)

    features = [deg] + neighbors_statistics
    features = [feature.numpy().ravel() for feature in features]

    # transform to NetworKit and calculate additional features
    graph = to_networkx(graph, to_undirected=True)
    graph = nx2nk(graph)
    graph.indexEdges()

    additional_ltp_features = calculate_additional_ltp_features(graph)
    features.extend(additional_ltp_features)

    # aggregate each feature with a histogram and concatenate
    aggregated_features = [np.histogram(feature, bins=50)[0] for feature in features]
    aggregated_features = np.concatenate(aggregated_features)

    return aggregated_features


def calculate_additional_ltp_features(graph: networkit.Graph) -> list[np.ndarray]:
    """
    Calculates 3 additional topological features, proposed in LTP paper:
    - edge betweenness centrality
    - Jaccard index
    - Local Degree Score
    """

    # calculate betweenness and get edge scores
    betweeness = Betweenness(graph, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    edge_betweenness_scores = np.array(scores, dtype=np.float32)

    jaccard_index = JaccardIndex(graph)
    scores = [jaccard_index.run(u, v) for u, v in graph.iterEdges()]
    jaccard_index_scores = np.array(scores, dtype=np.float32)
    jaccard_index_scores = jaccard_index_scores[np.isfinite(scores)]

    # calculate Local Degree Score values
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    local_degree_scores = np.array(scores, dtype=np.float32)

    return [edge_betweenness_scores, jaccard_index_scores, local_degree_scores]


def train_ltp(
    smiles_train: list[str],
    smiles_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    graphs_train = create_pyg_dataset(smiles_train, y_train)
    graphs_test = create_pyg_dataset(smiles_test, y_test)

    X_train = np.stack([extract_ltp_features(graph) for graph in graphs_train])
    X_test = np.stack([extract_ltp_features(graph) for graph in graphs_test])

    rf_clf = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
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
