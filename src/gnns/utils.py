import os
import random

import numpy as np
import torch
from ogb.utils import smiles2graph
from torch.utils.data import Dataset
from torch_geometric.data import Data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_determinism(random_state: int = 0):
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)


class GraphDataset(Dataset):
    def __init__(self, data_list: list[Data]):
        super(Dataset, self).__init__()
        self.data_list = data_list
        self.y = np.array([data.y for data in data_list])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def create_pyg_dataset(
    smiles_list: list[str],
    labels: np.ndarray,
) -> GraphDataset:
    graphs = [smiles2graph(smiles) for smiles in smiles_list]
    graphs = [
        Data(
            x=torch.from_numpy(graph["node_feat"]),
            edge_index=torch.from_numpy(graph["edge_index"]),
            edge_attr=torch.from_numpy(graph["edge_feat"]),
            y=torch.tensor(label),
        )
        for graph, label in zip(graphs, labels)
    ]
    return GraphDataset(graphs)
