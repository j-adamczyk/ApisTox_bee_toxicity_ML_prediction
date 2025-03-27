import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from huggingmolecules import GroverConfig, GroverModel
from huggingmolecules.featurization.featurization_grover import (
    GroverBatchEncoding,
    GroverFeaturizer,
)


class GROVER:
    def __init__(self, model_name: str = "grover_large", verbose: bool = True):
        assert model_name in {"grover_base", "grover_large"}
        self.model_name = model_name
        self.verbose = verbose

        self.featurizer = GroverFeaturizer.from_pretrained(model_name)
        self.embedder = GroverEmbedder.from_pretrained(model_name)

    def get_embeddings(self, smiles_list: list[str]) -> np.ndarray:
        iterable = tqdm(smiles_list) if self.verbose else smiles_list
        embeddings = []
        with torch.inference_mode(), torch.no_grad():
            for smiles in iterable:
                inputs = self.featurizer([smiles])
                vec = self.embedder(inputs).flatten().numpy()
                embeddings.append(vec)

        return np.array(embeddings)


class GroverEmbedder(GroverModel):
    def __init__(self, config: GroverConfig):
        super(GroverEmbedder, self).__init__(config)
        self.mol_atom_from_atom_ffn = nn.Identity()
        self.mol_atom_from_bond_ffn = nn.Identity()

    def forward(self, batch: GroverBatchEncoding):
        atom_ffn_output, bond_ffn_output = super().forward(batch)
        return torch.cat([atom_ffn_output, bond_ffn_output], 1)
