import numpy as np
import torch
from tqdm import tqdm

from huggingmolecules import RMatConfig, RMatFeaturizer, RMatModel


class RMAT:
    def __init__(self, model_name: str = "rmat_4M_rdkit", verbose: bool = True):
        assert model_name in {"rmat_4M", "rmat_4M_rdkit"}
        self.model_name = model_name
        self.verbose = verbose

        self.featurizer = RMatFeaturizer.from_pretrained(model_name)
        self.embedder = RMatEmbedder.from_pretrained(model_name)

    def get_embeddings(self, smiles_list: list[str]) -> np.ndarray:
        iterable = tqdm(smiles_list) if self.verbose else smiles_list
        embeddings = []
        with torch.inference_mode(), torch.no_grad():
            for smiles in iterable:
                inputs = self.featurizer([smiles])
                vec = self.embedder(inputs).flatten().numpy()
                embeddings.append(vec)

        return np.array(embeddings)


class RMatEmbedder(RMatModel):
    def __init__(self, config: RMatConfig):
        super(RMatEmbedder, self).__init__(config)
        self.generator.proj = torch.nn.Identity()
