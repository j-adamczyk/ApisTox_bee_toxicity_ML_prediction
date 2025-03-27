import numpy as np
import torch
from tqdm import tqdm

from huggingmolecules import MatConfig, MatFeaturizer, MatModel


class MAT:
    def __init__(self, model_name: str = "mat_masking_20M", verbose: bool = True):
        assert model_name in {"mat_masking_200k", "mat_masking_2M", "mat_masking_20M"}
        self.model_name = model_name
        self.verbose = verbose

        self.featurizer = MatFeaturizer.from_pretrained(model_name)
        self.embedder = MatEmbedder.from_pretrained(model_name)

    def get_embeddings(self, smiles_list: list[str]) -> np.ndarray:
        iterable = tqdm(smiles_list) if self.verbose else smiles_list
        embeddings = []
        with torch.inference_mode(), torch.no_grad():
            for smiles in iterable:
                inputs = self.featurizer([smiles])
                vec = self.embedder(inputs).flatten().numpy()
                embeddings.append(vec)

        return np.array(embeddings)


class MatEmbedder(MatModel):
    def __init__(self, config: MatConfig):
        super(MatEmbedder, self).__init__(config)
        self.generator.proj = torch.nn.Identity()
