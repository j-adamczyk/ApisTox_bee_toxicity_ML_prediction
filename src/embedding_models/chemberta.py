import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


class ChemBERTa:
    def __init__(
        self, model_name: str = "DeepChem/ChemBERTa-77M-MTR", verbose: bool = True
    ):
        assert model_name in {
            "DeepChem/ChemBERTa-5M-MTR",
            "DeepChem/ChemBERTa-10M-MTR",
            "DeepChem/ChemBERTa-77M-MTR",
        }
        self.model_name = model_name
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedder = AutoModelForMaskedLM.from_pretrained(model_name)
        self.embedder._modules["lm_head"] = nn.Identity()

        # remove irritating warning
        os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"

    def get_embeddings(self, smiles_list: list[str]) -> np.ndarray:
        iterable = tqdm(smiles_list) if self.verbose else smiles_list
        embeddings = []
        with torch.inference_mode(), torch.no_grad():
            for smiles in iterable:
                encoded_input = self.tokenizer(
                    smiles,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                # transform: [1, n_tokens, 384] -> [n_tokens, 384] -> [384]
                # we extract [CLS] token embeddings (0-th token)
                vec = self.embedder(**encoded_input).logits[0, 0, :]
                vec = vec.numpy()
                embeddings.append(vec)

        return np.array(embeddings)
