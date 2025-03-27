import numpy as np
from deepchem.feat import Mol2VecFingerprint
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from src.embedding_models.chemberta import ChemBERTa
from src.embedding_models.grover import GROVER
from src.embedding_models.mat import MAT
from src.embedding_models.rmat import RMAT
from src.utils import evaluate_sklearn_classifier, load_split


def get_mol_embeddings(
    model_name: str,
    smiles_list: list[str],
) -> np.ndarray:
    if model_name == "ChemBERTa":
        model = ChemBERTa()
        return model.get_embeddings(smiles_list)
    elif model_name == "GROVER":
        model = GROVER()
        return model.get_embeddings(smiles_list)
    elif model_name == "MAT":
        model = MAT()
        return model.get_embeddings(smiles_list)
    elif model_name == "Mol2Vec":
        model = Mol2VecFingerprint()
        return np.row_stack([model.featurize(smi) for smi in smiles_list])
    elif model_name == "R-MAT":
        model = RMAT()
        return model.get_embeddings(smiles_list)
    else:
        raise ValueError(f"Model {model_name} not recognized")


def train_mol_embeddings_model(
    model_name: str,
    smiles_train: list[str],
    smiles_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    X_train = get_mol_embeddings(model_name, smiles_train)
    X_test = get_mol_embeddings(model_name, smiles_test)

    model = LogisticRegressionCV(
        Cs=100,
        cv=5,
        scoring="roc_auc",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
    )
    model.fit(X_train, y_train)

    # extract tuned C value to avoid re-tuning during retraining in evaluation
    model = LogisticRegression(
        C=model.C_[0],
        class_weight="balanced",
        random_state=0,
        max_iter=1000,
        n_jobs=-1,
    )

    evaluate_sklearn_classifier(model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    for split_type in ["time", "maxmin"]:
        print("SPLIT TYPE", split_type)
        smiles_train, smiles_test, y_train, y_test = load_split(split_type)

        for model_name in [
            "ChemBERTa",
            "GROVER",
            "MAT",
            "Mol2Vec",
            "R-MAT",
        ]:
            print(model_name)
            train_mol_embeddings_model(
                model_name,
                smiles_train,
                smiles_test,
                y_train,
                y_test,
            )
            print()
