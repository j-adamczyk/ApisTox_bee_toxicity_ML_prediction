from typing import Sequence

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols
from sklearn.ensemble import RandomForestClassifier

from src.utils import evaluate_sklearn_classifier


class AtomCountsFeaturizer(BaseFingerprintTransformer):
    def __init__(self):
        # we include first 89 atomic numbers
        super().__init__(n_features_out=90)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        mols = ensure_mols(X)

        X = np.zeros((len(mols), 90))
        for i, mol in enumerate(mols):
            for atom in mol.GetAtoms():
                X[i, atom.GetAtomicNum()] += 1

        return X


def train_atom_counts(
    smiles_train: list[str],
    smiles_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    atom_counts = AtomCountsFeaturizer()
    X_train = atom_counts.transform(smiles_train)
    X_test = atom_counts.transform(smiles_test)

    rf_clf = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        class_weight="balanced",
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
