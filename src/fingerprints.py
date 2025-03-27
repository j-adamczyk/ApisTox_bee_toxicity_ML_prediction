import joblib
import numpy as np
import skfp.fingerprints as fps
from skfp.preprocessing import MolFromSmilesTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.utils import evaluate_sklearn_classifier, load_split


def get_fingerprint_pipeline(fp_name: str, fp_hyperparam_tuning: bool) -> GridSearchCV:
    """
    Creates classification pipeline for a given molecular fingerprint.
    Note that it has grid search with cross-validation, so that .fit()
    tunes hyperparameters.
    """
    n_cores = joblib.cpu_count(only_physical_cores=True)

    if fp_name == "AtomPairs":
        fingerprint = fps.AtomPairFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "count": [False, True],
        }
    elif fp_name == "Avalon":
        fingerprint = fps.AvalonFingerprint()
        fp_params_grid = {
            "fp_size": [256, 512, 1024, 2048],
            "count": [False, True],
        }
    elif fp_name == "Autocorrelation":
        fingerprint = fps.AutocorrFingerprint()
        fp_params_grid = {}
    elif fp_name == "ECFP":
        fingerprint = fps.ECFPFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "radius": [2, 3],
            "count": [False, True],
        }
    elif fp_name == "ERG":
        fingerprint = fps.ERGFingerprint()
        fp_params_grid = {"max_path": list(range(5, 26))}
    elif fp_name == "EState":
        fingerprint = fps.EStateFingerprint()
        fp_params_grid = {"variant": ["sum", "bit", "count"]}
    elif fp_name == "FCFP":
        fingerprint = fps.ECFPFingerprint(use_pharmacophoric_invariants=True)
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "radius": [2, 3],
            "count": [False, True],
        }
    elif fp_name == "FunctionalGroups":
        fingerprint = fps.FunctionalGroupsFingerprint()
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "GhoseCrippen":
        fingerprint = fps.GhoseCrippenFingerprint(n_cores)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "KlekotaRoth":
        fingerprint = fps.KlekotaRothFingerprint(n_jobs=n_cores)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "Laggner":
        fingerprint = fps.LaggnerFingerprint(n_jobs=n_cores)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "Layered":
        fingerprint = fps.LayeredFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "max_path": [5, 6, 7, 8, 9],
        }
    elif fp_name == "Lingo":
        fingerprint = fps.LingoFingerprint()
        fp_params_grid = {
            "substring_length": [3, 4, 5, 6],
            "count": [False, True],
        }
    elif fp_name == "MACCS":
        fingerprint = fps.MACCSFingerprint(n_jobs=n_cores)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "MAP":
        fingerprint = fps.MAPFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "radius": [2, 3],
            "count": [False, True],
        }
    elif fp_name == "Mordred":
        fingerprint = fps.MordredFingerprint(n_jobs=n_cores)
        fp_params_grid = {}
    elif fp_name == "MQNs":
        fingerprint = fps.MQNsFingerprint()
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "Pattern":
        fingerprint = fps.PatternFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "tautomers": [False, True],
        }
    elif fp_name == "PhysiochemicalProperties":
        fingerprint = fps.PhysiochemicalPropertiesFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "variant": ["BP", "BT"],
        }
    elif fp_name == "PubChem":
        fingerprint = fps.PubChemFingerprint(n_jobs=n_cores)
        fp_params_grid = {"count": [False, True]}
    elif fp_name == "RDKit":
        fingerprint = fps.RDKitFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "max_path": [5, 6, 7, 8, 9],
            "count": [False, True],
        }
    elif fp_name == "RDKit2DDescriptors":
        fingerprint = fps.RDKit2DDescriptorsFingerprint()
        fp_params_grid = {"normalized": [False, True]}
    elif fp_name == "SECFP":
        fingerprint = fps.SECFPFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "radius": [1, 2, 3, 4],
        }
    elif fp_name == "TopologicalTorsion":
        fingerprint = fps.TopologicalTorsionFingerprint()
        fp_params_grid = {
            "fp_size": [512, 1024, 2048],
            "count": [False, True],
        }
    elif fp_name == "VSA":
        fingerprint = fps.VSAFingerprint()
        fp_params_grid = {}
    else:
        raise ValueError(f"Fingerprint name '{fp_name}' not recognized")

    rf = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        class_weight="balanced",
        random_state=0,
    )
    rf_params_grid = {
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    # create parameters grid with prefixes required by GridSearchCV with Pipeline
    params_grid = {}
    if fp_hyperparam_tuning:
        for name, values in fp_params_grid.items():
            params_grid[f"fp__{name}"] = values

    # we always tune Random Forest hyperparameters
    for name, values in rf_params_grid.items():
        params_grid[f"rf__{name}"] = values

    pipeline = Pipeline([("fp", fingerprint), ("rf", rf)])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=joblib.cpu_count(only_physical_cores=True),
    )

    return grid_search


def train_molecular_fingerprint(
    smiles_train: list[str],
    smiles_test: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    fp_name: str,
    fp_hyperparam_tuning: bool,
) -> None:
    grid_search = get_fingerprint_pipeline(fp_name, fp_hyperparam_tuning)
    grid_search.fit(smiles_train, y_train)
    fp = grid_search.best_estimator_[0]
    rf_clf = grid_search.best_estimator_[1]
    hyperparams = grid_search.best_params_

    X_train = fp.transform(smiles_train)
    X_test = fp.transform(smiles_test)

    evaluate_sklearn_classifier(
        model=rf_clf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    print(f"Hyperparameters: {hyperparams}")


if __name__ == "__main__":
    mol_from_smiles = MolFromSmilesTransformer()

    for split_type in ["time", "maxmin"]:
        print("SPLIT TYPE", split_type)
        smiles_train, smiles_test, y_train, y_test = load_split(split_type)

        for fp_name in [
            "AtomPairs",
            "Avalon",
            "Autocorrelation",
            "ECFP",
            "ERG",
            "EState",
            "FCFP",
            "FunctionalGroups",
            "GhoseCrippen",
            "KlekotaRoth",
            "Laggner",
            "Layered",
            "Lingo",
            "MACCS",
            "MAP",
            "Mordred",
            "MQNs",
            "Pattern",
            "PhysiochemicalProperties",
            "PubChem",
            "RDKit",
            "RDKit2DDescriptors",
            "SECFP",
            "TopologicalTorsion",
            "VSA",
        ]:
            for fp_hyperparam_tuning in [False, True]:
                print(f"FINGERPRINT {fp_name}, tuning {fp_hyperparam_tuning}")
                train_molecular_fingerprint(
                    smiles_train,
                    smiles_test,
                    y_train,
                    y_test,
                    fp_name,
                    fp_hyperparam_tuning,
                )
                print()
