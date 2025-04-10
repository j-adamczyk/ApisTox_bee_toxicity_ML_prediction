import os
import pathlib
import shutil

import exmol
import joblib
import numpy as np
from grakel import WeisfeilerLehmanOptimalAssignment
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC

from src.graph_kernels import SmilesToGrakelGraphTransformer
from src.utils import load_split

MODEL_FILES_DIR = "model_files"
PLOTS_DIR = "../plots/counterfactuals"


def train_wl_oa_kernel_svm(smiles_train: list[str], y_train: np.ndarray) -> Pipeline:
    n_cores = joblib.cpu_count(only_physical_cores=True)
    kernel = WeisfeilerLehmanOptimalAssignment(normalize=True, n_jobs=n_cores)
    smiles_to_grakel = SmilesToGrakelGraphTransformer()
    svm = SVC(
        kernel="precomputed",
        probability=True,
        class_weight="balanced",
        cache_size=1024,
        random_state=0,
    )
    pipeline = make_pipeline(smiles_to_grakel, kernel, svm)
    pipeline.fit(smiles_train, y_train)
    return pipeline


def make_plots(
    pipeline: Pipeline,
    smiles_test: list[str],
    y_test: np.ndarray,
) -> None:
    smiles_test = np.array(smiles_test)
    smiles_test_neg = smiles_test[y_test == 0]
    smiles_test_pos = smiles_test[y_test == 1]

    for i, smi in enumerate(smiles_test_neg):
        model_func = lambda x: pipeline.predict(x)
        samples = exmol.sample_space(smi, model_func, num_samples=5000)
        cfs = exmol.cf_explain(samples, nmols=3, filter_nondrug=False)
        exmol.plot_cf(cfs)
        plt.savefig(f"{PLOTS_DIR}/negative_{i + 1}.png")
        plt.clf()

    for i, smi in enumerate(smiles_test_pos):
        model_func = lambda x: pipeline.predict(x)
        samples = exmol.sample_space(smi, model_func, num_samples=5000)
        cfs = exmol.cf_explain(samples, nmols=3, filter_nondrug=False)
        exmol.plot_cf(cfs)
        plt.savefig(f"{PLOTS_DIR}/positive_{i + 1}.png")
        plt.clf()


if __name__ == "__main__":
    if os.path.exists(PLOTS_DIR):
        shutil.rmtree(PLOTS_DIR)
    pathlib.Path(PLOTS_DIR).mkdir(parents=True)

    smiles_train, smiles_test, y_train, y_test = load_split("time")
    pipeline = train_wl_oa_kernel_svm(smiles_train, y_train)
    make_plots(pipeline, smiles_test, y_test)
