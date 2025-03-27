import ast
import os.path
import pathlib
import shutil

import joblib
import numpy as np
import pandas as pd
from grakel import WeisfeilerLehmanOptimalAssignment
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

from src.graph_kernels import SmilesToGrakelGraphTransformer
from src.utils import load_split

UNCERTAIN_DATA_FILE_PATH = "../data/uncertain.csv"
PLOTS_DIR = "../plots/uncertain_mols"

DATASET_NAME_TO_COMMON_NAME = {
    "1,1'-(2,2,2-Trichloroethylidene)bis[4-chlorobenzene]": "Dichlorodifenylotrichloroetan (DDT)",
    "[C(E)]-N-[(2-Chloro-5-thiazolyl)methyl]-N'-methyl-N''-nitroguanidine": "Clothianidin",
    "O,O-Dimethyl S-[2-(methylamino)-2-oxoethyl]phosphorodithioic acid ester": "Dimethoate",
    "1-Naphthalenol methylcarbamate": "Carbaryl",
    "6,7,8,9,10,10-Hexachloro-1,5,5a,6,9,9a-hexahydro-6,9-methano-2,4,3-benzodioxathiepin 3-oxide": "Endosulfan",
    "(1E)-N-[(6-Chloro-3-pyridinyl)methyl]-N'-cyano-N-methylethanimidamide": "Acetamiprid",
    "2-(4-Chloro-2-methylphenoxy)propanoic acid": "Mecoprop-P",
    "2-[(Dimethoxyphosphinothioyl)thio]butanedioic acid 1,4-diethyl ester": "Malathion",
    "(1R,3R)-3-(2,2-Dibromoethenyl)-2,2-dimethylcyclopropanecarboxylic acid (S)cyano(3-phenoxyphenyl)methyl ester": "Deltamethrin",
    "(4aS)-7-Chloro-2,5-dihydro-2-[[(methoxycarbonyl)[4-(trifluoromethoxy)phenyl]amino]carbonyl]indeno[1,2-e][1,3,4]oxadiazine-4a(3H)-carboxylic acid methyl ester": "Indoxacarb",
    "Toxaphene": "Toxaphene",
    "S-[(6-Chloro-2-oxo-3(2H)-benzoxazolyl)methyl]O,O-diethyl ester phosphorodithioic acid": "Phosalone",
    "N,N-Dimethyl-N'-[3-(trifluoromethyl)phenyl]urea": "Fluometuron",
    "Phosphoramidothioic acid, O,S-Dimethyl ester": "Methamidophos",
    "3,3-Dimethylbutanoic acid 2-oxo-3-(2,4,6-trimethylphenyl)-1-oaxspiro[4.4]non-3-en-4-yl ester": "Spiromesifen",
    "2,2-Dimethylpropanoic acid  8-(2,6-diethyl-4-methylphenyl)-1,2,4,5- tetrahydro-7-oxo-7H-pyrazolo[1,2-d][1,4,5]oxadiazepin-9-yl ester": "Pinoxaden",
    "2-[4-(1,1-Dimethylethyl)phenoxy]cyclohexyl 2-propynyl ester sulfurous acid": "Propargite",
}


def train_wl_oa_kernel_svm(
    smiles_train: list[str],
    y_train: np.ndarray,
) -> Pipeline:
    n_cores = joblib.cpu_count(only_physical_cores=True)
    smiles_to_grakel = SmilesToGrakelGraphTransformer()
    kernel = WeisfeilerLehmanOptimalAssignment(normalize=True, n_jobs=n_cores)
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


def load_uncertain_molecules() -> pd.DataFrame:
    df = pd.read_csv(UNCERTAIN_DATA_FILE_PATH)
    df["measurements"] = df["measurements"].apply(ast.literal_eval)
    df = df[["name", "toxicity_type", "SMILES", "measurements"]]
    return df


def make_predictions_and_plots(model: Pipeline, df_uncertain: pd.DataFrame) -> None:
    preds = model.predict(df_uncertain["SMILES"])
    for idx, row in df_uncertain.iterrows():
        data = row["measurements"]
        y_pred = preds[idx]

        pd.Series(data).plot.hist(bins=len(data), rwidth=0.8)

        plt.axvline(
            x=11, linewidth=3, color="red", label="Toxicity threshold\n(11 μg/bee)"
        )

        name = row["common_name"]
        pred_str = "toxic" if y_pred else "non-toxic"
        title = f"{name}\nPrediction: {pred_str}"

        plt.title(title, wrap=True)
        plt.legend(loc="upper right")
        plt.savefig(f"{PLOTS_DIR}/{idx + 1}.png", bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    if os.path.exists(PLOTS_DIR):
        shutil.rmtree(PLOTS_DIR)
    pathlib.Path(PLOTS_DIR).mkdir(parents=True)

    smiles_train, smiles_test, y_train, y_test = load_split("time")
    smiles_list = smiles_train + smiles_test
    y = np.concatenate((y_train, y_test))
    pipeline = train_wl_oa_kernel_svm(smiles_list, y)

    df_uncertain = load_uncertain_molecules()
    df_uncertain["common_name"] = df_uncertain["name"].replace(
        DATASET_NAME_TO_COMMON_NAME
    )
    make_predictions_and_plots(pipeline, df_uncertain)
