import os
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import GetMolFrags
from skfp.datasets.moleculenet import (
    load_bace,
    load_bbbp,
    load_clintox,
    load_hiv,
    load_sider,
    load_tox21,
    load_toxcast,
)
from skfp.distances import (
    bulk_tanimoto_binary_distance,
    bulk_tanimoto_binary_similarity,
)
from skfp.filters import (
    BrenkFilter,
    GhoseFilter,
    HaoFilter,
    LipinskiFilter,
    TiceInsecticidesFilter,
)
from skfp.fingerprints import ECFPFingerprint, LaggnerFingerprint
from skfp.fingerprints.pubchem import PubChemFingerprint
from skfp.preprocessing import MolFromSmilesTransformer
from sklearn.exceptions import DataConversionWarning
from tqdm import tqdm

from utils import DATA_DIR, PLOTS_DIR

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def load_moleculenet_datasets() -> dict[str, list[str]]:
    # functions return tuples (smiles list, labels)
    return {
        "BACE": load_bace()[0],
        "BBBP": load_bbbp()[0],
        "SIDER": load_sider()[0],
        "ClinTox": load_clintox()[0],
        "ToxCast": load_toxcast()[0],
        "Tox21": load_tox21()[0],
        "HIV": load_hiv()[0],
    }


def load_apistox() -> list[str]:
    train_path = os.path.join(DATA_DIR, "maxmin_train.csv")
    test_path = os.path.join(DATA_DIR, "maxmin_test.csv")

    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)])
    smiles = df["SMILES"].tolist()
    return smiles


def smiles_to_mols(datasets: dict[str, list[str]]) -> dict[str, list[Mol]]:
    mol_from_smiles = MolFromSmilesTransformer(suppress_warnings=True)
    datasets = {
        dataset: mol_from_smiles.transform(smiles_list)
        for dataset, smiles_list in datasets.items()
    }
    datasets = {
        dataset: [mol for mol in mols if mol is not None]
        for dataset, mols in datasets.items()
    }
    return datasets


def get_non_medical_mols_perc(mols: list[Mol]) -> float:
    med_atoms = {
        "C",
        "N",
        "O",
        "Si",
        "Cl",
        "S",
        "F",
        "P",
        "B",
        "Se",
        "I",
        "Br",
        "As",
    }

    non_med_mols_count = 0
    for mol in mols:
        mol_non_med_atoms_count = sum(
            atom.GetSymbol() not in med_atoms for atom in mol.GetAtoms()
        )
        non_med_mols_count += mol_non_med_atoms_count > 0

    non_med_atoms_perc = non_med_mols_count / len(mols)
    return non_med_atoms_perc


def get_fragmented_mols_perc(mols: list[Mol]) -> float:
    fragmented_mols = sum(len(GetMolFrags(mol)) > 1 for mol in mols)
    return fragmented_mols / len(mols)


def make_molecular_filters_table(
    datasets: dict[str, list[Mol]], file_name: str
) -> None:
    mol_filters = [
        ("Lipinski", LipinskiFilter()),
        ("Ghose", GhoseFilter()),
        ("Hao", HaoFilter()),
        ("Tice Insecticides", TiceInsecticidesFilter()),
        ("Brenk", BrenkFilter()),
    ]

    datasets_names = [name for name, mols in datasets.items()]
    filter_names = [name for name, filter in mol_filters]

    results = []
    for name, filter in mol_filters:
        filter_results = []
        for dataset_name, mols in datasets.items():
            mols_passing_perc = len(filter.transform(mols)) / len(mols)
            mols_passing_perc = round(100 * mols_passing_perc, 1)
            filter_results.append(mols_passing_perc)
        results.append(filter_results)

    df = pd.DataFrame(results, columns=datasets_names)
    df.index = filter_names
    df.index.name = "Filter"

    file_path = os.path.join(PLOTS_DIR, file_name)
    df.to_csv(file_path)


def compute_property_for_all_datasets(
    datasets: dict[str, list[Mol]], func: Callable
) -> dict[str, list]:
    return {
        dataset: func(mols)
        for dataset, mols in tqdm(datasets.items(), total=len(datasets))
    }


def get_unique_functional_groups_perc(
    mols_data: dict[str, list[Mol]],
) -> dict[str, float]:
    func_groups_dict = {
        dataset: get_dataset_func_groups(mols) for dataset, mols in mols_data.items()
    }

    unique_func_groups_percentages = {}
    for curr_dataset_name, dataset_func_groups in func_groups_dict.items():
        other_datasets_func_groups = set.union(
            *[
                func_groups_dict[dataset_name]
                for dataset_name in func_groups_dict
                if dataset_name != curr_dataset_name
            ]
        )
        unique_func_groups = dataset_func_groups - other_datasets_func_groups
        perc = len(unique_func_groups) / len(dataset_func_groups)
        unique_func_groups_percentages[curr_dataset_name] = perc

    return unique_func_groups_percentages


def get_dataset_func_groups(mols: list[Mol]) -> set[int]:
    # select functional groups that occur in at least 5% of molecules
    fp = LaggnerFingerprint(n_jobs=-1)
    X = fp.transform(mols)
    func_groups_perc = X.sum(axis=0) / len(mols)

    detected_func_groups = {
        i for i in range(len(func_groups_perc)) if func_groups_perc[i] >= 0.05
    }

    return detected_func_groups


def compute_normalized_n_circles(
    mols: list[Mol],
    distance_matrix_func: Callable = bulk_tanimoto_binary_distance,
    threshold: float = 0.7,
) -> float:
    fp = ECFPFingerprint(fp_size=1024)  # original paper setting
    fps = fp.transform(mols)

    # algorithm 3 of #Circles paper (fast sequential approximation)
    # exact computation is exponential
    rng = np.random.default_rng(0)
    rng.shuffle(fps)

    dists = distance_matrix_func(fps)
    np.fill_diagonal(dists, np.inf)  # exclude distance from molecule to itself

    n_circles = 0
    for i, fp in enumerate(fps):
        min_dist = dists[i].min()
        if min_dist > threshold:
            n_circles += 1

    # we normalize n_circles by dataset size to measure relative diversity
    normalized_n_circles = n_circles / len(mols)
    return normalized_n_circles


def make_barplot(datasets: dict, file_name: str, percentage: bool = True) -> None:
    df = pd.DataFrame(datasets.items(), columns=["Dataset", "Value"])
    df = df.sort_values(by="Value")

    dataset_names = [
        "BACE",
        "BBBP",
        "SIDER",
        "ClinTox",
        "ToxCast",
        "Tox21",
        "HIV",
        "ApisTox",
    ]
    colors = sns.color_palette("Blues", len(dataset_names))
    colors = {record: color for record, color in zip(dataset_names, colors)}

    plt.figure(figsize=(8, 5))
    barplot = sns.barplot(
        x="Value",
        y="Dataset",
        data=df,
        palette=[colors[rec] for rec in datasets["Dataset"]],
    )

    if percentage:
        current_labels = [label.get_text() for label in barplot.get_xticklabels()]
        new_labels = [f"{int(float(label) * 100)}%" for label in current_labels]
        barplot.set_xticklabels(new_labels)

    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()

    fig = barplot.get_figure()
    output_path = os.path.join(PLOTS_DIR, file_name)
    fig.savefig(output_path, dpi=300)


def create_pairwise_tanimito_similarity_plot(
    datasets: dict[str, list[Mol]],
) -> pd.DataFrame:
    fp = PubChemFingerprint(n_jobs=-1)

    datasets = list(datasets.items())

    datasets_names = [name for name, mols in datasets]
    datasets_fps = [fp.transform(mols) for name, mols in datasets]

    # matrix of average Tanimoto similarities between datasets
    # average self-similarity on the diagonal
    dataset_sims = np.zeros((len(datasets), len(datasets)))

    for i, X_i in enumerate(datasets_fps):
        for j, X_j in enumerate(datasets_fps[: i + 1]):
            sim = bulk_tanimoto_binary_similarity(X_i, X_j)
            avg_sim = np.round(np.mean(sim), 2)
            dataset_sims[i][j] = avg_sim

    df = pd.DataFrame(dataset_sims, columns=datasets_names)
    df.index = datasets_names
    return df


def visualize_tanimito_similarity_map(df: pd.DataFrame) -> None:
    # this turns off plotting the upper triangle
    df.replace(0, np.NAN, inplace=True)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        df,
        annot=True,
        cmap="mako",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Tanimoto Similarity"},
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(False)
    fig = ax.get_figure()
    fig_path = os.path.join(PLOTS_DIR, "tanimoto_similarity_map.png")
    fig.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    datasets = {**load_moleculenet_datasets(), "ApisTox": load_apistox()}
    datasets = smiles_to_mols(datasets)

    non_med_atoms_data = compute_property_for_all_datasets(
        datasets, func=get_non_medical_mols_perc
    )
    make_barplot(non_med_atoms_data, "non_med_atoms.png")

    fragmented_mols_data = compute_property_for_all_datasets(
        datasets, func=get_fragmented_mols_perc
    )
    make_barplot(fragmented_mols_data, "fragmented_molecules.png")

    unique_fragments_data = get_unique_functional_groups_perc(datasets)
    make_barplot(unique_fragments_data, "unique_fragments.png")

    make_molecular_filters_table(datasets, "filters_results.csv")

    n_circles_data = compute_property_for_all_datasets(
        datasets, func=compute_normalized_n_circles
    )
    make_barplot(n_circles_data, "nCircles.png")

    create_pairwise_tanimito_similarity_plot(datasets)
