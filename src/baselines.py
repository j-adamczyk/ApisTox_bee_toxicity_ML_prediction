from src.baselines.atom_counts import train_atom_counts
from src.baselines.ltp import train_ltp
from src.baselines.moltop import train_moltop
from src.utils import load_split


if __name__ == "__main__":
    for split_type in ["time", "maxmin"]:
        print("SPLIT TYPE", split_type)
        smiles_train, smiles_test, y_train, y_test = load_split(split_type)

        for model_name, model_func in [
            ("Atom counts", train_atom_counts),
            ("LTP", train_ltp),
            ("MOLTOP", train_moltop),
        ]:
            print(model_name)
            model_func(
                smiles_train,
                smiles_test,
                y_train,
                y_test,
            )
            print()
