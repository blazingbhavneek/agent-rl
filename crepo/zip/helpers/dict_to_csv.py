import os  # for capturing session variables.
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from state.state import State


def save_dict_csv(
    data_dict,
    path_of_csv: Path | None = None,
    nested_joiner: str = "->",
    save: bool = True,
    path_no: int = -1,
):
    """
    Takes a dictionary representing the Combined model and returns
    a flattened pandas DataFrame.
    """
    project_state = State()
    if path_of_csv is None and save:
        results_dir = Path("/home/seigyo/c_repo/c_repo/results/csv_results")
        results_dir.mkdir(exist_ok=True, parents=True)
        path_of_csv = results_dir / f"{project_state.get('PROJECT_NAME')}.csv"
    data_list = [data_dict] if isinstance(data_dict, dict) else data_dict

    # 1. Standard Flattening
    data_list = [data_dict] if isinstance(data_dict, dict) else data_dict
    df = pd.json_normalize(data_list, sep=nested_joiner)

    # 2. Get the exact order of keys from the original dictionary
    # We create a flat list of keys in the order they appear
    def get_ordered_keys(d, sep=nested_joiner):
        keys = []
        for k, v in d.items():
            if isinstance(v, dict):
                # If it's a sub-dict (like target_number),
                # get its keys and prefix them
                sub_keys = get_ordered_keys(v, sep=sep)
                keys.extend([f"{k}{sep}{sk}" for sk in sub_keys])
            else:
                keys.append(k)
        return keys

    # 3. Re-order the columns based on the dictionary structure
    ordered_columns = get_ordered_keys(data_list[0])

    # Filter to ensure we only look for columns that actually exist in the DF
    final_columns = [col for col in ordered_columns if col in df.columns]

    # 4. Final Formatting
    df = df[final_columns]

    if f"target_number{nested_joiner}ans" in df.columns:
        df[f"target_number{nested_joiner}ans"] = df[
            f"target_number{nested_joiner}ans"
        ].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)
    if "call_number" in df.columns:
        df["call_number"] = df["call_number"].apply(
            lambda x: "NA" if x == -1 or str(x) == "-1" or str(x) == "-1.0" else str(x)
        )

    # TODO : IN future if we want a path_no column as well.
    if path_no != -1:
        df["PATH_NO"] = path_no
    if save:
        if path_of_csv.exists():
            # CRITICAL: keep_default_na=False prevents 'NA' from becoming NaN
            loaded = pd.read_csv(path_of_csv, keep_default_na=False)

            # Combine the new dataframe with the old one
            loaded = pd.concat([loaded, df], axis=0, ignore_index=True)

            # Re-apply the formatting to the WHOLE column to ensure consistency
            if "call_number" in loaded.columns:
                loaded["call_number"] = loaded["call_number"].apply(
                    lambda x: "NA" if str(x) in ["-1", "-1.0", "nan", "NA"] else str(x)
                )

            loaded.to_csv(path_of_csv, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(path_of_csv, index=False, encoding="utf-8-sig")
        return
    return df  # if we want to do something with it.
