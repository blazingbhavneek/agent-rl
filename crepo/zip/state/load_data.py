import json
import pickle
from pathlib import Path
from typing import Literal

from state.state import State


def read_file(file_path: Path) -> any:
    if not file_path:
        raise (f"File not found: {file_path}")

    if file_path.suffix == ".json":
        with open(file_path, "r") as f:
            return json.load(f)
        # return data
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise (f"Function not implement for extension {file_path.suffix}")


def save_file(data: any, save_location: Path) -> None:
    save_location.mkdir(parents=True, exist_ok=True)

    if save_location.suffix == ".json":
        with open(save_location, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif save_location.suffix == ".pkl":
        with open(save_location, "wb") as f:
            pickle.dump(f)
    else:
        raise (f"Function not implement for extension {save_location.suffix}")


def load_json_files(json_dir: Path, state: State) -> None:
    function_pointer_path = json_dir / "function_callback_info.json"
    mpf_data_path = json_dir / "mpf_data.json"
    try:
        data = read_file(mpf_data_path)
        state.set(name="FUNCTION_TYPES", value=data)
        data = read_file(function_pointer_path)
        state.set(name="FUNCTION_POINTER_ARGS", value=data)
    except Exception as e:
        raise


def load_pickle_files(json_dir: Path, pickle_dir: Path, state: State) -> None:
    function_map_pickle_path = pickle_dir / "function_map.pkl"
    if not function_map_pickle_path.exists():
        function_map_json_path = json_dir / "combined_data.json"
        data = read_file(function_map_json_path)  # data is now a Python dict
        save_file(data, function_map_pickle_path)
        state.set(name="FUNCTION_MAP", value=data)

    else:
        data = read_file(function_map_pickle_path)
        state.set(name="FUNCTION_MAP", value=data)


def load_files() -> State:
    """
    Loads all the json and pickle files necessary and then set them in the state.
    """
    json_dir = Path(__file__).resolve().parent.parent / "json_data"
    pickle_dir = Path(__file__).resolve().parent.parent / "pickle_data"
    GLOBAL_STATE = State()
    load_json_files(json_dir=json_dir, state=GLOBAL_STATE)
    load_pickle_files(json_dir, pickle_dir=pickle_dir, state=GLOBAL_STATE)

    return GLOBAL_STATE
