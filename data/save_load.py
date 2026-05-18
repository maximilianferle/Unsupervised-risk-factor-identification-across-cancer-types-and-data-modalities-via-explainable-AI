import pickle
from pathlib import Path


def save(obj: object, name: Path | str, protocol: int = 4):
    with open(name, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def load(name: Path | str):
    with open(name, "rb") as f:
        return pickle.load(f)
