from .mit_loader import MITLoader, load_mit_processed
from .bsebench_loader import BSEBenchLoader, BSECell, parse_policy
from .mit_meta import extract_all as extract_mit_meta, load_meta as load_mit_meta

__all__ = ["MITLoader", "load_mit_processed",
           "BSEBenchLoader", "BSECell", "parse_policy",
           "extract_mit_meta", "load_mit_meta",
           "BSEEarlyPredictDataset", "severson_split"]


def __getattr__(name):
    # Lazy-import torch-dependent helpers so the package is usable without torch
    if name in {"BSEEarlyPredictDataset", "severson_split"}:
        from .dataset import BSEEarlyPredictDataset, severson_split
        if name == "BSEEarlyPredictDataset":
            return BSEEarlyPredictDataset
        return severson_split
    raise AttributeError(name)
