from __future__ import annotations

from dataclasses import asdict


def dataclass_to_jsonable(obj):
    """
    Recursively convert dataclasses/lists/dicts to JSON-serializable structures.
    """
    if isinstance(obj, list):
        return [dataclass_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: dataclass_to_jsonable(v) for k, v in obj.items()}
    return obj
