"""Serialization utilities for BWD objects."""

import json

import numpy as np

from .bwd import BWD
from .bwd_random import BWDRandom
from .multi_bwd import MultiBWD
from .online import Online

name2class = {
    "BWD": BWD,
    "BWDRandom": BWDRandom,
    "MultiBWD": MultiBWD,
    "Online": Online,
}


def normalize(to_serialize):
    """Normalize data structures for JSON serialization

    Recursively converts numpy arrays to lists and normalizes nested dictionaries
    to ensure all data types are JSON-serializable.

    Parameters
    ----------
    to_serialize : dict
        Dictionary containing data to normalize

    Returns
    -------
    dict
        Normalized dictionary with JSON-compatible types
    """
    result = {}
    for k, v in to_serialize.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, dict):
            v = normalize(v)
        result[k] = v
    return result


def serialize(obj):
    """Serialize a balancer object to JSON string

    Serializes the balancer's definition and state to a JSON-formatted string
    that can be saved and later deserialized.

    Parameters
    ----------
    obj : BWD, BWDRandom, MultiBWD, or Online
        The balancer object to serialize

    Returns
    -------
    str
        JSON string representation of the object
    """
    return json.dumps(
        {
            str(type(obj).__name__): {
                "definition": normalize(obj.definition),
                "state": normalize(obj.state),
            }
        }
    )


def deserialize(json_str):
    """Deserialize a balancer object from JSON string

    Reconstructs a balancer object from its serialized JSON representation,
    restoring both its definition and state.

    Parameters
    ----------
    json_str : str
        JSON string containing the serialized balancer

    Returns
    -------
    BWD, BWDRandom, MultiBWD, or Online
        The deserialized balancer object with restored state
    """
    defs = json.loads(json_str)
    cls_name = list(defs.keys())[0]
    defs = defs[cls_name]

    bal_object = name2class[cls_name](**defs["definition"])
    bal_object.update_state(**defs["state"])
    return bal_object
