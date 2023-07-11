import numpy as np
import json

from .bwd import BWD
from .bwd_random import BWDRandom
from .multi_bwd import MultiBWD
from .online import Online

name2class = {
    "BWD": BWD,
    "BWDRandom": BWDRandom,
    "MultiBWD": MultiBWD,
    "Online": Online
}


def normalize(to_serialize):
    result = {}
    for k, v in to_serialize.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, dict):
            v = normalize(v)
        result[k] = v
    return result

def serialize(obj):
    return json.dumps({
        str(type(obj).__name__) : {"definition": normalize(obj.definition), "state": normalize(obj.state)}
    })


def deserialize(str):
    defs = json.loads(str)
    cls_name = list(defs.keys())[0]
    defs = defs[cls_name]
    
    object = name2class[cls_name](**defs["definition"])
    object.update_state(**defs["state"])
    return object
