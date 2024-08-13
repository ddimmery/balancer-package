import numpy as np
import pytest
from bwd import BWDRandom
from bwd.serialization import serialize, deserialize
from numpy.random import default_rng

rng = default_rng(84698384)  # ASCII for 'test' in decimal, concatenated

n = 100
d = 5

test_x = rng.normal(size=d)
test_X = rng.normal(size=(n, d))


@pytest.mark.order(0)
def test_instantiate():
    global balancer
    balancer = BWDRandom(N=n, D=d)


@pytest.mark.order(1)
def test_serialize():
    dump = serialize(balancer)
    assert isinstance(dump, str)


@pytest.mark.order(2)
def test_deserialize():
    dump = serialize(balancer)
    bal = deserialize(dump)
    assert isinstance(bal, BWDRandom)
    bal.assign_next(test_x)


def test_instantiate_no_args():
    with pytest.raises(TypeError):
        BWDRandom()


@pytest.mark.order(1)
def test_assign_one():
    balancer.assign_next(test_x)


@pytest.mark.order(3)
def test_assign_big():
    balancer = BWDRandom(N=n, D=d)
    balancer.assign_next(test_x)
    balancer.assign_next(test_x)


@pytest.mark.order(2)
def test_assign_all():
    balancer = BWDRandom(N=n, D=d)
    balancer.assign_all(test_X)
