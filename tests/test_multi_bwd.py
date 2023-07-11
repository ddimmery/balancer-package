import numpy as np
import pytest
from bwd import MultiBWD
from bwd.serialization import serialize, deserialize
from numpy.random import default_rng

rng = default_rng(84698384) # ASCII for 'test' in decimal, concatenated

n = 100
d = 5

test_x = rng.normal(size = d)
test_X = rng.normal(size = (n, d))

@pytest.mark.order(0)
def test_instantiate():
    global balancer
    balancer = MultiBWD(N = n, D = d)

@pytest.mark.order(1)
def test_serialize():
    dump = serialize(balancer)
    assert isinstance(dump, str)

@pytest.mark.order(2)
def test_deserialize():
    dump = serialize(balancer)
    bal = deserialize(dump)
    assert isinstance(bal, MultiBWD)
    bal.assign_next(test_x)

def test_instantiate_no_args():
    with pytest.raises(TypeError):
        MultiBWD()

@pytest.mark.order(1)
def test_assign_one():
    balancer.assign_next(test_x)

@pytest.mark.order(5)
def test_assign_big():
    balancer = MultiBWD(N = n, D = d)
    balancer.assign_next(test_x)
    balancer.assign_next(test_x)

@pytest.mark.order(2)
def test_assign_all():
    balancer = MultiBWD(N = n, D = d)
    balancer.assign_all(test_X)

@pytest.mark.order(3)
def test_instantiate_multi():
    global multi_balancer
    multi_balancer = MultiBWD(N = n, D = d, q = [1/4, 1/4, 1/2])

@pytest.mark.order(4)
def test_assign_one_multi():
    multi_balancer.assign_next(test_x)

@pytest.mark.order(5)
def test_assign_big_multi():
    multi_balancer = MultiBWD(N = n, D = d, q = [1/4, 1/4, 1/2])
    multi_balancer.assign_next(test_x)
    multi_balancer.assign_next(test_x)



