import numpy as np
import pytest
from balancer import BWD
from numpy.random import default_rng

rng = default_rng(84698384) # ASCII for 'test' in decimal, concatenated

n = 100
d = 5

test_x = rng.normal(size = d)
test_X = rng.normal(size = (n, d))

@pytest.mark.order(0)
def test_instantiate():
    global balancer
    balancer = BWD(N = n, D = d)

def test_instantiate_no_args():
    with pytest.raises(TypeError):
        BWD()

@pytest.mark.order(1)
def test_assign_one():
    test_x_with_intercept = np.concatenate([[1], test_x])
    balancer.assign_next(test_x_with_intercept)

@pytest.mark.order(3)
def test_assign_big():
    balancer = BWD(N = n, D = d)
    test_x_with_intercept = np.concatenate([[1], 100 * test_x])
    balancer.assign_next(test_x_with_intercept)
    balancer.assign_next(test_x_with_intercept)

@pytest.mark.order(2)
def test_assign_all():
    balancer = BWD(N = n, D = d)
    balancer.assign_all(test_X)



