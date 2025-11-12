import numpy as np
import pytest
from numpy.random import default_rng

from bwd import BWDRandom
from bwd.serialization import deserialize, serialize

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


def test_bwd_random_reduces_imbalance():
    """Test that BWDRandom achieves lower imbalance than pure randomization on average"""
    # BWDRandom reverts to randomization when imbalance is too large,
    # so we test that it performs better on average across multiple runs
    n_test = 1000
    d_test = 5
    n_runs = 5

    bwd_norms = []
    rand_norms = []

    for seed in range(n_runs):
        rng_test = default_rng(10000 + seed)
        X_test = rng_test.normal(size=(n_test, d_test))

        # BWDRandom assignments
        balancer_test = BWDRandom(N=n_test, D=d_test)
        imbalance_bwd = np.zeros(d_test)

        for x in X_test:
            a = balancer_test.assign_next(x)
            imbalance_bwd += (2 * a - 1) * x

        # Pure randomization
        A_rand = rng_test.binomial(n=1, p=0.5, size=n_test)
        imbalance_rand = np.zeros(d_test)
        for i, x in enumerate(X_test):
            imbalance_rand += (2 * A_rand[i] - 1) * x

        bwd_norms.append(np.linalg.norm(imbalance_bwd))
        rand_norms.append(np.linalg.norm(imbalance_rand))

    # BWDRandom should have lower average imbalance
    avg_bwd = np.mean(bwd_norms)
    avg_rand = np.mean(rand_norms)

    assert avg_bwd < avg_rand, (
        f"BWDRandom avg imbalance ({avg_bwd:.2f}) should be less than random ({avg_rand:.2f})"
    )
