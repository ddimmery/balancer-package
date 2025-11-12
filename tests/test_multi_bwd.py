import numpy as np
import pytest
from numpy.random import default_rng

from bwd import MultiBWD
from bwd.serialization import deserialize, serialize

rng = default_rng(84698384)  # ASCII for 'test' in decimal, concatenated

n = 100
d = 5

test_x = rng.normal(size=d)
test_X = rng.normal(size=(n, d))


@pytest.mark.order(0)
def test_instantiate():
    global balancer
    balancer = MultiBWD(N=n, D=d)


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
    balancer = MultiBWD(N=n, D=d)
    balancer.assign_next(test_x)
    balancer.assign_next(test_x)


@pytest.mark.order(2)
def test_assign_all():
    balancer = MultiBWD(N=n, D=d)
    balancer.assign_all(test_X)


@pytest.mark.order(3)
def test_instantiate_multi():
    global multi_balancer
    multi_balancer = MultiBWD(N=n, D=d, q=[1 / 4, 1 / 4, 1 / 2])


@pytest.mark.order(4)
def test_assign_one_multi():
    multi_balancer.assign_next(test_x)


@pytest.mark.order(5)
def test_assign_big_multi():
    multi_balancer = MultiBWD(N=n, D=d, q=[1 / 4, 1 / 4, 1 / 2])
    multi_balancer.assign_next(test_x)
    multi_balancer.assign_next(test_x)


def test_multi_bwd_reduces_imbalance():
    """Test that MultiBWD achieves lower pairwise imbalance for 3 treatment groups"""
    n_test = 1500
    d_test = 5
    n_runs = 5
    q_probs = [1 / 3, 1 / 3, 1 / 3]

    # Track pairwise imbalances for all three pairs: (0,1), (0,2), (1,2)
    multi_norms_pairs = {(0, 1): [], (0, 2): [], (1, 2): []}
    rand_norms_pairs = {(0, 1): [], (0, 2): [], (1, 2): []}

    for seed in range(n_runs):
        rng_test = default_rng(10000 + seed)
        X_test = rng_test.normal(size=(n_test, d_test))

        # MultiBWD assignments (3 groups)
        balancer_test = MultiBWD(N=n_test, D=d_test, q=q_probs)
        A_multi = []
        for x in X_test:
            a = balancer_test.assign_next(x)
            A_multi.append(a)

        # Pure randomization with 3 groups
        A_rand = rng_test.choice([0, 1, 2], size=n_test, p=q_probs)

        # Calculate pairwise imbalances for each pair of groups
        for g1, g2 in [(0, 1), (0, 2), (1, 2)]:
            # MultiBWD pairwise imbalance
            imbalance_multi = np.zeros(d_test)
            for i, x in enumerate(X_test):
                if A_multi[i] == g1:
                    imbalance_multi += x
                elif A_multi[i] == g2:
                    imbalance_multi -= x

            # Random pairwise imbalance
            imbalance_rand = np.zeros(d_test)
            for i, x in enumerate(X_test):
                if A_rand[i] == g1:
                    imbalance_rand += x
                elif A_rand[i] == g2:
                    imbalance_rand -= x

            multi_norms_pairs[(g1, g2)].append(np.linalg.norm(imbalance_multi))
            rand_norms_pairs[(g1, g2)].append(np.linalg.norm(imbalance_rand))

    # MultiBWD should have lower average pairwise imbalance for all pairs
    for g1, g2 in [(0, 1), (0, 2), (1, 2)]:
        avg_multi = np.mean(multi_norms_pairs[(g1, g2)])
        avg_rand = np.mean(rand_norms_pairs[(g1, g2)])
        assert avg_multi < avg_rand, (
            f"MultiBWD avg imbalance for groups ({g1},{g2}): {avg_multi:.2f} "
            f"should be less than random: {avg_rand:.2f}"
        )
