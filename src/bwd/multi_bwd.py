from typing import Union
from collections.abc import Iterable
from .bwd import BWD

import numpy as np


def _left(i):
    return 2 * i + 1

def _right(i):
    return 2 * (i + 1)

def _parent(i):
    return int(np.floor((i - 1) / 2))


class MultiBWD(object):
    """**The Multi-treatment Balancing Walk Design with Restarts**

    This method implements an extension to the Balancing Walk Design to balance
    across multiple treatments. It accomplishes this by constructing a binary tree.
    At each node in the binary tree, it balanced between the treatment groups on the
    left and the right. Thus it ensures balance between any pair of treatment groups.
    """
    def __init__(self, N: int, D: int, delta: float=0.05, q: Union[float, Iterable]=0.5, intercept: bool=True, phi: float=1.0):
        """
        Args:
            N: total number of points
            D: dimension of the data
            delta: probability of failure
            q: Target marginal probability of treatment
            intercept: Whether an intercept term be added to covariate profiles
            phi: Robustness parameter. A value of 1 focuses entirely on balance, while a value
                approaching zero does pure randomization.
        """
        self.N = N
        self.D = D
        self.delta = delta
        self.intercept = intercept
        self.phi = phi

        if isinstance(q, float):
            q = q if q < 0.5 else 1 - q
            self.qs = [1 - q, q]
            self.classes = [0, 1]
        elif isinstance(q, Iterable):
            self.qs = [pr / sum(q) for pr in q]
            self.classes = [i for i, q in enumerate(self.qs)]
        num_groups = len(self.qs)
        self.K = num_groups - 1
        self.intercept = intercept
        
        num_levels = int(np.ceil(np.log2(num_groups)))
        num_leaves = int(np.power(2, num_levels))
        extra_leaves = num_leaves - num_groups
        num_nodes = int(np.power(2, num_levels + 1) - 1)
        self.nodes = [None] * num_nodes
        self.weights = [None] * num_nodes

        trt_by_leaf = []
        num_leaves_by_trt = []
        for trt in range(num_groups):
            if len(trt_by_leaf) % 2 == 0 and extra_leaves > 0:
                num_trt = 2 * (int(np.floor((extra_leaves - 1) / 2)) + 1)
                extra_leaves -= num_trt - 1
            else:
                num_trt = 1
            trt_by_leaf += [trt] * num_trt
            num_leaves_by_trt.append(num_trt)
        

        for leaf, trt in enumerate(trt_by_leaf):
            node = num_nodes - num_leaves + leaf
            self.nodes[node] = trt
            self.weights[node] = 1 / self.qs[trt] / num_leaves_by_trt[trt]


        for cur_node in range(num_nodes)[::-1]:
            if cur_node == 0:
                break
            parent = _parent(cur_node)
            left = _left(parent)
            right = _right(parent)
            if self.nodes[left] == self.nodes[right]:
                self.nodes[parent] = self.nodes[left]
                self.weights[parent] = self.weights[left] + self.weights[right]
            if self.nodes[left] is not None and self.nodes[right] is not None:
                left_weight = self.weights[_left(parent)]
                right_weight = self.weights[_right(parent)]
                pr_right = right_weight / (left_weight + right_weight)
                self.nodes[parent] = BWD(N=N, D=D, intercept=intercept, delta=delta, q=pr_right, phi = phi)
                self.weights[parent] = left_weight + right_weight

    def assign_next(self, x: np.ndarray) -> np.ndarray:
        """Assign treatment to the next point

        Args:
            x: covariate profile of unit to assign treatment
        """
        cur_idx = 0
        while isinstance(self.nodes[cur_idx], BWD):
            assign = self.nodes[cur_idx].assign_next(x)
            cur_idx = _right(cur_idx) if assign > 0 else _left(cur_idx)
        return self.nodes[cur_idx]

    def assign_all(self, X: np.ndarray) -> np.ndarray:
        """Assign all points

        This assigns units to treatment in the offline setting in which all covariate
        profiles are available prior to assignment. The algorithm assigns as if units
        were still only observed in a stream.

        Args:
            X: array of size n × d of covariate profiles
        """
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])
    
    @property
    def definition(self):
        return {
            "N": self.N, "D": self.D, "delta": self.delta, "q": self.qs,
            "intercept": self.intercept, "phi": self.phi
        }
    
    @property
    def state(self):
        return {idx: node.state for idx, node in enumerate(self.nodes) if isinstance(node, BWD)}
    
    def update_state(self, **node_state_dict):
        for node, state in node_state_dict.items():
            self.nodes[int(node)].update_state(**state)
