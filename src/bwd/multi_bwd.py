"""Multi-treatment Balancing Walk Design implementation."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from .bwd import BWD


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

    def __init__(
        self,
        N: int,
        D: int,
        delta: float = 0.05,
        q: float | Iterable[float] = 0.5,
        intercept: bool = True,
        phi: float = 1.0,
    ):
        """
        Parameters
        ----------
        N : int
            Total number of points
        D : int
            Dimension of the data
        delta : float, optional
            Probability of failure, by default 0.05
        q : float | Iterable[float], optional
            Target marginal probability of treatment. Can be a single float for binary
            treatment or an iterable of probabilities for multiple treatments, by default 0.5
        intercept : bool, optional
            Whether an intercept term be added to covariate profiles, by default True
        phi : float, optional
            Robustness parameter. A value of 1 focuses entirely on balance, while a value
            approaching zero does pure randomization, by default 1.0
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

        # Use dictionaries for type-stable storage
        # nodes: dict mapping index -> BWD object (for internal nodes) or int (for leaf nodes)
        # weights: dict mapping index -> float
        self.nodes: dict[int, BWD | int] = {}
        self.weights: dict[int, float] = {}

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

        # Initialize leaf nodes with treatment assignments
        for leaf, trt in enumerate(trt_by_leaf):
            node = num_nodes - num_leaves + leaf
            self.nodes[node] = trt
            self.weights[node] = 1 / self.qs[trt] / num_leaves_by_trt[trt]

        # Build internal nodes from leaves up
        for cur_node in range(num_nodes)[::-1]:
            if cur_node == 0:
                break
            parent = _parent(cur_node)
            left = _left(parent)
            right = _right(parent)

            # Skip if children haven't been initialized yet
            if left not in self.nodes or right not in self.nodes:
                continue

            # If both children have the same treatment, propagate it up
            if self.nodes[left] == self.nodes[right]:
                self.nodes[parent] = self.nodes[left]
                self.weights[parent] = self.weights[left] + self.weights[right]
            # Otherwise, create a BWD balancer at this node
            else:
                left_weight = self.weights[left]
                right_weight = self.weights[right]
                pr_right = right_weight / (left_weight + right_weight)
                self.nodes[parent] = BWD(
                    N=N, D=D, intercept=intercept, delta=delta, q=pr_right, phi=phi
                )
                self.weights[parent] = left_weight + right_weight

    def assign_next(self, x: np.ndarray) -> int:
        """Assign treatment to the next point

        Parameters
        ----------
        x : np.ndarray
            Covariate profile of unit to assign treatment

        Returns
        -------
        int
            Treatment assignment (treatment group index)
        """
        cur_idx = 0
        while isinstance(self.nodes[cur_idx], BWD):
            assign = self.nodes[cur_idx].assign_next(x)
            cur_idx = _right(cur_idx) if assign > 0 else _left(cur_idx)
        # At this point, we've reached a leaf node which contains an int
        result = self.nodes[cur_idx]
        assert isinstance(result, int), "Leaf node must be an int"
        return result

    def assign_all(self, X: np.ndarray) -> np.ndarray:
        """Assign all points

        This assigns units to treatment in the offline setting in which all covariate
        profiles are available prior to assignment. The algorithm assigns as if units
        were still only observed in a stream.

        Parameters
        ----------
        X : np.ndarray
            Array of size n × d of covariate profiles

        Returns
        -------
        np.ndarray
            Array of treatment assignments
        """
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])

    @property
    def definition(self):
        """Get the definition parameters of the balancer

        Returns
        -------
        dict
            Dictionary containing N, D, delta, q, intercept, and phi
        """
        return {
            "N": self.N,
            "D": self.D,
            "delta": self.delta,
            "q": self.qs,
            "intercept": self.intercept,
            "phi": self.phi,
        }

    @property
    def state(self):
        """Get the current state of all BWD nodes in the tree

        Returns
        -------
        dict
            Dictionary mapping node indices to their states
        """
        return {
            idx: node.state for idx, node in self.nodes.items() if isinstance(node, BWD)
        }

    def update_state(self, **node_state_dict: Any) -> None:
        """Update the state of BWD nodes in the tree

        Parameters
        ----------
        **node_state_dict : dict
            Dictionary mapping node indices (as strings) to state dictionaries
        """
        for node, state in node_state_dict.items():
            node_obj = self.nodes[int(node)]
            if isinstance(node_obj, BWD):
                node_obj.update_state(**state)

    def reset(self):
        """Reset all BWD nodes in the tree to initial state"""
        for node in self.nodes.values():
            if isinstance(node, BWD):
                node.reset()
