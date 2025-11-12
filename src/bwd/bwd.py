"""Balancing Walk Design with Restarts implementation."""

import numpy as np

from .exceptions import SampleSizeExpendedError

SERIALIZED_ATTRIBUTES = ["N", "D", "delta", "q", "intercept", "phi"]


class BWD(object):
    """**The Balancing Walk Design with Restarts**

    This is the primary suggested algorithm from [Arbour et al (2022)](https://arxiv.org/abs/2203.02025).
    At each step, it adjusts randomization probabilities to ensure that imbalance tends towards zero. In
    particular, if current imbalance is w and the current covariate profile is $x$, then the probability of
    treatment conditional on history will be:

    $$p_i = q \\left(1 - \\phi \\frac{x \\cdot w}{\\alpha}\\right)$$

    $q$ is the desired marginal probability, $\\phi$ is the parameter which controls robustness and
    $\\alpha$ is the normalizing constant which ensures the probability is well-formed.

    !!! important "If $|x \\cdot w| > \\alpha$"
        A restart is performed by resetting the algorithm:

        - $w$ is reset to the zero vector
        - $\\alpha$ is reset to a constant based on the number of units remaining in the sample
    """

    q: float
    intercept: bool
    delta: float
    N: int
    D: int
    value_plus: float
    value_minus: float
    phi: float
    alpha: float
    w_i: np.ndarray
    iterations: int

    def __init__(
        self,
        N: int,
        D: int,
        delta: float = 0.05,
        q: float = 0.5,
        intercept: bool = True,
        phi: float = 1,
    ) -> None:
        """
        Parameters
        ----------
        N : int
            Total number of points
        D : int
            Dimension of the data
        delta : float, optional
            Probability of failure, by default 0.05
        q : float, optional
            Target marginal probability of treatment, by default 0.5
        intercept : bool, optional
            Whether an intercept term be added to covariate profiles, by default True
        phi : float, optional
            Robustness parameter. A value of 1 focuses entirely on balance, while a value
            approaching zero does pure randomization, by default 1
        """
        self.q = q
        self.intercept = intercept
        self.delta = delta
        self.N = N
        self.D = D + int(self.intercept)
        self.value_plus = 2 * (1 - self.q)
        self.value_minus = -2 * self.q
        self.phi = phi
        self.reset()

    def set_alpha(self, N: int) -> None:
        """Set normalizing constant for remaining N units

        Parameters
        ----------
        N : int
            Number of units remaining in the sample

        Raises
        ------
        SampleSizeExpendedError
            If N is negative
        """
        if N < 0:
            raise SampleSizeExpendedError()
        self.alpha = np.log(2 * N / self.delta) * min(1 / self.q, 9.32)

    def assign_next(self, x: np.ndarray) -> int:
        """Assign treatment to the next point

        Parameters
        ----------
        x : np.ndarray
            Covariate profile of unit to assign treatment

        Returns
        -------
        int
            Treatment assignment (0 or 1)
        """
        if self.intercept:
            x = np.concatenate(([1], x))
        dot = x @ self.w_i
        if abs(dot) > self.alpha:
            self.w_i = np.zeros((self.D,))
            self.set_alpha(self.N - self.iterations)
            dot = x @ self.w_i

        p_i = self.q * (1 - self.phi * dot / self.alpha)

        if np.random.rand() < p_i:
            value = self.value_plus
            assignment = 1
        else:
            value = self.value_minus
            assignment = -1
        self.w_i += value * x
        self.iterations += 1
        return int((assignment + 1) / 2)

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
            "q": self.q,
            "intercept": self.intercept,
            "phi": self.phi,
        }

    @property
    def state(self):
        """Get the current state of the balancer

        Returns
        -------
        dict
            Dictionary containing w_i and iterations
        """
        return {"w_i": self.w_i, "iterations": self.iterations}

    def update_state(self, w_i, iterations):
        """Update the state of the balancer

        Parameters
        ----------
        w_i : array-like
            Current imbalance vector
        iterations : int
            Current iteration count
        """
        self.w_i = np.array(w_i)
        self.iterations = iterations

    def reset(self):
        """Reset the balancer to initial state

        Resets the imbalance vector to zeros, reinitializes alpha, and sets iterations to 0.
        """
        self.w_i = np.zeros((self.D,))
        self.set_alpha(self.N)
        self.iterations = 0
