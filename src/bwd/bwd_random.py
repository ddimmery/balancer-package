import numpy as np
import json


SERIALIZED_ATTRIBUTES = ["N", "D", "delta", "q", "intercept", "phi"]


class BWDRandom(object):
    """**The Balancing Walk Design with Reversion to Bernoulli Randomization**

    This is an algorithm from [Arbour et al (2022)](https://arxiv.org/abs/2203.02025).
    At each step, it adjusts randomization probabilities to ensure that imbalance tends towards zero. In
    particular, if current imbalance is w and the current covariate profile is $x$, then the probability of
    treatment conditional on history will be:

    $$p_i = q \\left(1 - \\phi \\frac{x \\cdot w}{\\alpha}\\right)$$

    $q$ is the desired marginal probability, $\\phi$ is the parameter which controls robustness and
    $\\alpha$ is the normalizing constant which ensures the probability is well-formed.

    !!! important "If $|x \\cdot w| > \\alpha$"
        All future units will be assigned by complete randomization.
    """
    def __init__(
        self, N: int, D: int, delta: float=0.05, q: float=0.5, intercept: bool=True, phi: float=1
    ) -> None:
        """Initialize the object

        Arguments:
            N: total number of points
            D: dimension of the data
            delta: probability of failure
            q: Target marginal probability of treatment
            intercept: Whether an intercept term be added to covariate profiles
            phi: Robustness parameter. A value of 1 focuses entirely on balance, while a value
                approaching zero does pure randomization.
        """
        self.q = q
        self.intercept = intercept
        self.delta = delta
        self.N = N
        self.D = D + int(self.intercept)
        self.w_i = np.zeros((self.D,))
        self.value_plus = 2 * (1 - self.q)
        self.value_minus = -2 * self.q
        self.phi = phi
        self.alpha = np.log(2 * self.N / self.delta) * min(1 / self.q, 9.32)
        self.iterations = 0

    def set_alpha(self, N: int) -> None:
        """Set normalizing constant for remaining N units

        Args:
            N: Number of units remaining in the sample
        """
        self.alpha = -1

    def assign_next(self, x: np.ndarray) -> np.ndarray:
        """Assign treatment to the next point

        Args:
            x: covariate profile of unit to assign treatment
        """
        if self.intercept:
            x = np.concatenate(([1], x))
        dot = x @ self.w_i
        if abs(dot) > self.alpha:
            self.w_i = np.zeros((self.D,))
            self.set_alpha(self.N - self.iterations)
            dot = 0.0

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

        Args:
            X: array of size n × d of covariate profiles
        """
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])

    @property
    def definition(self):
        return {
            "N": self.N, "D": self.D, "delta": self.delta, "q": self.q,
            "intercept": self.intercept, "phi": self.phi
        }
    
    @property
    def state(self):
        return {"w_i": self.w_i, "iterations": self.iterations}
    
    def update_state(self, w_i, iterations):
        self.w_i = np.array(w_i)
        self.iterations = iterations
