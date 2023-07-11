import numpy as np

from .exceptions import SampleSizeExpendedError


class Online(object):
    def __init__(self, cls, **kwargs):
        N = 1
        self.cls = cls
        self.balancer = cls(N=N, **kwargs)
    
    def assign_next(self, x: np.ndarray) -> np.ndarray:
        try:
            return self.balancer.assign_next(x)
        except SampleSizeExpendedError:
            bal_def = self.balancer.definition
            bal_state = self.balancer.state
            bal_def["N"] = bal_def["N"] * 2
            self.balancer = self.cls(**bal_def)
            self.balancer.update_state(**bal_state)
            return self.balancer.assign_next(x)
            
    def assign_all(self, X: np.ndarray) -> np.ndarray:
        """Assign all points

        This assigns units to treatment in the offline setting in which all covariate
        profiles are available prior to assignment. The algorithm assigns as if units
        were still only observed in a stream.

        Args:
            X: array of size n × d of covariate profiles
        """
        if self.intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])

    @property
    def definition(self):
        return {"cls": self.cls, **self.balancer.definition}
    
    @property
    def state(self):
        return self.balancer.state
    
    def update_state(self, **kwargs):
        self.balancer.update_state(**kwargs)
