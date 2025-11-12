"""Online balancer wrapper with automatic sample size expansion."""

import numpy as np

from .exceptions import SampleSizeExpendedError


class Online(object):
    """Online balancer wrapper with automatic sample size expansion

    This wrapper allows a balancer to operate in an online setting where the total
    sample size is not known in advance. When the sample size is exceeded, it
    automatically doubles the sample size while preserving the current state.
    """

    def __init__(self, cls, **kwargs):
        """
        Parameters
        ----------
        cls : class
            The balancer class to wrap (e.g., BWD, BWDRandom, MultiBWD)
        **kwargs : dict
            Keyword arguments to pass to the balancer class constructor.
            If N is not provided, it defaults to 1.
        """
        kwargs["N"] = kwargs.get("N", 1)
        self.cls = cls
        self.balancer = cls(**kwargs)

    def assign_next(self, x: np.ndarray) -> np.ndarray:
        """Assign treatment to the next point with automatic expansion

        If the sample size is exceeded, automatically doubles the sample size
        and continues from the current state.

        Parameters
        ----------
        x : np.ndarray
            Covariate profile of unit to assign treatment

        Returns
        -------
        np.ndarray
            Treatment assignment
        """
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
        """Get the definition parameters of the wrapped balancer

        Returns
        -------
        dict
            Dictionary containing the balancer class and all definition parameters
        """
        return {"cls": self.cls, **self.balancer.definition}

    @property
    def state(self):
        """Get the current state of the wrapped balancer

        Returns
        -------
        dict
            Dictionary containing the current state
        """
        return self.balancer.state

    def update_state(self, **kwargs):
        """Update the state of the wrapped balancer

        Parameters
        ----------
        **kwargs : dict
            State parameters to update
        """
        self.balancer.update_state(**kwargs)

    def reset(self):
        """Reset the wrapped balancer to initial state"""
        self.balancer.reset()
