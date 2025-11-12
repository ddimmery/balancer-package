"""
Module level documentation for the Balancing Walk Design.

This package is a truly minimal implementation in Python with
only a dependency on numpy. There are only two classes implemented
in this package:

- [`BWD`](bwd.md) - The Balancing Walk Design with restarts of the algorithm.
- [`BWDRandom`](bwd_random.md) - The Balancing Walk Design which reverts to simple randomization
rather than restarting.
"""

from .bwd import BWD as BWD
from .bwd_random import BWDRandom as BWDRandom
from .multi_bwd import MultiBWD as MultiBWD
from .online import Online as Online
