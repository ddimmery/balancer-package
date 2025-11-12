"""Custom exceptions for the BWD package."""


class SampleSizeExpendedError(Exception):
    """Exception raised when the sample size has been exceeded

    This exception is raised when attempting to assign treatment to more units
    than the initially specified sample size N. It is typically caught by the
    Online wrapper to automatically expand the sample size.
    """
