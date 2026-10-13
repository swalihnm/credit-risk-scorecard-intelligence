"""Pytest loads conftest.py before collecting any test module. optbinning (via its ortools/
CP-SAT dependency) must be imported before pandas in this process, or it segfaults at import
time -- a native DLL-loading-order conflict on this Windows/Python 3.12 setup, not a bug in
either library (see scorecard/binning.py for the full explanation). Since pytest collects
ALL test files -- several of which import pandas at module level -- before running any test,
fixing import order within an individual test file isn't enough: optbinning has to be the
first of the two imported anywhere in the process, which this file guarantees.
"""
from optbinning import OptimalBinning  # noqa: F401
