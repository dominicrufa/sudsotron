"""
Unit and regression test for the sudsotron package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import sudsotron


def test_sudsotron_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "sudsotron" in sys.modules
