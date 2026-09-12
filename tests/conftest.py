"""
Make the project root importable for the test suite.

The directory that holds the `scripts` package is the PARENT of `tests/`, so
that is what belongs on `sys.path` -- not its parent's parent.

This matters for more than tidiness. The mutation gauntlet copies the project
into `mutants/` and runs pytest from there, so `mutants/tests/..` must resolve
to `mutants/`, putting the MUTATED `scripts` package ahead of the real one.
The previous expression walked up one level too far: from `mutants/tests/` it
resolved to the real repository root, so every test imported the unmutated
module and mutmut reported that no test covered any mutant.
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
