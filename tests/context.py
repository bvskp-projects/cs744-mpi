"""
Import main package into tests folder
See https://docs.python-guide.org/writing/structure/
"""

import os
import sys

# Import mpi after adding project root directory to the system path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_dir)
import mpi  # noqa: F401, E402
