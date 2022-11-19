"""
Import main package into tests folder
See https://docs.python-guide.org/writing/structure/
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mpi
