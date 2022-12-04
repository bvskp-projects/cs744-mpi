import os
import unittest
from pathlib import Path  # noqa: F401
from test.context import mpi  # noqa: F401
from test.context import project_dir

from mpi.compiler import run_compiler

test_dir = os.path.join(project_dir, "test")
examples_dir = os.path.join(test_dir, "examples")
codegen_dir = os.path.join(os.path.join(project_dir, "build"), "gen")


def run_codegen_test(filename):
    run_compiler(os.path.join(examples_dir, filename))
    return os.path.exists(
        os.path.join(codegen_dir, Path(filename).with_suffix(".h"))
    ) and os.path.exists(os.path.join(codegen_dir, Path(filename).with_suffix(".cpp")))


class TestCodeGen(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(run_codegen_test("basic_layer.py"))
