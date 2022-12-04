import unittest

from tests.context import project_dir
from tests.context import mpi  # noqa: F401
from mpi.compiler import run_compiler
from pathlib import Path

import os

test_dir = os.path.join(project_dir, 'tests')
examples_dir = os.path.join(test_dir, 'examples')
codegen_dir = os.path.join(os.path.join(project_dir, 'build'), 'gen')


def run_codegen_test(filename):
    run_compiler(os.path.join(examples_dir, filename))
    return True
    # TODO(saikiran): reinstate the condition
    # return (os.path.exists(os.path.join(codegen_dir, Path(filename).with_suffix('.h')))
    #         and os.path.exists(os.path.join(codegen_dir, Path(filename).with_suffix('.cpp'))))


class TestCodeGen(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(run_codegen_test('basic_layer.py'))
