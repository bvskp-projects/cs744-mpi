import os
import unittest
from test.context import mpi  # noqa: F401
from test.context import project_dir

from mpi.compiler import run_compiler
from mpi.types import CompileError

test_dir = os.path.join(project_dir, "test")
error_dir = os.path.join(test_dir, "errors")


def run_error_test(filename):
    run_compiler(os.path.join(error_dir, filename))


class TestModuleErrors(unittest.TestCase):
    def test_disable_imports(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_imports.py")

    def test_disable_globals(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_globals.py")

    def test_disable_non_layer_classes(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_non_layer_classes.py")

    def test_duplicate_classes(self):
        with self.assertRaises(CompileError):
            run_error_test("duplicate_class.py")


class TestClassErrors(unittest.TestCase):
    def test_duplciate_fn(self):
        with self.assertRaises(CompileError):
            run_error_test("duplicate_fn.py")


class TestFunctionErrors(unittest.TestCase):
    def test_disable_nesting(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_nesting.py")

    def test_require_argtypes(self):
        with self.assertRaises(CompileError):
            run_error_test("require_argtypes.py")

    def test_invalid_return(self):
        with self.assertRaises(CompileError):
            run_error_test("invalid_return.py")

    def test_disable_lambdas(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_lambdas.py")

    def test_disable_generators(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_generators.py")


class TestExpressionErrors(unittest.TestCase):
    def test_multiassign(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_multiassign.py")

    def test_call_invalid_args(self):
        with self.assertRaises(CompileError):
            run_error_test("call_invalid_args.py")


class TestStatementErrors(unittest.TestCase):
    def test_disable_format(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_format.py")

    def test_disable_print(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_print.py")

    def test_disable_assert(self):
        with self.assertRaises(CompileError):
            run_error_test("disable_assert.py")
