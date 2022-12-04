"""
Compile the input python source file into a cpp file
"""

from mpi.globalpass import run_global_pass

import ast
import logging


def run_compiler(filename: str):
    with open(filename, 'r') as modf:
        logging.info(f'Generating AST from file {filename} ...')
        tree = ast.parse(modf.read())
        run_global_pass(tree)
