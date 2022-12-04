from mpi.types import ClassType, Attrs, Arg, Callable
from mpi.types import SynError, SemError
from mpi.builtins import get_builtin_classes, NoneAttrs
from mpi.symtab import SymbolTable
from mpi.layerpass import run_layer_pass
from contextlib import contextmanager
from mpi import astpp  # noqa: F401

import ast
import logging


class FeatureFilter(ast.NodeVisitor):
    """
    The compiler does not support the complete python grammar
    Detect and filter syntax errors before semantic analysis

    Disables
    - comprehensions
    - print
    - assert
    - lambdas
    - generators
    - coroutines
    - interactive and eval modes
    - starred variables
    - formatting strings
    """
    ##########
    # Literals
    ##########
    def visit_FormattedValue(self, node):
        raise SynError(f'Format spec is not supported! lineno:{node.lineno}')

    def visit_JoinedStr(self, node):
        raise SynError(f'Python style formatting is not supported! lineno:{node.lineno}')

    ###########
    # Variables
    ###########
    def visit_Starred(self, node):
        raise SynError(f'Starred variables are not supported! lineno:{node.lineno}')

    #############
    # Expressions
    #############
    def visit_NamedExpr(self, node):
        raise SynError(f'Named expressions are not supported! lineno:{node.lineno}')

    ##############
    # Subscripting
    ##############
    def visit_Index(self, index):
        if not isinstance(index.value, str):
            raise SynError(f'Only string indices are supported! lineno:{index.lineno}')

    def visit_Slice(self, index):
        raise SynError(f'Index slices are not supported! lineno:{index.lineno}')

    def visit_ExtSlice(self, extslice):
        raise SynError(f'Ext slices are not supported! lineno:{extslice.lineno}')

    ################
    # Comprehensions
    ################
    def visit_ListComp(self, node):
        raise SynError(f'List comprehensions are not supported! lineno:{node.lineno}')

    def visit_SetComp(self, node):
        raise SynError(f'Set comprehensions are not supported! lineno:{node.lineno}')

    def visit_GeneratorExp(self, node):
        raise SynError(f'Generator expresions are not supported! lineno:{node.lineno}')

    def visit_DictComp(self, node):
        raise SynError(f'Dict comprehensions are not supported! lineno:{node.lineno}')

    ############
    # Statements
    ############
    def AugAssign(self, node):
        raise SynError(f'Augmented Assign is not supported! lineno:{node.lineno}')

    def visit_Print(self, node):
        raise SynError(f'Print is not supported! lineno:{node.lineno}')

    def visit_Assert(self, node):
        raise SynError(f'Assert statements are not supported! lineno:{node.lineno}')

    def visit_Delete(self, node):
        raise SynError(f'Delete statements are not supported! lineno:{node.lineno}')

    #########
    # Imports
    #########
    def visit_Import(self, node):
        raise SynError(f'Imports are not supported! lineno:{node.lineno}')

    def visit_ImportFrom(self, node):
        raise SynError(f'Imports are not supported! lineno:{node.lineno}')

    ##############
    # Control Flow
    ##############
    def visit_For(self, node):
        raise SynError(f'Loops are not supported! lineno:{node.lineno}')

    def visit_While(self, node):
        raise SynError(f'Loops are not supported! lineno:{node.lineno}')

    def visit_Break(self, node):
        raise SynError(f'Loops are not supported! lineno:{node.lineno}')

    def visit_Continue(self, node):
        raise SynError(f'Loops are not supported! lineno:{node.lineno}')

    def visit_Try(self, node):
        raise SynError(f'Exception handling is not supported! lineno:{node.lineno}')

    def visit_Finally(self, node):
        raise SynError(f'Exception handling is not supported! lineno:{node.lineno}')

    def visit_Except(self, node):
        raise SynError(f'Exception handling is not supported! lineno:{node.lineno}')

    ################################
    # Function and Class Definitions
    ################################
    def visit_Lambda(self, node):
        raise SynError(f'No lambda definitions allowed! lineno:{node.lineno}')

    def visit_Yield(self, node):
        raise SynError(f'Generators not supported! lineno:{node.lineno}')

    def visit_YieldFrom(self, node):
        raise SynError(f'Generators not supported! lineno:{node.lineno}')

    def Global(self, node):
        raise SynError(f'Global not supported! lineno:{node.lineno}')

    def NonLocal(self, node):
        raise SynError(f'NonLocal not supported! lineno:{node.lineno}')

    #################
    # Async and Await
    #################
    def visit_AsyncFunctionDef(self, node):
        raise SynError(f'Coroutines not supported! lineno:{node.lineno}')

    def visit_Await(self, node):
        raise SynError(f'Coroutines not supported! lineno:{node.lineno}')

    def visit_AsyncFor(self, node):
        raise SynError(f'Coroutines not supported! lineno:{node.lineno}')

    def visit_AsyncWith(self, node):
        raise SynError(f'Coroutines not supported! lineno:{node.lineno}')


class GlobalAttrsPass(ast.NodeVisitor):
    """
    Collect all class names in global scope
    Ensures
    - classes at the top level
    """
    def __init__(self):
        self.global_attrs = Attrs()
        self.classes = get_builtin_classes()

    def visit_ClassDef(self, classdef):
        if classdef.name in self.global_attrs:
            raise SemError(f'Class {classdef.name} is multiply defined! lineno:{classdef.lineno}')

        class_type = ClassType(classdef.name)
        self.global_attrs[classdef.name] = class_type
        self.classes[class_type] = Attrs({'__class__': class_type})

    def visit_Module(self, module):
        for child in module.body:
            if isinstance(child, ast.ClassDef):
                self.visit(child)
            elif not isinstance(child, ast.Expr):
                # Allow comments
                raise SemError(
                    f'Only mpi.Module classes allowed at the top level! lineno:{child.lineno}')

    def generic_visit(self, node):
        raise RuntimeError('Internal error!')


class InstanceAttrsPass(ast.NodeVisitor):
    """
    Generate a mapping from class names to the class attributes
    The attribute map has instance variables and instance methods
    Class variables and class methods are disallowed at the moment
    Instance variables map to its attribute map
    Instance methods map to its `Callable` object
    Expects a symbol table populated with symbols in the builtin and global scopes
    """
    def __init__(self, symbol_table, classes):
        self.symbol_table = symbol_table
        self.classes = classes
        self.instance_attrs = None

    def visit_Name(self, name):
        attrs = self.symbol_table.find_symbol(name.id)
        if attrs is None:
            raise SemError(f'Could not find symbol {name.id}! lineno:{name.lineno}')
        return attrs

    def visit_Attribute(self, attr):
        value_attrs = self.visit(attr.value)
        if attr.attr not in value_attrs:
            raise SemError(f'Could not find symbol {attr.attr}! lineno:{attr.lineno}')
        return value_attrs[attr.attr]

    def visit_AnnAssign(self, assign):
        """
        Collect instance variables
        Ensures
        - valid type annotations
        - unique names
        """
        if not isinstance(assign.target, ast.Name):
            raise SynError(f'Definition is not supported! lineno:{assign.lineno}')

        if not assign.annotation:
            raise SynError(
                f'Require type annotation for instance variables! lineno:{assign.lineno}')

        var_name = assign.target.id
        var_attrs = self.classes[self.visit(assign.annotation)]

        if var_name in self.instance_attrs:
            raise SemError(f'Duplicate attr definition! lineno:{assign.lineno}')

        self.instance_attrs[var_name] = var_attrs

    def visit_FunctionDef(self, func):
        """
        Collects instance functions
        Ensures
        - unique names
        - valid type annotations for all arguments and return
        - self first argument
        Disables
        - operator overriding
        - decorator lists
        - type comments
        - position only arguments
        - *args and **kwargs
        """
        if func.name in self.instance_attrs:
            raise SemError(f'Duplicate attr definition! func: lineno:{func.lineno}')

        if (func.name != '__init__'
                and func.name != '__call__'
                and func.name.startswith('__')
                and func.name.endswith('__')):
            raise SynError(f'Operator overriding is not supported! lineno:{func.lineno}')

        if func.decorator_list:
            raise SynError(f'Decorator lists are not supported! lineno:{func.lineno}')

        if func.type_comment:
            raise SynError(f'Type comments are not supported! lineno:{func.lineno}')

        if func.args.vararg:
            raise SynError(f'*args is not supported! lineno:{func.lineno}')

        if func.args.kwarg:
            raise SynError(f'**kwargs is not supported! func: lineno:{func.lineno}')

        if func.args.posonlyargs or func.args.kwonlyargs:
            raise SynError(f'pos only or kw only args are not supported! lineno:{func.lineno}')

        if func.args.kw_defaults or func.args.defaults:
            raise SynError(f'Defaults for arguments are not supported! lineno:{func.lineno}')

        if func.args.args[0].arg != 'self':
            raise SynError(f'Only instance methods are supported! lineno:{func.lineno}')

        args = []
        for arg in func.args.args[1:]:
            if not arg.annotation:
                raise SynError(f'Type annotations are required! lineno:{func.lineno}')
            arg_attrs = self.classes[self.visit(arg.annotation)]
            args.append(Arg(arg.arg, arg_attrs))

        if func.returns:
            return_attrs = self.classes[self.visit(func.returns)]
        else:
            return_attrs = NoneAttrs
        self.instance_attrs[func.name] = Callable(args, return_attrs)

    def visit_ClassDef(self, classdef):
        """
        Populates class attributes
        """
        if classdef.decorator_list:
            raise SemError(f'Decorator lists are not supported! lineno:{classdef.lineno}')

        if classdef.keywords:
            raise SemError(f'Class arguments are not supported! lineno:{classdef.lineno}')

        @contextmanager
        def managed_instance_attrs():
            self.instance_attrs = self.classes[ClassType(classdef.name)]
            yield
            self.instance_attrs = None

        with managed_instance_attrs():
            for child in classdef.body:
                self.visit(child)

    def visit_Module(self, module):
        for child in module.body:
            if isinstance(child, ast.ClassDef):
                self.visit(child)
            elif not isinstance(child, ast.Expr):
                # Allow comments
                raise SemError(
                    f'Only mpi.Module classes allowed at the top level! lineno:{child.lineno}')

    def generic_visit(self, node):
        raise RuntimeError('Internal error!')


class LayerExtractorPass(ast.NodeVisitor):
    """
    Run semantic checks for each GNN layer module
    Expects both a symbol table and class instance attributes
    Ensures
    - class inherits from mpi.Module and nothing else
    """
    def __init__(self, symbol_table, classes):
        self.symbol_table = symbol_table
        self.classes = classes

    def visit_ClassDef(self, classdef):
        # TODO: Support inheritance
        if (len(classdef.bases) != 1
                or ast.unparse(classdef.bases[0]) != 'mpi.Module'):
            raise SemError(f'All classes must inherit from mpi.Module! lineno:{classdef.lineno}')

        run_layer_pass(self.symbol_table, self.classes, classdef)

    def visit_Module(self, module):
        # XXX: cleaner to infer member variables in a separate pass
        for child in module.body:
            self.visit(child)


def run_global_pass(tree: ast.AST):
    logging.info('Running global pass ...')

    # Pass1: Detect and disable "fancy" features
    FeatureFilter().visit(tree)

    # Pass2: Global attribute pass to collect all classes
    global_attrs_pass = GlobalAttrsPass()
    global_attrs_pass.visit(tree)
    global_attrs = global_attrs_pass.global_attrs
    symbol_table = SymbolTable(global_attrs)
    classes = global_attrs_pass.classes

    # Pass3: Instance attrs pass to generate object schemas
    instance_attrs_pass = InstanceAttrsPass(symbol_table, classes)
    instance_attrs_pass.visit(tree)

    # Pass4: Type check each class
    LayerExtractorPass(symbol_table, classes).visit(tree)
