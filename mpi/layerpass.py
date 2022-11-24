from mpi.types import ClassType, Attrs, Callable
from mpi.types import SynError, SemError
from mpi.builtins import NodeData, EdgeData, GraphLocalScope
from mpi.builtins import IntAttrs, StrAttrs, NoneAttrs, LayerAttrs, TensorAttrs, DENSEGraphAttrs
from mpi.builtins import is_numeric, is_consistent_with
from mpi.symtab import SymbolTable
from contextlib import contextmanager
from mpi import astpp  # noqa: F401

import ast
import logging

graph_node_data = '__{}_ndata'
graph_edge_data = '__{}_edata'


class TypeChecker(ast.NodeVisitor):
    """
    visit_className returns object attributes of the expression, None otherwise
    Runs semantic checks
    Registers new instance variables introduced in __init__ using type inference
    Ensures
    - functions are called with valid arguments
    - called symbols are callable
    - new instance variables introduced in __init__ are mapped
    - assignments are on consistent types
    - expressions are properly typed
    - operators are called with compatible types
    Disables
    - named expressions
    - nested functions or classes
    - registering new variables with non-enclosing scopes
    - registering new variables in non __init__ functions
    - generic subscripting
    Trampolines
    - constructors using __init__
    - callable objects using __call__
    """
    def __init__(self, symbol_table, classes):
        self.symbol_table = symbol_table
        self.classes = classes
        self.self_attrs = None
        self.init_mode = False
        self.return_attrs = None
        self.graph_local_scope = 0

    ##########
    # Literals
    ##########
    def visit_Constant(self, constant):
        if isinstance(constant.value, int):
            return IntAttrs
        elif isinstance(constant.value, str):
            return StrAttrs
        elif isinstance(constant.value, Ellipsis):
            return None
        else:
            raise SemError(f'Literal is not supported! lineno:{constant.lineno}')

    def visit_List(self, node):
        raise SynError(f'Lists are not supported! lineno:{node.lineno}')

    def visit_Tuple(self, node):
        raise SynError(f'Tuples are not supported! lineno:{node.lineno}')

    def visit_Set(self, node):
        raise SynError(f'Sets are not supported! lineno:{node.lineno}')

    def visit_Dict(self, node):
        raise SynError(f'Dicts are not supported! lineno:{node.lineno}')

    ###########
    # Variables
    ###########
    def visit_Name(self, name):
        return self.symbol_table.find_symbol(name.id)

    #############
    # Expressions
    #############
    def visit_Expr(self, expr):
        return self.visit(expr.value)

    def visit_UnaryOp(self, node):
        raise SynError(f'Unary operations are not supported! lineno:{node.lineno}')

    def visit_BinOp(self, bin_op):
        if not isinstance(bin_op.op, ast.Mult):
            raise SynError(f'Unexpected operation! lineno:{bin_op.lineno}')

        if not is_numeric(self.visit(bin_op.left)):
            raise SemError(f'Unexpected operand! lineno:{bin_op.lineno}')

        if not is_numeric(self.visit(bin_op.right)):
            raise SemError(f'Unexpected operand! lineno:{bin_op.lineno}')

        return IntAttrs

    def visit_BoolOp(self, node):
        raise SynError(f'bool ops are not supported! lineno:{node.lineno}')

    def visit_Compare(self, node):
        raise SynError(f'Compare ops not supported! lineno:{node.lineno}')

    def visit_Call(self, call):
        call_attrs = self.visit(call.func)
        fqname = ast.unparse(call.func)
        lno = call.lineno

        if not fqname.startswith('super') and fqname.endswith('__init__'):
            # XXX: __init__ returns None in python but class type in our model
            raise SemError(f'Avoid function calls ending with double underscore. lineno:{lno}')

        # Extract the type of callable
        if fqname == 'super':
            # TODO: semantic checking for call to super
            return LayerAttrs
        elif isinstance(call_attrs, ClassType):
            # Constructor
            class_type = call_attrs
            class_attrs = self.classes[class_type]
            func_sig = class_attrs['__init__']
        elif isinstance(call_attrs, Attrs):
            # callable object
            if '__call__' not in call_attrs:
                raise SemError(f'Expression {fqname} is not callable! lineno:{lno}')
            func_sig = call_attrs['__call__']
        else:
            func_sig = call_attrs

        if not isinstance(func_sig, Callable):
            raise SemError(f'Expression {fqname} is not callable! lineno:{lno}')

        # Verify function call arguments
        formal_args = [arg for arg in reversed(func_sig.args)]  # reversed to pop args from behind
        if len(formal_args) < len(call.args):
            raise SemError(f'Function called with invalid number of arguments! lineno:{lno}')
        for actual_arg in call.args:
            # Verify positional arguments
            actual_attrs = self.visit(actual_arg)
            if not is_consistent_with(actual_attrs, formal_args[-1].attrs):
                aname = ast.unparse(actual_arg)
                raise SemError(f'Function called with invalid arg {aname}! lineno:{lno}')

            formal_args.pop()

        # Expect the rest of the arguments to be called using kw args
        formal_args = {arg.name: arg.attrs for arg in formal_args}
        if len(formal_args) != len(call.keywords):
            raise SemError(f'Function called with invalid number of arguments! lineno:{lno}')
        for keyword in call.keywords:
            aname = keyword.arg
            # Verify keyword arguments
            if aname not in formal_args:
                raise SemError(f'Function called with unavailable kw arg {aname}! lineno:{lno}')
            actual_attrs = self.visit(keyword.value)
            if not is_consistent_with(actual_attrs, formal_args[aname].attrs):
                raise SemError(f'Function called with invalid arg {aname}! lineno:{lno}')
            del formal_args[aname]

        # Type of the call expression is the return type of the call expression
        return func_sig.return_attrs

    def visit_IfExp(self, node):
        raise SynError(f'if-else is not supported! lineno:{node.lineno}')

    def visit_Attribute(self, attr):
        value_attrs = self.visit(attr.value)
        if attr.attr in value_attrs:
            return value_attrs[attr.attr]

        return None

    ##############
    # Subscripting
    ##############
    def get_graph_local_name(self, subscript):
        if not self.graph_local_scope:
            raise SemError(f'Not in graph local scope! {subscript.lineno}')
        gdata = self.visit(subscript.value)
        if isinstance(gdata, NodeData):
            return graph_node_data.format(subscript.slice.value)
        elif isinstance(gdata, EdgeData):
            return graph_edge_data.format(subscript.slice.value)
        else:
            raise SemError(f'Generic subscripts are not supported! {subscript.lineno}')

    def visit_Subscript(self, subscript):
        if not isinstance(subscript.ctx, ast.Load):
            raise RuntimeError('Internal error!')

        return self.visit(self.get_graph_local_name(subscript))

    ############
    # Statements
    ############
    def visit_Assign(self, assign):
        """
        Implements core logic of automatically inferring new local and instance variables
        """
        lno = assign.lineno
        if len(assign.targets) != 1:
            raise SynError(f'Multi-assignments are not supported! {lno}')

        target = assign.targets[0]
        value_attrs = self.visit(assign.value)

        formal_attrs = self.visit(target)
        if formal_attrs is not None:
            # Definition already exists!
            if not is_consistent_with(value_attrs, formal_attrs):
                raise SemError(f'Type mismatch: cannot assign to variable! lineno:{lno}')
        elif isinstance(target, ast.Name):
            # register a new local variable
            self.symbol_table.add_symbol(target.id, value_attrs)
        elif isinstance(target, ast.Attribute):
            # Only allow self.<attr> = <value> to register new varaibles
            if ast.unparse(target.value) != 'self':
                raise SemError(f'Cannot add symbols to non-owning scopes! lineno:{lno}')
            if not self.init_mode:
                raise SemError(
                    f'Cannot register new instance variables outside of __init__! lineno:{lno}')
            self.self_attrs[target.attr] = value_attrs
        elif isinstance(target, ast.Subscript):
            # graph.ndata['h'] = ...
            self.symbol_table.add_symbol(self.get_graph_local_name(target), value_attrs)

    def visit_AnnAssign(self, assign):
        if self.return_attrs:
            raise SemError(
                f'Annotated assignments are not supported in func body! lineno:{assign.lineno}')

    def visit_Raise(self, node):
        raise SynError(f'Cannot raise exceptions! lineno:{node.lineno}')

    def visit_Pass(self, _):
        pass

    ##############
    # Control Flow
    ##############
    def visit_If(self, node):
        raise SemError(f'Control flow is not supported! lineno:{node.lineno}')

    ################################
    # Function and Class Definitions
    ################################
    def visit_FunctionDef(self, func):
        if self.return_attrs:
            raise SemError(f'Nested functions are not supported! lineno:{func.lineno}')

        func_sig = self.self_attrs[func.name]

        @contextmanager
        def managed_returns():
            self.return_attrs = func_sig.return_attrs
            yield
            self.return_attrs = None

        @contextmanager
        def managed_args():
            # add self
            self.symbol_table.add_symbol('self', self.self_attrs)
            # add other args
            for arg in func_sig.args:
                self.symbol_table.add_symbol(arg.name, arg.attrs)
            yield
            # XXX: symbols removed automatically through managed scope
            ...

        with managed_returns(), self.symbol_table.managed_scope(), managed_args():
            for child in func.body:
                self.visit(child)

    def visit_With(self, with_node):
        lno = with_node.lineno
        if len(with_node.items) != 1:
            raise SemError(f'General case `with` is not supported! lineno:{lno}')

        with_item = with_node.items[0]
        if with_item.optional_vars:
            raise SemError(f'General case `with` is not supported! lineno:{lno}')

        if self.visit(with_item.context_expr) != GraphLocalScope:
            raise SemError(f'General case `with` is not supported! lineno:{lno}')

        @contextmanager
        def managed_graph_local_scope(self):
            self.graph_local_scope += 1
            yield
            self.graph_local_scope -= 1

        with self.symbol_table.managed_graph_local_scope():
            self.visit(with_node.body)

    def visit_Return(self, value):
        if not is_consistent_with(self.visit(value), self.return_attrs):
            raise SemError(f'Return type does not match! lineno:{value.lineno}')

    def visit_ClassDef(self, classdef):
        lno = classdef.lineno

        if self.self_attrs or self.return_attrs:
            raise SemError(f'Nested classes are not supported! lineno:{lno}')

        self_attrs = self.classes[ClassType(classdef.name)]

        # verify the presence of mandatory functions
        if ('__init__' not in self_attrs
                or 'reset_parameters' not in self_attrs
                or 'forward' not in self_attrs):
            raise SemError(f'Must define __init__, reset_parameters, and forward! lineno:{lno}')
        init_func = self_attrs['__init__']
        reset_func = self_attrs['reset_parameters']
        forward_func = self_attrs['forward']

        # verify the signature of the reset_parameters function
        if (len(reset_func.args) != 0
                or not is_consistent_with(reset_func.return_attrs, NoneAttrs)):
            raise SemError(f'Invalid signature of reset_parameters! lineno:{lno}')

        # verify the signature of the forward function
        if (len(forward_func.args) != 2
                or not is_consistent_with(forward_func.args[0].attrs, DENSEGraphAttrs)
                or not is_consistent_with(forward_func.args[1].attrs, TensorAttrs)
                or not is_consistent_with(forward_func.return_attrs, TensorAttrs)):
            raise SemError(f'Invalid signature of forward! lineno:{lno}')

        @contextmanager
        def managed_self_attrs():
            self.self_attrs = self_attrs
            yield
            self.self_attrs = None

        @contextmanager
        def managed_init():
            self.init_mode = True
            yield
            self.init_mode = False

        with managed_self_attrs():
            # Pass1: process __init__
            with managed_init():
                for child in classdef.body:
                    if child.name == '__init__':
                        self.visit(child)

            # Pass2: process other nodes
            for child in classdef.body:
                if child.name == '__init__':
                    self.visit(child)

    def generic_visit(self, _):
        raise RuntimeError('Internal error!')


def run_layer_pass(symbol_table: SymbolTable, classes: dict[ClassType], classdef: ast.AST):
    logging.info(f'Running layer pass for {classdef.name} ...')
    TypeChecker(symbol_table, classes).visit(classdef)
