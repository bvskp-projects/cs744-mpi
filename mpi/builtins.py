from mpi.types import ClassType, Attrs, Arg, Callable

IntType = ClassType('int')
StrType = ClassType('str')
VoidType = ClassType('None')

IntAttrs = Attrs({
    '__class__': IntType
})
StrAttrs = Attrs({
    '__class__': StrType
})
NoneAttrs = Attrs({
    '__class__': VoidType
})


def is_numeric(value_attrs: Attrs) -> bool:
    return value_attrs == IntAttrs


def is_consistent_with(src_attrs: Attrs, dst_attrs: Attrs) -> bool:
    """
    Return True iff objects of type src can be converted to objects of type dst
    """
    return src_attrs == dst_attrs


MessageFunc = ClassType('__mpi_MessageFunc__')  # placeholder type for message_func
ReduceFunc = ClassType('__mpi_ReduceFunc__')  # placeholder type for reduce_func
GraphLocalScope = ClassType('__mpi_GraphLocalScope__')  # make graph updates local
NodeData = ClassType('__mpi_NodeData__')  # placeholder type for node data in graph
EdgeData = ClassType('__mpi_EdgeData__')  # placeholder type for node data in graph

Tensor = ClassType('__mpi_Tensor__')  # torch Tensor
Linear = ClassType('__mpi_Linear__')  # torch Linear
DENSEGraph = ClassType('__mpi_DENSEGraph__')  # Marius DENSEGraph
Layer = ClassType('__mpi_Module__')  # Marius base layer

LayerAttrs = Attrs({
    '__init__': Callable([Arg('input_dim', IntAttrs), Arg('output_dim', IntAttrs)], NoneAttrs),
    '__class__': Layer
})
TensorAttrs = Attrs({
    '__class__': Tensor
})
LinearAttrs = Attrs({
    '__init__': Callable([
        Arg('in_features', IntAttrs),
        Arg('out_features', IntAttrs),
    ], Linear),
    'reset_parameters': Callable([], None),
    '__call__': Callable([
        Arg('inputs', Tensor)
    ], Tensor),
    '__class__': Linear

})
DENSEGraphAttrs = Attrs({
    'update_all': Callable([
        ('message_func', MessageFunc),
        ('reduce_func', ReduceFunc)
    ], None),
    'local_scope': Callable([], GraphLocalScope),
    'ndata': NodeData,
    'edata': EdgeData,
    '__class__': DENSEGraph
})


def get_builtin_classes():
    # XXX: operator overriding is not supported!
    return {
        IntType: IntAttrs,
        StrType: StrAttrs,
        VoidType: NoneAttrs,
        Tensor: TensorAttrs,
        Linear: LinearAttrs,
        DENSEGraph: DENSEGraphAttrs,
        Layer: LayerAttrs
    }


builtin_attrs = Attrs({
    'int': IntType,
    'str': StrType
})
builtin_attrs['mpi'] = {
    # XXX: no inheritance supported => self is current type
    # XXX: builtin functions do not require `__locals`
    'Tensor': Tensor,
    'Linear': Linear,
    'DENSEGraph': DENSEGraph,
    'Module': Layer,
    'copy_u': Callable([
        Arg('u', str),
        Arg('out', str)
    ], MessageFunc),
    'mean': Callable([
        Arg('msg', str),
        Arg('out', str)
    ], ReduceFunc),
    'cat': Callable([
        Arg('tensors', list[Tensor]),
        Arg('dim', int)
    ], Tensor)
    # XXX: No mpi.Module since it is only meant to be used to define GNN modules
}
