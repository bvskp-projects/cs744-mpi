from mpi.codegen import CodeRenderer

if __name__ == '__main__':
    renderer = CodeRenderer()
    renderer.render_header({}, 'graph_sage_layer.h')
    renderer.render_source({}, 'graph_sage_layer.cpp')
