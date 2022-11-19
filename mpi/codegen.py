"""
Generate cpp code from the IR
"""

from jinja2 import Environment, FileSystemLoader

import os
import logging

class CodeRenderer:
    def __init__(self):
        output_dir = 'build/gen'

        # TODO(saikiran): Change to package loader
        self.env = Environment(loader=FileSystemLoader('templates'),
                               keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True,
                               line_comment_prefix='//*')
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def render_header(self, params: dict, output_file: str):
        self.render(params, 'gnn_layer.jinja.h', output_file)

    def render_source(self, params: str, output_file: str):
        self.render(params, 'gnn_layer.jinja.cpp', output_file)

    def render(self, params: dict, template: str, output_file: str):
        output_file = os.path.join(self.output_dir, output_file)
        with open(output_file, 'w') as outfile:
            output = self.env.get_template(template).render(**params)
            outfile.write(output)
