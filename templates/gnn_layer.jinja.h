//
// Autogenerated file!
//

#pragma once

#include "common/datatypes.h"
#include "configuration/options.h"
#include "configuration/config.h"
#include "data/graph.h"
#include "nn/initialization.h"
#include "gnn_layer.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop

struct {{LayerClassName}}Options : GNNLayerOptions {
    {% for name, option_type in options.items() %}
    {{option_type}} {{name}};
    {% endfor %}
};

class {{LayerClassName}} : public GNNLayer {
   public:
    shared_ptr<{{LayerClassName}}Options> options_;
    {% for name, var_type in member_vars.items() %}
    {{var_type}} {{name}};
    {% endfor %}

    {{LayerClassName}}(shared_ptr<LayerConfig> layer_config, torch::Device device);

    void reset() override;

    torch::Tensor forward(torch::Tensor inputs, DENSEGraph dense_graph, bool train = true) override;
    {% for fn in member_fns %}

    {{fn.returns}} {{fn.name}}({{fn.args|join(', ')}});
    {% endfor %}
};
