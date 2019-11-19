from model.FM import *
from model.SGD_NFM import *
from model.ONN_NFM import *
from model.ONN_NFM_v4 import *

def _make_models(field_size,
                 feature_sizes,
                 max_num_hidden_layers,
                 qtd_neuron_per_hidden_layer,
                 dropout_shallow,
                 embedding_size,
                 batch_size,
                 verbose,
                 interaction_type,
                 eval_metric,
                 b,
                 n,
                 s,
                 use_cuda,
                 model_name):
    assert(isinstance(model_name, str))
    option_name = model_name
    # if model_name != 'FM':
    #     option_name += "_HiddenLayers" + str(max_num_hidden_layers)
    #     option_name += "_QtdNeuron" + str(qtd_neuron_per_hidden_layer)
    # option_name += "_EmbeddingDim" + str(embedding_size)
    # if model_name == 'ONN':
    #     option_name += '_BatchSize' + str(1)
    # else:
    #     option_name += '_BatchSize' + str(batch_size)
    # option_name += '_LearningRate' + str(n)

    if model_name == 'FM':
        model_instance = FM(field_size=field_size,
                            feature_sizes=feature_sizes,
                            embedding_size=embedding_size,
                            batch_size=batch_size,
                            n=n)
    elif model_name == 'SGD_NFM':
        model_instance = SGD_NFM(field_size=field_size,
                                 feature_sizes=feature_sizes,
                                 max_num_hidden_layers=max_num_hidden_layers,
                                 qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                                 embedding_size=embedding_size,
                                 batch_size=batch_size,
                                 n=n)
    elif model_name == 'ONN_NFM':
        model_instance = ONN_NFM(field_size=field_size,
                                 feature_sizes=feature_sizes,
                                 max_num_hidden_layers=max_num_hidden_layers,
                                 qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                                 embedding_size=embedding_size,
                                 n=n)
    else:
        model_instance = ONN_NFM_V4(field_size=field_size,
                                 feature_sizes=feature_sizes,
                                 max_num_hidden_layers=max_num_hidden_layers,
                                 qtd_neuron_per_hidden_layer=qtd_neuron_per_hidden_layer,
                                 embedding_size=embedding_size,
                                 n=n)
    return model_instance, option_name