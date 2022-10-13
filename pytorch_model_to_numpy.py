from typing import Union, Sequence
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float64)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 2)
        )

    def forward(self, inputs):
        return self.net(inputs)


class NumpyLayer:
    def __init__(self, operator, *data):
        self.operator = operator
        self.data = data

    def forward(self, inputs):
        return self.operator(inputs, self.data)


class Torch2NP:
    @staticmethod
    def flatten_sequential(module):
        if not isinstance(module, nn.Sequential):
            return module
        layers = []
        for _, child in module.named_children():
            layers.append(Torch2NP.flatten_sequential(child))
        return layers

    @staticmethod
    def get_flattened_layers(model: nn.Module) -> Sequence[nn.Module]:
        layers = []
        for name, child in model.named_children():
            if isinstance(child, nn.Sequential):
                flattened = Torch2NP.flatten_sequential(child)
                layers.extend(flattened)
                for layer in flattened:
                    print(layer)
            else:
                layers.append(child)
                print(child)
        return layers

    @staticmethod
    def activation_layer(layer: Union[nn.ReLU, nn.Sigmoid, nn.Softmax]) -> NumpyLayer:
        def relu(inputs, data):
            return np.maximum(0, inputs)

        def sigmoid(inputs, data):
            return 1 / (1 + np.exp(-inputs))

        def softmax(x, data):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x) / np.sum(np.exp(x), axis=1)

        if isinstance(layer, nn.ReLU):
            func = relu
        elif isinstance(layer, nn.Sigmoid):
            func = sigmoid
        elif isinstance(layer, nn.Softmax):
            func = softmax
        else:
            raise NotImplementedError(f"Activation function {layer} is not supported for now")
        return NumpyLayer(func)

    @staticmethod
    def linear_layer(layer: nn.Linear) -> NumpyLayer:
        def linear(inputs, data):
            weight = data[0]
            bias = data[1]
            output = inputs @ weight.T + bias
            return output

        weight = layer.weight.data.numpy()
        bias = layer.bias.data.numpy()
        return NumpyLayer(linear, weight, bias)

    @staticmethod
    def torch2np(network: nn.Module) -> Sequence[NumpyLayer]:
        layers = Torch2NP.get_flattened_layers(network)
        np_layers = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                np_layers.append(Torch2NP.linear_layer(layer))
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Softmax):
                np_layers.append(Torch2NP.activation_layer(layer))
        return np_layers

    @staticmethod
    def np_forward(np_model: Sequence[NumpyLayer], inputs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        outputs = inputs.copy()
        if torch.is_tensor(outputs):
            outputs = outputs.numpy()
        assert isinstance(outputs, np.ndarray)
        for np_layer in np_model:
            outputs = np_layer.forward(outputs)
        return outputs


if __name__ == '__main__':
    net = NN()
    tool = Torch2NP()
    np_network = tool.torch2np(net)
    inputs = np.random.random(30).reshape([3, 10])
    inputs_torch = torch.tensor(inputs)
    print(net(inputs_torch))
    print(tool.np_forward(np_network, inputs))
