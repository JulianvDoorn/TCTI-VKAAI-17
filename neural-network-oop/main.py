import random
from numpy import tanh
from typing import List, Tuple, Dict

## Output or hidden neuron
class Neuron:
    @staticmethod
    def get_initial_weight():
        return random.random()

    index_cnt = 0

    def __init__(self, override_weight=None):
        Neuron.index_cnt += 1
        self.index = Neuron.index_cnt
        self.override_weight = override_weight
        self.forward_bound: List['Neuron'] = []
        self.backward_bound: Dict['Neuron', 'Weight'] = {}
        self.stored_activation = None
        def error_g(x):
            raise Exception("Error, no g and dev_g function is set")
        self.g = error_g
        self.dev_g = error_g

    def backward_bind(self, n, w):
        self.backward_bound[n] = w

    def set_g(self, g, dev_g):
        self.g = g
        self.dev_g = dev_g

    def reset(self):
        self.stored_activation = None

    def get_activation(self):
        if self.stored_activation is None:
            self.stored_activation = self.g(
                sum([n.get_activation() * w for n, w in self.backward_bound.items()])
            )

        return self.stored_activation

    def __repr__(self):
        return "Neuron " + str(self.index)

## Input neuron
class InputNeuron(Neuron):
    def __init__(self, override_weight=None):
        super(InputNeuron, self).__init__(override_weight)
        self.input = None

    def get_activation(self):
        return self.input

    def set_input(self, v):
        self.input = v

## Bias neuron
class BiasNeuron(InputNeuron):
    def __init__(self, override_weight=None):
        super(BiasNeuron, self).__init__(override_weight)
        self.input = -1

    ## Do nothing in order to maintain self.input == -1
    def set_input(self, v):
        pass


class Layer:
    def __init__(self):
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        return neuron

    def reset(self):
        for n in self.neurons:
            n.reset()

    def add_bias(self, bias):
        for n in self.neurons:
            n.backward_bind(
                bias, n.override_weight if n.override_weight is not None else Neuron.get_initial_weight())
        return bias

    def set_g(self, g, dev_g):
        for n in self.neurons:
            n.set_g(g, dev_g)

    def __iter__(self):
        for n in self.neurons:
            yield n

    def __repr__(self):
        return str(self.neurons)

    def __getitem__(self, i):
        return self.neurons[i]


class InputLayer(Layer):
    def __init__(self):
        super(InputLayer, self).__init__()

    def input(self, input_vector):
        for i, v in enumerate(input_vector):
            self.neurons[i].set_input(v)


class OutputLayer(Layer):
    def __init__(self):
        super(OutputLayer, self).__init__()

    def output(self):
        return (*[n.get_activation() for n in self.neurons],)


class Network:
    def __init__(self):
        self.layers = []
        self.hidden_layers = []
        self.input_layer = None
        self.output_layer = None

    def append_layer(self, layer):
        self.layers.append(layer)

        if len(self.layers) > 2:
            self.hidden_layers = self.layers[1:-1]
        else:
            self.hidden_layers = []

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        return layer

    def bind_axons(self):
        assert len(self.layers) > 1, "Layers not > 1"
        assert isinstance(self.input_layer,
                          InputLayer), "Layer[0] is not an InputLayer"
        assert isinstance(self.output_layer,
                          OutputLayer), "Layer[-1] is not an OutputLayer"

        for i, l in enumerate(self.layers[1:], 1):
            prev_l = self.layers[i - 1]

            for prev_n in prev_l:
                for n in l:
                    n.backward_bind(
                        prev_n,
                        n.override_weight if n.override_weight is not None else
                        Neuron.get_initial_weight()
                    )
                    
    def set_g(self, g, dev_g):
        for l in self.layers:
            l.set_g(g, dev_g)

    def input(self, input_vector):
        self.input_layer.input(input_vector)

    def output(self):
        tmp = self.output_layer.output()

        for l in self.layers:
            l.reset()

        return tmp

    # First propagate forward
    # then porpagate backward, calculate error per neuron.
    # Adjust weights with calculated errors.
    # Repeat for n times for every training_data input.
    def train(self, **kwargs):
        assert "iterations" in kwargs.keys(), "Missing kwarg: iterations"
        assert "training_data" in kwargs.keys(), "Missing kwarg: training_data"
        assert "stepsize" in kwargs.keys(), "Missing kwarg: stepsize"

        n = kwargs["iterations"]
        training_data = kwargs["training_data"]
        stepsize = kwargs["stepsize"]

        error_per_neuron: Dict[Neuron, float] = {}

        for input, desired_output in training_data.items():
            self.input(input)
            obtained_output = self.output()
            for i, v in enumerate(obtained_output):
                error_per_neuron[self.output_layer[i]] = desired_output[i] - v

        for i, l in enumerate([self.output_layer] + self.hidden_layers[::-1]):
            print(l.neurons)

        print(error_per_neuron)
        



def read_input():
    irisdtata = []
    inputdata = open("../data/iris.data", "r")
    for line in inputdata:
        if len(line) > 1:
            split = line.split(",")
            split[4] = split[4].strip()
            irisdtata.append(split)
    return irisdtata


def main():
    nn = Network()

    l1 = nn.append_layer(InputLayer())
    l2 = nn.append_layer(Layer())
    l3 = nn.append_layer(OutputLayer())

    n1 = l1.add_neuron(InputNeuron())
    n2 = l1.add_neuron(InputNeuron())
    n3 = l2.add_neuron(Neuron())
    n4 = l2.add_neuron(Neuron())
    b1 = l2.add_bias(BiasNeuron(0))
    n5 = l3.add_neuron(Neuron())
    n6 = l3.add_neuron(Neuron())
    b2 = l3.add_bias(BiasNeuron(0))

    nn.bind_axons()

    data = read_input()

    nn.set_g(lambda x: tanh(x), lambda x: 1 - tanh(x)**2)

    nn.train(
        # Training data in format of:
        # <input>: <expected-output>
        training_data={
            (0, 0): (0, 0),
            (0, 1): (0, 1),
            (1, 0): (1, 0),
            (1, 1): (1, 1),
        },
        iterations=100,
        stepsize=0.1
    )

    nn.input((1, 0))
    print(nn.output())
    nn.input((1, 1))
    print(nn.output())
    nn.input((0, 1))
    print(nn.output())
    nn.input((0, 0))
    print(nn.output())


if __name__ == "__main__":
    main()
