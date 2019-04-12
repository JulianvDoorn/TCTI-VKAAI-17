import random
from numpy import tanh
from typing import List, Tuple, Dict
import csv

# Output or hidden neuron


class Neuron:
    @staticmethod
    def get_initial_weight():
        return random.random()

    index_cnt = 0

    def __init__(self):
        Neuron.index_cnt += 1
        self.index = Neuron.index_cnt
        self.forward_bound: List['Neuron'] = []
        self.backward_bound: Dict['Neuron', 'Weight'] = {}
        self.stored_activation = None
        self.stored_error = None

        def error_g(x):
            raise Exception("Error, no g and der_g function is set")

        self.g = error_g
        self.der_g = error_g

    def backward_bind(self, n, w):
        self.backward_bound[n] = w

    def get_weight_to_neuron(self, n):
        # Only backward bound neurons' weights are stored in backward_bound[n]
        # Attempting to reference a forward bound neuron will raise an exception
        return self.backward_bound[n]

    def forward_bind(self, n):
        self.forward_bound.append(n)

    def set_g(self, g, der_g):
        self.g = g
        self.der_g = der_g

    def reset(self):
        self.stored_activation = None
        self.stored_error = None

    def get_input(self):
        return sum([
            n.get_activation() * w for n, w in self.backward_bound.items()
        ])

    def get_activation(self):
        if self.stored_activation is None:
            self.stored_activation = self.g(self.get_input())

        return self.stored_activation

    def get_error(self):
        if self.stored_error is None:
            self.stored_error = self.der_g(self.get_input()) * sum([
                n.get_weight_to_neuron(self) * n.get_error() for n in self.forward_bound
            ])

        return self.stored_error

    def __repr__(self):
        return "Neuron " + str(self.index)

# Input neuron


class InputNeuron(Neuron):
    def __init__(self):
        super(InputNeuron, self).__init__()
        self.input = None

    def get_input(self):
        return self.input

    def get_activation(self):
        return self.input

    def set_input(self, v):
        self.input = v

    def get_error(self):
        # Invoke forward bound neurons for their errors
        for n in self.forward_bound:
            n.get_error()

        # Input neurons cannot have an error
        return 0


class OutputNeuron(Neuron):
    def __init__(self):
        super(OutputNeuron, self).__init__()
        self.desired_output = None

    def get_error(self):
        if self.stored_error is None:
            self.stored_error = self.der_g(
                self.get_input()) * (self.desired_output - self.get_activation())

        return self.stored_error

    def set_desired(self, d):
        self.desired_output = d

# Bias neuron


class BiasNeuron(InputNeuron):
    def __init__(self):
        super(BiasNeuron, self).__init__()
        self.input = -1

    # Do nothing in order to maintain self.input == -1
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

    def set_g(self, g, der_g):
        for n in self.neurons:
            n.set_g(g, der_g)

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
                        Neuron.get_initial_weight()
                    )
                    prev_n.forward_bind(n)

    def set_g(self, g, der_g):
        for l in self.layers:
            l.set_g(g, der_g)

    def input(self, input_vector):
        self.input_layer.input(input_vector)

    def output(self):
        for l in self.layers:
            l.reset()
        tmp = self.output_layer.output()
        return tmp

    def calc(self, *args):
        self.input((args))
        return self.output()

    # First propagate forward
    # then porpagate backward, calculate error per neuron.
    # Adjust weights with calculated errors.
    # Repeat for n times for every training_data input.
    def train(self, **kwargs):
        assert "iterations" in kwargs.keys(), "Missing kwarg: iterations"
        assert "training_data" in kwargs.keys(), "Missing kwarg: training_data"
        assert "stepsize" in kwargs.keys(), "Missing kwarg: stepsize"

        iterations = kwargs["iterations"]
        training_data = kwargs["training_data"]
        stepsize = kwargs["stepsize"]

        error_per_neuron: Dict[Neuron, float] = {}

        for _ in range(iterations):
            for input, desired_output in training_data.items():
                self.input(input)
                obtained_output = self.output()

                for i, _ in enumerate(obtained_output):
                    self.output_layer[i].set_desired(desired_output[i])

                for l in self.layers:
                    for n in l:
                        error_per_neuron[n] = n.get_error()

                # Adjust error_per_neuron
                # But the input layer holds no weights, skip it
                for l in self.layers[1:]:
                    for n in l:
                        for prev_n, w in n.backward_bound.items():
                            # print("Processing:", prev_n, "->", n)
                            n.backward_bound[prev_n] = w + stepsize * \
                                prev_n.get_activation() * n.get_error()

                # Repeat until all neurons have been backpropagated
                # print("Error per neuron:", error_per_neuron)


def read_input(file):
    labels = []
    csv_data = []

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        for r in reader:
            if r:
                if r[4] not in labels:
                    labels.append(r[4])
                r[4] = labels.index(r[4])
                csv_data.append([float(f) for f in r[0:4]] + [r[4]])

    return csv_data, labels

def expected_output_for_row(row):
    if row[4] == 0:
        return (1, 0, 0)
    if row[4] == 1:
        return (0, 1, 0)
    if row[4] == 2:
        return (0, 0, 1)

def get_label_from_output(val, labels):
    if val == (1, 0, 0):
        return labels[0]
    if val == (0, 1, 0):
        return labels[1]
    if val == (0, 0, 1):
        return labels[2]

def generate_training_data(rows):
    training_data = {}

    for r in rows:
        training_data[(*r[0:4],)] = expected_output_for_row(r)

    return training_data

def main():
    csv, labels = read_input('./iris.data.txt')
    validation_csv, validation_labels = read_input('./iris.data.validation.txt')

    assert labels == validation_labels, 'Labels and validation_labels do not match!'

    training_data = generate_training_data(csv)

    nn = Network()

    l1 = nn.append_layer(InputLayer())
    l2 = nn.append_layer(Layer())
    l3 = nn.append_layer(Layer())
    l4 = nn.append_layer(OutputLayer())

    n1 = l1.add_neuron(InputNeuron())
    n2 = l1.add_neuron(InputNeuron())
    n3 = l1.add_neuron(InputNeuron())
    n4 = l1.add_neuron(InputNeuron())
    b1 = l1.add_neuron(BiasNeuron())
    n5 = l2.add_neuron(Neuron())
    n6 = l2.add_neuron(Neuron())
    n7 = l2.add_neuron(Neuron())
    n8 = l2.add_neuron(Neuron())
    n9 = l2.add_neuron(Neuron())
    b1 = l2.add_neuron(BiasNeuron())
    n10 = l3.add_neuron(Neuron())
    n11 = l3.add_neuron(Neuron())
    n12 = l3.add_neuron(Neuron())
    n13 = l3.add_neuron(Neuron())
    n14 = l3.add_neuron(Neuron())
    b2 = l3.add_neuron(BiasNeuron())
    n15 = l4.add_neuron(OutputNeuron())
    n16 = l4.add_neuron(OutputNeuron())
    n17 = l4.add_neuron(OutputNeuron())

    nn.bind_axons()

    # Tanh
    nn.set_g(lambda x: tanh(x), lambda x: 1 - tanh(x)**2)
    # ReLU
    # nn.set_g(lambda x: x if x >= 0 else 0, lambda x: 1 if x >= 0 else 0)

    nn.train(
        # Training data in format of:
        # <input>: <expected-output>
        training_data=training_data,
        iterations=10000,
        stepsize=0.1
    )

    for r in validation_csv:
        expect = expected_output_for_row(r)
        winner = max(expect)
        print('Expected:', expect, '(' + str(labels[expect.index(winner)]) + ')')

        ret = nn.calc(*r[0:4])
        winner = max(ret)
        print('Got:', '(%.2f, %.2f, %.2f)' % ret, '(' + str(labels[ret.index(winner)]) + ')')

if __name__ == "__main__":
    main()
