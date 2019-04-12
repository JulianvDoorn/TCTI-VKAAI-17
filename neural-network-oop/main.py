import random
from numpy import tanh
from typing import List, Tuple, Dict
import csv

## Neuron class used for hidden neurons
#
# @details
# Every neuron should inherit from Neuron. This provides:
# - an neuron index for easy identification;
# - a forward bound list of neurons (self neuron -> connected neuron);
# - a backward bound lists of neurons (connected neuron <- self neuron);
# - functions for calculating an activation, also known as forward propagation;
# - and functions for updating weights based on the error, also known as
#   backward propagation.
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

    ## Places neuron n with weight w into the backward bound dictionary
    #
    # @details
    # The backward bound dictionary is a backward reference to the neuron in
    # the layer on the left side of layer where neuron 'self' resides in.
    # Assuming the network topology goes from left to right in terms of
    # input -> output.
    #
    # @param n Neuron to backward bind
    # @param w Weight to bind neuron n with
    def backward_bind(self, n, w):
        self.backward_bound[n] = w

    ## Retrieves the weight of backward bound neuron n
    #
    # @details
    # Only backward bound neurons' weights are stored in backward_bound[n]
    # Attempting to reference a forward bound neuron will raise an exception
    #
    # @param n Neuron to retrieve the weight of
    def get_weight_to_neuron(self, n):
        return self.backward_bound[n]

    ## Binds neuron n on the right side of neuron self, assuming left-to-right
    #  topology
    #
    # @details
    # Binding a neuron does not mean it's immediately ready for use. Binding a
    # neuron should be postseeded by nn.bind_axoms() where nn is a neural
    # network. This requires all neurons to be apart of a network instance.
    #
    # @param n Neuron to forward bind
    def forward_bind(self, n):
        self.forward_bound.append(n)

    ## Sets the activation function for neuron self
    #
    # @details
    # A neuron does not derive automatically based on g(). That's why a
    # hand-written derivative g should be passed in as well as der_g.
    #
    # @param g Activation function like ReLU, Tanh, Sigmoid et al.
    # @param der_g Derivative function of g
    def set_g(self, g, der_g):
        self.g = g
        self.der_g = der_g

    ## Resets the calculated activation and error so forward and backward
    #  propagation call works with new values
    def reset(self):
        self.stored_activation = None
        self.stored_error = None

    ## Returns the sum of the activation of all backward bound neurons
    #
    # @details
    # The returned value should be used as input for self.g(x) or self.der_g(x)
    # 
    # @returns sum(activation * weight for all backward bound items) 
    def get_input(self):
        return sum([
            n.get_activation() * w for n, w in self.backward_bound.items()
        ])

    ## Calculates and stores the activation using self.g and self.get_input
    #
    # @returns Calculated activation, essentially a neuron's output
    def get_activation(self):
        if self.stored_activation is None:
            self.stored_activation = self.g(self.get_input())

        return self.stored_activation

    ## Calculates the accountable error for neuron self
    #
    # @returns Accountable error for self
    def get_error(self):
        if self.stored_error is None:
            self.stored_error = self.der_g(self.get_input()) * sum([
                n.get_weight_to_neuron(self) * n.get_error() for n in self.forward_bound
            ])

        return self.stored_error

    def __repr__(self):
        return "Neuron " + str(self.index)

## Input neuron class for receiving inputs into the neural network
#
# @details
# The input neuron is stripped of some functionality which can only work for
# hidden neurons or output neurons. Aside from that it also has an additional
# set_input() function
class InputNeuron(Neuron):
    def __init__(self):
        super(InputNeuron, self).__init__()
        self.input = None

    ## Retrieves the input as set by set_input(v)
    def get_input(self):
        return self.input

    ## Retrieves activation as self.input as set by set_input(v)
    def get_activation(self):
        return self.input

    ## Sets the input of this input neuron
    def set_input(self, v):
        self.input = v

    ## Invokes n.get_error() on all forward bound neurons to initate recursive
    #  backward propagation. Always returns 0, because the input cannot have
    #  an error
    #
    # @return 0
    def get_error(self):
        for n in self.forward_bound:
            n.get_error()

        return 0

## Output neuron class for setting a desired output for the neural network
#
# @details
# The output neuron overrides the get_error() function which calculates an
# error of the input and output of the whole network. get_error() is supported
# by set_desired(d), which sets the desired activation value for neuron self.
class OutputNeuron(Neuron):
    def __init__(self):
        super(OutputNeuron, self).__init__()
        self.desired_output = None

    ## Calculates an error over the input and output of the whole neural
    #  network so that the hidden neurons can backpropagate
    def get_error(self):
        if self.stored_error is None:
            self.stored_error = self.der_g(
                self.get_input()) * (self.desired_output - self.get_activation())

        return self.stored_error

    ## Sets the desired output for the neural network where neuron self resides
    #  in
    def set_desired(self, d):
        self.desired_output = d

## Bias neuron class in order to have a generically trainable threshold
#
# @details
# A bias should be bound to a right layer just as any neuron. Only bias neurons
# cannot have an input set by a caller or another neuron. The input always
# equals -1 in this implementation. However- the weight is still trainable as a
# regular hidden neuron
class BiasNeuron(InputNeuron):
    def __init__(self):
        super(BiasNeuron, self).__init__()
        self.input = -1

    ## Does nothing in order to maintain self.input == -1
    def set_input(self, v):
        pass

## Layer class for ease of describing a network topology
class Layer:
    def __init__(self):
        self.neurons = []

    ## Adds a neuron to the list of neurons
    def add_neuron(self, neuron):
        self.neurons.append(neuron)
        return neuron

    ## Resets all neurons in the list of neurons
    def reset(self):
        for n in self.neurons:
            n.reset()

    ## Sets g and derivative g for all neurons in the list of neurons
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

## Input layer class for the ease of providing an input vector
class InputLayer(Layer):
    def __init__(self):
        super(InputLayer, self).__init__()

    ## For every neuron with index i, the i'th element from input vector is
    #  taken as input
    def input(self, input_vector):
        for i, v in enumerate(input_vector):
            self.neurons[i].set_input(v)

## Output layer class for the ease of retrieving an output vector
class OutputLayer(Layer):
    def __init__(self):
        super(OutputLayer, self).__init__()

    ## For every neuron n the activation is calculated and expanded into a
    #  output vector
    #
    # @return Tuple of n.get_activation() for every neuron n
    def output(self):
        return (*[n.get_activation() for n in self.neurons],)

## Network class for allowing to appending layers and binding axons
#
# @details
# Note that neurons should not be used without an owning network instance
class Network:
    def __init__(self):
        self.layers = []
        self.hidden_layers = []
        self.input_layer = None
        self.output_layer = None

    ## Appends a layer to the list of layers and then updates the hidden_layers
    #  list, the input_layer and the output_layer
    #
    # @return The appended layer, allows for easier instantation (e.g. my_layer
    #         = nn.append_layer(InputLayer()))
    def append_layer(self, layer):
        self.layers.append(layer)

        if len(self.layers) > 2:
            self.hidden_layers = self.layers[1:-1]
        else:
            self.hidden_layers = []

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        return layer

    ## Prepares all the owned neurons in network self for forward and backward
    #  propagation
    #
    # @details
    # For every neuron, the correct neurons are forward and backward bounds.
    # Preceded by a couple of sanity checks.
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

    ## Sets g and derivative g for every layer in this network (which in order
    #  sets it for every neuron in said layer)
    def set_g(self, g, der_g):
        for l in self.layers:
            l.set_g(g, der_g)

    ## Forwards the input vector to the input layer
    def input(self, input_vector):
        self.input_layer.input(input_vector)

    ## Forward propagates through the network to calculate an output
    def output(self):
        for l in self.layers:
            l.reset()
        return self.output_layer.output()

    def calc(self, *args):
        self.input((args))
        return self.output()

    ## Backward propagates through the network to calculate error deltas, these
    #  deltas are then applied. The previous operation is repeated for a given
    #  amount of iterations
    #
    # @details
    # First propagate forward
    # then propagate backward and calculate error per neuron.
    # Adjust weights with caluclated errors.
    # Repeat for n times for all training data.
    #
    # @param iterations Amount of iterations to train
    # @param training_data Pairs of input and outputs to train the network for
    # @param stepsize Stepsize of the gradient descent of network self
    def train(self, **kwargs):
        assert "iterations" in kwargs.keys(), "Missing kwarg: iterations"
        assert "training_data" in kwargs.keys(), "Missing kwarg: training_data"
        assert "stepsize" in kwargs.keys(), "Missing kwarg: stepsize"

        iterations = kwargs["iterations"]
        training_data = kwargs["training_data"]
        stepsize = kwargs["stepsize"]

        for _ in range(iterations):
            for input, desired_output in training_data.items():
                self.input(input)
                self.propagate_backward(desired_output, stepsize)

    ## Backward propagates thorugh the network to calculate error deltas, these
    #  deltas are then applied, exactly one time.
    #
    # @details
    # It is required that the input is already given using
    # self.input(input_vector), otherwise the error has no basis. And will
    # instead raise an exception.
    #
    # @param desired_output Desired output for the already given input
    # @param stepsize Stepsize of the gradient descent of network self
    def propagate_backward(self, desired_output, stepsize):
        obtained_output = self.output()

        for i, _ in enumerate(obtained_output):
            self.output_layer[i].set_desired(desired_output[i])

        # Adjust error_per_neuron
        # But the input layer holds no weights, skip it
        for l in self.layers[1:]:
            for n in l:
                for prev_n, w in n.backward_bound.items():
                    n.backward_bound[prev_n] = w + stepsize * \
                        prev_n.get_activation() * n.get_error()

## Obtains a formatted csv file for the given file url
#
# @details
# CSV is formatted as such:
# An input for "5.9,3.0,5.1,1.8,Iris-virginica"
# results in a row of [5.9, 3.0, 5.1, 1.8, 1]
# Where the latest index (with value 1) is the label translated to a unique id
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

## Returns the expected output for a given row where the latest index is the
#  validation label for the given row.
#
# @details
# Note that this is an implementation specific function. This function is
# tailored exactly to the needs of our dataset.
#
# @return (1, 0, 0) when id equals 0 (Iris Setosa)
# @return (0, 1, 0) when id equals 1 (Iris Versicolour)
# @return (0, 0, 1) when id equals 2 (Iris Virginica)
def expected_output_for_row(row):
    if row[4] == 0:
        return (1, 0, 0)
    if row[4] == 1:
        return (0, 1, 0)
    if row[4] == 2:
        return (0, 0, 1)

## Returns the name from the output value of the neural network and a list of
#  associated labels
#
# @return Iris Setosa when val equals (1, 0, 0)
# @return Iris Versicolour when val equals (0, 1, 0)
# @return Iris Virginica when val equals (0, 0, 1)
def get_label_from_output(val, labels):
    if val == (1, 0, 0):
        return labels[0]
    if val == (0, 1, 0):
        return labels[1]
    if val == (0, 0, 1):
        return labels[2]

## Turns the input rows into a trainable list of tuples
#
# @details
# Format:
# {
#   (0, 0, 1): 2  # The 3rd element in the tuple represents uid 2, which in
#                 # turn represents Iris Virginica
#   ...           # The list of training data may continue indefinitely
# }
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

    l1.add_neuron(InputNeuron())
    l1.add_neuron(InputNeuron())
    l1.add_neuron(InputNeuron())
    l1.add_neuron(InputNeuron())
    l1.add_neuron(BiasNeuron())
    
    l2.add_neuron(Neuron())
    l2.add_neuron(Neuron())
    l2.add_neuron(Neuron())
    l2.add_neuron(Neuron())
    l2.add_neuron(Neuron())
    l2.add_neuron(BiasNeuron())
    
    l3.add_neuron(Neuron())
    l3.add_neuron(Neuron())
    l3.add_neuron(Neuron())
    l3.add_neuron(Neuron())
    l3.add_neuron(Neuron())
    l3.add_neuron(BiasNeuron())
    
    l4.add_neuron(OutputNeuron())
    l4.add_neuron(OutputNeuron())
    l4.add_neuron(OutputNeuron())

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
        print('Got: (%.2f, %.2f, %.2f)' % ret, '(' + str(labels[ret.index(winner)]) + ')')

if __name__ == "__main__":
    main()
