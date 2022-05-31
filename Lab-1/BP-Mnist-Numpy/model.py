import random

import numpy as np

from func import sigmoid
from func import sigmoid_diff


# neural network class
class neuralNetwork:

    # initialize the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        """
        The network consists of three layers: input layer, hidden layer and output layer.
        Here defined these layers.
        :param input_nodes: dimension of input
        :param hidden_nodes: dimension of hidden nodes
        :param output_nodes: dimension of output
        :param learning_rate: the learning rate of neural network
        """

        self.layer = 3  # 3 layers in total
        self.learning_rate = learning_rate  # learning rate ita

        weight1 = np.random.randn(hidden_nodes, input_nodes) * np.sqrt(1 / hidden_nodes)  # weights matrix first [784, 400]
        weight2 = np.random.randn(output_nodes, hidden_nodes) * np.sqrt(1 / hidden_nodes)  # weights matrix second [400, 10]
        self.weights = np.array([None, weight1, weight2])
        offset1 = np.random.rand(hidden_nodes, 1)  # offset for single middle layer first, [400, 1]
        offset2 = np.random.rand(output_nodes, 1)  # offset for single middle layer second, [10, 1]
        self.offsets = np.array([None, offset1, offset2])

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.netput = np.array([None, None, None])
        self.output = np.array([None, None, None])
        self.grad = np.array([None, None, None])
        self.final_outputs = None

    def forward(self, input_feature):
        """
        Forward the neural network
        :param input_feature: single input image, flattened [784, ]
        """

        self.output[0] = np.reshape(input_feature, (self.input_nodes, 1))
        for i in range(1, self.layer):
            self.netput[i] = np.dot(self.weights[i], self.output[i - 1]) + self.offsets[i]
            self.output[i] = sigmoid(self.netput[i])

        self.final_outputs = self.output[self.layer - 1]

    def backpropagation(self, targets_list):
        """
        Propagate backwards
        :param targets_list: output one hot code of a single image, [10, ]
        """

        targets = np.reshape(targets_list, (self.output_nodes, 1))
        self.grad[self.layer - 1] = (self.output[self.layer - 1] - targets) * sigmoid_diff(self.netput[self.layer - 1])

        for i in reversed(range(1, self.layer - 1)):
            self.grad[i] = np.dot(self.weights[i + 1].T, self.grad[i + 1]) * sigmoid_diff(self.netput[i])

        for i in range(1, self.layer):
            self.weights[i] = self.weights[i] - self.learning_rate * np.dot(self.grad[i], self.output[i - 1].T)
            self.offsets[i] = self.offsets[i] - self.learning_rate * self.grad[i]

        return np.sum(0.5 * (targets - self.output[self.layer - 1]) ** 2)
