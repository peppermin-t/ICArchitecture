import numpy as np


def sigmoid(in_ver):
    return (1 + np.exp(-in_ver)) ** -1


def sigmoid_diff(in_ver):
    return sigmoid(in_ver) * (1 - sigmoid(in_ver))
