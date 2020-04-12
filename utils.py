
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(original_array):
    x = np.array(original_array)
    max = np.max(x)
    return np.exp(x-max) / np.sum(np.exp(x-max))

def tanh(original_array):
    x = np.array(original_array)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
