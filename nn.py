# / ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *
# *Neural
# Networks *
# *Project
# number
# 2 - Back
# Propagation *
# * *
# *This
# project
# executed
# by: *
# *Volinsky
# Irina *
# *ID: 310598255 *
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# * /
#
# // This is a
# Back
# Propagation
# network.It
# consists
# layers:
# // 25
# neurons
# on
# input
# layer, 10
# neurons
# on
# hidden
# layer and one
# // neuron
# on
# output
# layer.
#
# // This
# programm
# use
# function: F(NET) = tanh(NET), that
# takes
# values
# // from
#
# -1
# to + 1.
# // The
# values
# of
# the
# neurons in the
# hidden
# layer
# are
# continuous.
# // The
# values
# of
# the
# input and output
# neurons
# are
# diskreet.
#
# // This
# program
# do
# not print
# anything
# to
# display, and all
# results
# of
# // this
# programm
# will
# be in file
# "result.txt" and "bias_result.txt"
# // after
# runing
# of
# this
# programm.
# // Before
# runing
# programm
# againe, please
# delete
# old
# file
# with results.
import numpy as np


# include<iostream.h>
# include<stdlib.h>
# include<time.h>
# include<math.h>

# include<fcntl.h>
# include<sys\stat.h>
# include<io.h>

# include "Patterns.dat"        //File with patterns for input end output.

# define Low           -1
# define Hi	          +1
# define Bias          1

# define InputNeurons  25
# define HiddenNeurons 10

# define sqr(x)        ((x)*(x))

# typedef
# int
# InArr[InputNeurons];
#
# enum
# bool
# {false, true};


class Data:
  def __init__(self, units):
    """
    Constructor to initialize the data object with the number of units.
    """
    self.units = units
    self.input = None  # Initialize input as None
    self.output = None  # Initialize output as None

  def set_input_output(self, input_patterns, output_patterns):
    """
    Sets the input and output patterns for the data object.

    Args:
        input_patterns (list): A list of lists representing the input patterns.
        output_patterns (list): A list of integers representing the output patterns.

    Raises:
        ValueError: If the length of input patterns doesn't match the number of units.
        ValueError: If the length of output patterns doesn't match the number of units.
    """
    if len(input_patterns[0]) != self.units:
      raise ValueError("Number of elements in input pattern doesn't match units")
    if len(output_patterns) != self.units:
      raise ValueError("Number of elements in output pattern doesn't match units")

    self.input = np.array(input_patterns, dtype=int)  # Convert to NumPy array for efficiency
    self.output = np.array(output_patterns, dtype=int)  # Convert to NumPy array for efficiency

  def reset(self):
    """
    Resets the input and output patterns to None, effectively freeing memory.
    """
    self.input = None
    self.output = None

# Example usage
units = 3  # Number of units in input and output patterns
data_obj = Data(units)

# Assuming you have your input and output patterns prepared elsewhere
input_patterns = [[1, 2, 3], [4, 5, 6]]
output_patterns = [0, 1]

data_obj.set_input_output(input_patterns, output_patterns)

# Access input and output patterns
# (assuming you're using NumPy for further processing)
print(data_obj.input)
print(data_obj.output)

data_obj.reset()  # Free memory when done

import numpy as np

class BackpropagationNet:
    def __init__(self, input_neurons, hidden_neurons, learning_rate, threshold=0):
        """
        Initialize the Backpropagation network with given parameters.

        Args:
            input_neurons (int): Number of neurons in the input layer.
            hidden_neurons (int): Number of neurons in the hidden layer.
            learning_rate (float): Learning rate for weight updates.
            threshold (float, optional): Threshold value for activation function. Defaults to 0.
        """
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.threshold = threshold

        # Initialize weights with random values between -1 and 1
        self.weights_ih = np.random.rand(hidden_neurons, input_neurons + 1) * 2 - 1  # Include bias
        self.weights_ho = np.random.rand(1, hidden_neurons + 1) * 2 - 1  # Include bias

        # Initialize activation layer outputs (all set to 1 initially)
        self.hidden_layer = np.ones(hidden_neurons + 1)  # +1 for bias
        self.output_layer = 0

        # No error initially
        self.net_error = False

    def random_equal_real(self, low, high):
        """
        Generates a random number between low and high (inclusive).
        """
        return np.random.rand() * (high - low) + low

    def sigmoid(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid activation function.
        """
        return x * (1 - x)

    def calculate_output_with_bias(self, input_pattern):
        """
        Calculates the network output with bias.

        Args:
            input_pattern (np.ndarray): Input pattern for the network.

        Returns:
            float: The network output.
        """
        # Add bias to input
        biased_input = np.append(input_pattern, 1)

        # Calculate hidden layer activation
        self.hidden_layer = self.sigmoid(np.dot(self.weights_ih, biased_input))

        # Calculate output layer activation
        self.output_layer = self.sigmoid(np.dot(self.weights_ho, self.hidden_layer))
        return self.output_layer

    def calculate_output(self, input_pattern):
        """
        Calculates the network output without bias.

        Args:
            input_pattern (np.ndarray): Input pattern for the network.

        Returns:
            float: The network output.
        """
        # No bias for input
        hidden_layer_activation = self.sigmoid(np.dot(self.weights_ih[:, :-1], input_pattern))  # Exclude bias weight
        self.hidden_layer[1:] = hidden_layer_activation  # Update hidden layer activations (excluding bias)
        self.output_layer = self.sigmoid(np.dot(self.weights_ho, self.hidden_layer))
        return self.output_layer

    def is_error(self, target):
        """
        Checks if the network output differs from the target value by a threshold.

        Args:
            target (int): Target value for the output.

        Returns:
            bool: True if there is an error, False otherwise.
        """
        self.net_error = abs(self.output_layer - target) > self.threshold
        return self.net_error

    def adjust_weights_with_bias(self, target):
        """
        Adjusts the weights of the network with bias based on the error.

        Args:
            target (int): Target value for the output.
        """
        # Backpropagation with bias
        output_delta = (self.output_layer - target) * self.sigmoid_derivative(self.output_layer)
        hidden_delta = np.dot(self.weights_ho.T, output_delta) * self.sigmoid_derivative(self.hidden_layer)

        # Update output layer weights
        self.weights_ho += self.learning_rate *

        # Update output layer weights (including bias)
        self.weights_ho += self.learning_rate * np.outer(output_delta, self.hidden_layer)

        # Update hidden layer weights (including bias)
        self.weights_ih += self.learning_rate * np.outer(hidden_delta[1:], np.append(self.input_pattern, 1))  # Exclude bias delta and include bias in input



