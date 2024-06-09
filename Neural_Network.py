import random

import numpy as np
import json
from decimal import Decimal, getcontext

# Set the precision to 200 digits
getcontext().prec = 20

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1, self.hidden_size2, self.output_size = hidden_size1, hidden_size2, output_size
        # Initialize weights and biases with random values
        self.w1 = [Decimal(random.random()) for _ in
                   range(input_size * hidden_size1)]  # np.random.rand(input_size, hidden_size1)
        self.b1 = np.zeros((1, hidden_size1))  # Bias for first hidden layer
        self.w2 = [Decimal(random.random()) for _ in
                   range(hidden_size2 * hidden_size1)]  # np.random.rand(hidden_size1, hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))  # Bias for second hidden layer
        self.w3 = [Decimal(random.random()) for _ in
                   range(output_size * hidden_size2)]  # np.random.rand(hidden_size2, output_size)
        self.b3 = np.zeros((1, output_size))  # Bias for output layer

        # Choose an activation function (e.g., sigmoid, ReLU)
        self.activation_func = self.sigmoid  # Placeholder, replace with your desired function

    def sigmoid(self, x):
        """Sigmoid activation function (example)"""
        return 1 / (1 + np.exp(-x))

    def scalar_product(self, v1, v2):
        sum = 0
        for i in range(len(v1)):
            sum += v1[i] + v2[i]
        return sum

    def vector_connection(self, v1, v2):
        v = []
        for i in range(len(v1)):
            v.append(v1[i] + v2[i])
        return v

    def dif_sigmoid(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def predict(self, X):
        # Forward pass
        outputs=[]
        for x in X:
            z1 = []
            a1 = []
            for i in range(self.hidden_size1):
                z = self.scalar_product(self.w1[i * len(x):(i + 1) * len(x)], x)
                a = self.sigmoid(z)
                z1.append(z)
                a1.append(a)
            z2 = []
            a2 = []
            for i in range(self.hidden_size2):
                z = self.scalar_product(self.w2[i * len(a1):(i + 1) * len(a1)], a1)
                a = self.sigmoid(z)
                z2.append(z)
                a2.append(a)
            z3 = []

            output = []
            for i in range(self.output_size):
                w = self.w3[i * len(a2):(i + 1) * len(a2)]
                z = self.scalar_product(w, a2)
                a = self.sigmoid(z)
                z3.append(z)
                output.append(a)
            outputs.append(output)



        """Forward pass to calculate output"""
        # z1 = np.dot(x, self.w1) + self.b1  # Weighted sum at hidden layer 1
        # a1 = self.activation_func(z1)  # Apply activation function
        #
        # z2 = np.dot(a1, self.w2) + self.b2  # Weighted sum at hidden layer 2
        # a2 = self.activation_func(z2)  # Apply activation function
        #
        # z3 = np.dot(a2, self.w3) + self.b3  # Weighted sum at output layer
        # output = self.activation_func(z3)  # Apply activation function (can be different for output)


        return outputs

    def train(self, learning_rate, epochs, X_train, y_train):
        for t in range(len(X_train)):#range(len(X_train)):

            print('|', end="")
            # x = np.array([X_train[i]])
            x = X_train[t]

            # if i == 0:
            #     print(x)
            # Forward pass
            z1 = []
            a1 = []
            for i in range(self.hidden_size1):
                z = self.scalar_product(self.w1[i * len(x):(i + 1) * len(x)], x)
                a = self.sigmoid(z)
                z1.append(z)
                a1.append(a)
            z2 = []
            a2 = []
            for i in range(self.hidden_size2):
                z = self.scalar_product(self.w2[i * len(a1):(i + 1) * len(a1)], a1)
                a = self.sigmoid(z)
                z2.append(z)
                a2.append(a)
            z3 = []
            output = []
            for i in range(self.output_size):
                z = self.scalar_product(self.w3[i * len(a2):(i + 1) * len(a2)], a2)
                a = self.sigmoid(z)
                z3.append(z)
                output.append(a)

            # output is single lyere
            dy = Decimal(0.01)*self.dif_sigmoid(z3[0]) * (y_train[t] - output[0])  # if out putput size!=1=>crach
            # print(dy)
            for i in range(len(self.w3)):
                self.w3[i] += dy * a2[i]
            d2 = []
            for i in range(self.hidden_size2):
                d = Decimal(0.01)*self.dif_sigmoid(z2[i]) * (dy * self.w3[i])
                d2.append(d)
                for j in range(self.hidden_size1):
                    self.w2[i * self.hidden_size1 + j] += d * a1[j]
            d1 = []
            for i in range(self.hidden_size1):
                # calc d2*w2
                dw = 0
                for j in range(self.hidden_size2):
                    dw += d2[j] * self.w2[i + j * self.hidden_size1]
                d = Decimal(0.01)*self.dif_sigmoid(z1[i]) * dw  # * (dy * self.w2[i])
                d1.append(d)
                # todo check
                for j in range(self.input_size):
                    self.w1[i * self.input_size + j] += d * x[j]

            # todo w2 w1
        print()
<<<<<<< HEAD



=======
>>>>>>> 78f9d83 (decimal)
