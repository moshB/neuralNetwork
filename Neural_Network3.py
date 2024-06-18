import numpy as np
import json


class NeuralNetwork:
    def __init__(self, a, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.a = a
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        # Initialize weights and biases with random values
        self.w1 = np.random.rand(input_size, hidden_size1)*2-1

        self.w2 = np.random.rand(hidden_size1, hidden_size2)*2-1
        self.w3 = np.random.rand(hidden_size2, output_size)*2-1

        # Choose an activation function (e.g., sigmoid, ReLU)
        self.activation_func = self.sigmoid  # Placeholder, replace with your desired function

    def sigmoid(self, x):
        """Sigmoid activation function (example)"""
        return 1 / (1 + np.exp(-self.a * x))

    def dif_sigmoid(self, x):
        return self.a * np.exp(-self.a * x) / ((1 + np.exp(-self.a * x)) ** 2)

    def predict(self, X_train):
        outputs = []
        for row in range(len(X_train)):
            x = X_train[row]
            for col in range(len(X_train[row])):
                input = x[col]

                # Forward pass
                z1 = np.dot(input, self.w1)
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.w2)
                a2 = self.sigmoid(z2)
                z3 = np.dot(a2, self.w3)
                output = self.sigmoid(z3)
                outputs.append(output)
        return outputs

    def train(self, learning_rate, epochs, X_train, y_train):
        ans = np.array(y_train)
        num_groups = 8
        for epoch in range(epochs):
            for row in range(num_groups):  # len(X_train)):
                x = X_train[row]
                for col in range(len(X_train[row])):
                    input = x[col]

                    # Forward pass
                    z1 = np.dot(input, self.w1)
                    a1 = self.sigmoid(z1)
                    z2 = np.dot(a1, self.w2)
                    a2 = self.sigmoid(z2)
                    z3 = np.dot(a2, self.w3)
                    output = self.sigmoid(z3)

                    d = ans[col]

                    # Backpropagation
                    error =  (d - output)  # todo small the affect
                    # err = d - output#todo small the affect
                    # error = (sum(err**2))**0.5/len(err)
                    # if error !=0:
                    #     err = err/error

                    fi_3 = self.dif_sigmoid(z3)
                    fi_2 = self.dif_sigmoid(z2)
                    fi_1 = self.dif_sigmoid(z1)

                    delta3 = error * fi_3

                    sum2 = []
                    for i in range(self.hidden_size2):
                        sum2.append(sum(self.w3[i] * delta3))
                    delta2 = fi_2 * sum2
                    print('delta2-------------')
                    print(len(delta2))
                    print(delta2)
                    print(fi_2)
                    print(delta3)

                    sum1 = []
                    for i in range(self.hidden_size1):
                        sum1.append(sum(self.w2[i] * delta2))
                    delta1 = fi_1 * sum1

                    self.w3 += learning_rate * np.outer(a2, delta3)
                    self.w2 += learning_rate * np.outer(a1, delta2)
                    self.w1 += learning_rate * np.outer(input, delta1)
                    # print(self.w2)
            r = X_train
            # print('--------')
            # print(len(r))
            pr = self.predict(r)
            count_sucses = 0
            if epoch % 100 == 0:
                print('|', end='')
            # for i in range(0,12*3):
            # p = pr[i:i + 2]
            # print(p)
            # for j in range(3):
            #         if (max(pr[i:i + 2]) == p[j]) and (i//3) % 3 == j:
            #             count_sucses += 1
            # if count_sucses / (12 * 3) >= 0.6:
            # print()
            # print(epoch)
            # print(count_sucses / (12 * 3))
            # return epoch
            elif epoch % 1000 == 0:
                print()
                # print(count_sucses )
                print()
