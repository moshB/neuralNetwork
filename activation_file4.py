import numpy as np
from sets import prepare_data
from Neural_Network4 import NeuralNetwork

input_layer, labels = prepare_data()

nn = NeuralNetwork(learning_rate=0.001, input_size=100, hidden_size1=48, hidden_size2=21, output_size=3)
nn.train(input_layer, labels, epochs=1000)

predictions = nn.predict(input_layer)

count_success = 0
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        if max(predictions[i]) == predictions[i][j] and i % 3 == j:
            count_success += 1

accuracy = count_success / (len(predictions) * 3)
print(f'Accuracy: {accuracy:.2f}')

errors = []
for j in range(len(predictions)):
    errors.append(sum((predictions[j] - labels[j % 3]) ** 2))
data = np.array(errors)
mean_error = np.mean(data)
print(f'Mean Error: {mean_error:.4f}')
