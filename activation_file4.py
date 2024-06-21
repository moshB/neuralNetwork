import numpy as np
from sets import prepare_data
from Neural_Network4 import NeuralNetwork

input_layer, labels = prepare_data()

print(f'Input shape: {input_layer.shape}')
print(f'Labels shape: {labels.shape}')

nn = NeuralNetwork(learning_rate=0.1, input_size=100, hidden_size1=89, hidden_size2=40, output_size=3)
nn.train(input_layer, labels, epochs=10000)

predictions = nn.predict(input_layer)

count_success = 0
for i in range(len(predictions)):
    if np.argmax(predictions[i]) == np.argmax(labels[i]):
        count_success += 1

accuracy = count_success / len(predictions)
print(f'Accuracy: {accuracy:.2f}')

errors = []
for j in range(len(predictions)):
    errors.append(sum((predictions[j] - labels[j]) ** 2))
data = np.array(errors)
mean_error = np.mean(data)
print(f'Mean Error: {mean_error:.4f}')