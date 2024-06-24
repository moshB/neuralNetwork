import numpy as np
from sets import prepare_data
from Neural_Network_1_hidden_layer import NeuralNetwork

input_layer, labels = prepare_data()


print(f'Input shape: {input_layer.shape}')
print(f'Labels shape: {labels.shape}')
split=17*3
nn = NeuralNetwork(learning_rate=0.0001, input_size=100, hidden_size=89, output_size=3)
nn.train(input_layer[:split], labels[:split], epochs=20000)
# print(len(input_layer))
predictions = nn.predict(input_layer[split:])

count_success = 0
for i in range(len(predictions)):
    if np.argmax(predictions[i]) == np.argmax(labels[split:][i]):
        count_success += 1

accuracy = count_success / len(predictions)
print(f'Accuracy: {accuracy:.2f}')

errors = []
for j in range(len(predictions)):
    errors.append(sum((predictions[j] - labels[split:][j]) ** 2))
data = np.array(errors)
mean_error = np.mean(data)
print(f'Mean Error: {mean_error:.4f}')
er = mean_error
while accuracy<0.8:
    # nn.lr = nn.lr*0.98
    nn.train(input_layer[:split], labels[:split], epochs=16000)
    # print(len(input_layer))
    predictions = nn.predict(input_layer[split:])
    print('circle')
    count_success = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(labels[split:][i]):
            count_success += 1

    accuracy = count_success / len(predictions)
    print(f'Accuracy: {accuracy:.2f}')

    errors = []
    for j in range(len(predictions)):
        errors.append(sum((predictions[j] - labels[split:][j]) ** 2))
    data = np.array(errors)
    mean_error = np.mean(data)
    print(f'Mean Error: {mean_error:.4f}')
    # print(nn.lr)
    # if mean_error>er:
    #     nn.lr *=0.99
    #     er = mean_error
    # else:
    #     er = mean_error
