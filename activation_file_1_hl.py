import numpy as np
from sets import prepare_data
from sets import prepare_random_data
from Neural_Network_1_hidden_layer import NeuralNetwork

input_layer, labels = prepare_random_data()


split = 15 * 3
nn = NeuralNetwork(learning_rate=0.1, input_size=100, hidden_size=80, output_size=1)
nn.train(input_layer[:split], labels[:split], epochs=1)
predictions = nn.predict(input_layer[:split])  # [1,0,0][0,0.39,0]

count_success = 0
for i in range(len(predictions)):
    mod = i % 3
    if labels[i][0] == 1/6 and predictions[i][0] < 1 / 3:
        count_success += 1
    elif labels[i][0] == 0.5 and 2 / 3 > predictions[i][0] > 1 / 3:
        count_success += 1
    elif labels[i][0] == 5/6 and predictions[i][0] > 2 / 3:
        count_success += 1
accuracy = count_success / len(predictions)
print(f'Accuracy: {accuracy:.2f}')

errors = []
for j in range(len(predictions)):
    errors.append(sum((predictions[j] - labels[:split][j]) ** 2))  # [1,0,0]-[0,0.39,0]
data = np.array(errors)
mean_error = np.mean(data)
print(f'Mean Error: {mean_error:.4f}')
er = mean_error
counter = 1
while accuracy < 1:
    counter += 1
    nn.train(input_layer[:split], labels[:split], epochs=1)
    predictions = nn.predict(input_layer[:split])
    print('circle')
    count_success = 0
    for i in range(len(predictions)):
        mod = i % 3
        if labels[i][0] == 1 / 6 and predictions[i][0] < 1 / 3:
            count_success += 1
        elif labels[i][0] == 0.5 and 2 / 3 > predictions[i][0] > 1 / 3:
            count_success += 1
        elif labels[i][0] == 5 / 6 and predictions[i][0] > 2 / 3:
            count_success += 1

    accuracy = count_success / len(predictions)
    print(f'Accuracy: {accuracy:.2f}')

print(counter)

# לבדוק אחוזים על יתר הקבוצות

predictions = nn.predict(input_layer[split:])  # [1,0,0][0,0.39,0]

count_success = 0
for i in range(len(predictions)):
    mod = i % 3
    if labels[i+split][0] == 1/6 and predictions[i][0] < 1 / 3:
        count_success += 1
    elif labels[i + split ][0] == 0.5 and 2 / 3 > predictions[i][0] > 1 / 3:
        count_success += 1
    elif labels[i+split][0] == 5/6 and predictions[i][0] > 2 / 3:
        count_success += 1

accuracy = count_success / len(predictions)
print(f'Accuracy: {accuracy:.2f}')
