import numpy as np
import random
from fpdf import FPDF


def create_shape(shape_type, size, position):
    matrix = np.zeros((10, 10))
    if shape_type == 'triangle':
        for i in range(size):
            matrix[position[0] + i, position[1]:position[1] + size - i] = 1
    elif shape_type == 'rectangle':
        matrix[position[0]:position[0] + size, position[1]:position[1] + size] = 1
    elif shape_type == 'trapezoid':
        for i in range(size):
            matrix[position[0] + i, position[1] + i:position[1] + 2 * size - i] = 1
    return matrix

def generate_data():
    shapes = ['triangle', 'rectangle', 'trapezoid']
    data = []
    for _ in range(20):
        group = []
        for shape in shapes:
            size = random.randint(1, 4)
            position = (random.randint(0, 9 - size), random.randint(0, 9 - size))
            group.append(create_shape(shape, size, position))
        data.append(group)
    return data

data = generate_data()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    return W1, W2


def feedforward(X, W1, W2):
    z1 = sigmoid(np.dot(X, W1))
    z2 = sigmoid(np.dot(z1, W2))
    return z1, z2


def backpropagate(X, y, W1, W2, z1, z2, learning_rate=0.1):
    output_error = y - z2
    output_delta = output_error * sigmoid_derivative(z2)

    hidden_error = output_delta.dot(W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(z1)

    W2 += z1.T.dot(output_delta) * learning_rate
    W1 += X.T.dot(hidden_delta) * learning_rate


def train_network(X, y, hidden_size, epochs=1000, learning_rate=0.1):
    input_size, output_size = X.shape[1], y.shape[1]
    W1, W2 = initialize_weights(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        z1, z2 = feedforward(X, W1, W2)
        backpropagate(X, y, W1, W2, z1, z2, learning_rate)
    return W1, W2


def predict(X, W1, W2):
    _, z2 = feedforward(X, W1, W2)
    return np.argmax(z2, axis=1)


def prepare_dataset(data):
    X = []
    y = []
    for group in data:
        for shape in group:
            X.append(shape.flatten())
        y.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # One-hot encoding
    return np.array(X), np.array(y)


def evaluate_model(W1, W2, X_test, y_test):
    predictions = predict(X_test, W1, W2)
    labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy


def run_experiment(data, hidden_size, training_scenario):
    results = {}
    for train_size, test_size in training_scenario:
        np.random.shuffle(data)
        train_data, test_data = data[:train_size], data[train_size:train_size + test_size]
        X_train, y_train = prepare_dataset(train_data)
        X_test, y_test = prepare_dataset(test_data)

        W1, W2 = train_network(X_train, y_train, hidden_size)
        accuracy = evaluate_model(W1, W2, X_test, y_test)
        results[f'{train_size}_groups'] = accuracy
    return results


data = generate_data()
training_scenario = [(4, 16), (9, 11), (15, 5)]
results_one_hidden = run_experiment(data, hidden_size=50, training_scenario=training_scenario)
results_two_hidden = run_experiment(data, hidden_size=100, training_scenario=training_scenario)

print("One Hidden Layer Network Results:", results_one_hidden)
print("Two Hidden Layers Network Results:", results_two_hidden)



class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Neural Network Training Results', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()


def create_pdf(results_one_hidden, results_two_hidden):
    pdf = PDF()
    pdf.add_page()

    pdf.chapter_title('One Hidden Layer Network Results')
    body = "\n".join([f"{k}: {v * 100:.2f}%" for k, v in results_one_hidden.items()])
    pdf.chapter_body(body)

    pdf.chapter_title('Two Hidden Layers Network Results')
    body = "\n".join([f"{k}: {v * 100:.2f}%" for k, v in results_two_hidden.items()])
    pdf.chapter_body(body)

    pdf.chapter_title('Analysis')
    analysis = "Explain the results here..."
    pdf.chapter_body(analysis)

    pdf.output('neural_network_results.pdf')


create_pdf(results_one_hidden, results_two_hidden)
