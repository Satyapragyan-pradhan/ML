# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
import math

iris = load_iris()

X = iris.data.tolist()
y = iris.target.tolist()
feature_names = iris.feature_names

# Split data manually into training and testing sets
def split_data(X, y):
    X_train, X_test = [], []
    y_train, y_test = [], []
    classes = [0, 1, 2]

    for c in classes:
        count = 0
        for label in y:
            if label == c:
                count += 1

        train_count = int(count * 0.70)
        added = 0

        for i in range(len(y)):
            if y[i] == c:
                if added < train_count:
                    X_train.append(X[i])
                    y_train.append(y[i])
                    added += 1
                else:
                    X_test.append(X[i])
                    y_test.append(y[i])

    return X_train, X_test, y_train, y_test

# Normalize features using training data
def normalize(X_train, X_test):
    minimum = []
    maximum = []

    for j in range(len(X_train[0])):
        low = X_train[0][j]
        high = X_train[0][j]

        for row in X_train:
            if row[j] < low:
                low = row[j]
            if row[j] > high:
                high = row[j]

        minimum.append(low)
        maximum.append(high)

    for dataset in [X_train, X_test]:
        for row in dataset:
            for j in range(len(row)):
                difference = maximum[j] - minimum[j]
                if difference != 0:
                    row[j] = (row[j] - minimum[j]) / difference
                else:
                    row[j] = 0.0

# Sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# Initialize weights and biases
def initialize_network(inputs, hidden, outputs):
    network = []
    hidden_layer = []
    for i in range(hidden):
        neuron = {
            "weights": [],
            "bias": 0.1
        }

        for j in range(inputs):
            neuron["weights"].append(
                (i + j + 1) * 0.01
            )

        hidden_layer.append(neuron)

    output_layer = []

    for i in range(outputs):
        neuron = {
            "weights": [],
            "bias": 0.1
        }

        for j in range(hidden):
            neuron["weights"].append(
                (i + j + 1) * 0.01
            )

        output_layer.append(neuron)

    network.append(hidden_layer)
    network.append(output_layer)
    return network

# Forward propagation through the network
def forward(network, inputs):
    activations = inputs[:]

    for layer in network:
        new_activations = []
        for neuron in layer:
            total = neuron["bias"]
            for j in range(len(neuron["weights"])):
                total += neuron["weights"][j] * activations[j]
            new_activations.append(sigmoid(total))
        activations = new_activations

    return activations

# Train network using backpropagation
def train(network, X_train, y_train):
    learning_rate = 0.5
    epochs = 500

    for epoch in range(epochs):
        for i in range(len(X_train)):
            # Forward pass
            hidden_outputs = []
            inputs = X_train[i][:]
            for neuron in network[0]:
                total = neuron["bias"]
                for j in range(len(inputs)):
                    total += neuron["weights"][j] * inputs[j]
                hidden_outputs.append(sigmoid(total))

            final_outputs = []
            for neuron in network[1]:
                total = neuron["bias"]
                for j in range(len(hidden_outputs)):
                    total += neuron["weights"][j] * hidden_outputs[j]
                final_outputs.append(sigmoid(total))

            # One-hot encode target
            targets = [0, 0, 0]
            targets[y_train[i]] = 1

            # Output layer errors
            output_deltas = []

            for j in range(len(final_outputs)):
                output = final_outputs[j]
                delta = (targets[j] - output) * output * (1 - output)
                output_deltas.append(delta)

            # Hidden layer errors
            hidden_deltas = []
            for j in range(len(hidden_outputs)):
                error = 0.0
                for k in range(len(output_deltas)):
                    error += (output_deltas[k]* network[1][k]["weights"][j])
                output = hidden_outputs[j]
                hidden_deltas.append(error * output * (1 - output))

            # Update output layer weights and biases
            for j in range(len(network[1])):
                neuron = network[1][j]
                for k in range(len(neuron["weights"])):
                    neuron["weights"][k] += (
                        learning_rate
                        * output_deltas[j]
                        * hidden_outputs[k]
                    )

                neuron["bias"] += (learning_rate * output_deltas[j])

            # Update hidden layer weights and biases
            for j in range(len(network[0])):
                neuron = network[0][j]
                for k in range(len(neuron["weights"])):
                    neuron["weights"][k] += (
                        learning_rate
                        * hidden_deltas[j]
                        * inputs[k]
                    )

                neuron["bias"] += (learning_rate * hidden_deltas[j])

# Predict class for one sample
def predict(network, sample):
    outputs = forward(network, sample)
    best_class = 0
    for i in range(1, len(outputs)):
        if outputs[i] > outputs[best_class]:
            best_class = i
    return best_class

# Calculate accuracy manually
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual)

# Main program
X_train, X_test, y_train, y_test = split_data(X, y)
normalize(X_train, X_test)
network = initialize_network(inputs=4,hidden=6,outputs=3)
train(network, X_train, y_train)

predictions = []
for sample in X_test:
    predictions.append(predict(network, sample))

accuracy = calculate_accuracy(y_test, predictions)
print("Multilayer Perceptron")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Accuracy (%):", accuracy * 100)
