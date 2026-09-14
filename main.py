import numpy as np


input_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

target_output = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def sigmoid_derivative(value):
    sigmoid_value = sigmoid(value)
    return sigmoid_value * (1 - sigmoid_value)


np.random.seed(42)

input_to_hidden_weights = np.random.randn(2, 4) * 0.5
hidden_layer_bias = np.zeros((1, 4))

hidden_to_output_weights = np.random.randn(4, 1) * 0.5
output_layer_bias = np.zeros((1, 1))


learning_rate = 1.0
number_of_epochs = 10000


# Training
for epoch in range(number_of_epochs):

    # Forward propagation
    hidden_layer_input = (
        input_data @ input_to_hidden_weights
        + hidden_layer_bias
    )

    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = (
        hidden_layer_output @ hidden_to_output_weights
        + output_layer_bias
    )

    predicted_output = sigmoid(output_layer_input)


    # Calculate loss
    loss = np.mean(
        (predicted_output - target_output) ** 2
    )


    # Backpropagation
    output_error = predicted_output - target_output

    output_gradient = (
        output_error
        * sigmoid_derivative(output_layer_input)
    )

    hidden_to_output_weight_gradient = (
        hidden_layer_output.T @ output_gradient
    )

    output_bias_gradient = np.sum(
        output_gradient,
        axis=0,
        keepdims=True
    )


    hidden_layer_gradient = (
        output_gradient @ hidden_to_output_weights.T
    ) * sigmoid_derivative(hidden_layer_input)

    input_to_hidden_weight_gradient = (
        input_data.T @ hidden_layer_gradient
    )

    hidden_bias_gradient = np.sum(
        hidden_layer_gradient,
        axis=0,
        keepdims=True
    )


    # Update weights and biases
    hidden_to_output_weights -= (
        learning_rate
        * hidden_to_output_weight_gradient
    )

    output_layer_bias -= (
        learning_rate
        * output_bias_gradient
    )

    input_to_hidden_weights -= (
        learning_rate
        * input_to_hidden_weight_gradient
    )

    hidden_layer_bias -= (
        learning_rate
        * hidden_bias_gradient
    )


    # Training progress
    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch}, Loss: {loss:.6f}"
        )


print("\nPredictions:")

for inputs in input_data:

    hidden_layer = sigmoid(
        inputs @ input_to_hidden_weights
        + hidden_layer_bias
    )

    predicted_value = sigmoid(
        hidden_layer @ hidden_to_output_weights
        + output_layer_bias
    )

    print(
        inputs,
        "->",
        predicted_value[0],
        "->",
        round(float(predicted_value[0, 0]))
    )
