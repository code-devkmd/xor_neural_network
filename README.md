# XOR Neural Network from Scratch

A small neural network built from scratch using **Python and NumPy** to learn the XOR problem.

I made this project to understand what actually happens inside a neural network instead of relying on a machine-learning library to do everything for me.

## What it does

The network learns the XOR logic:

| Input | Expected Output |
| ----- | --------------- |
| `0 0` | `0`             |
| `0 1` | `1`             |
| `1 0` | `1`             |
| `1 1` | `0`             |

It uses a simple network with:

```text
Input Layer
    ↓
Hidden Layer (4 neurons)
    ↓
Output Layer (1 neuron)
```

## How it works

The network is trained using:

* **Sigmoid activation**
* **Forward propagation**
* **Mean Squared Error (MSE)**
* **Backpropagation**
* **Gradient descent**

The weights start with random values, and the network gradually adjusts them during training to reduce the error.

## Requirements

* Python 3
* NumPy

Install NumPy with:

```bash
pip install numpy
```

## Run the project

Clone the repository:

```bash
git clone https://github.com/code-devkmd/neural_network
cd neural_network
```

Then run:

```bash
python main.py
```

You should see the loss decreasing during training, followed by predictions similar to:

```text
Predictions:

[0. 0.] -> [0.0...] -> 0
[0. 1.] -> [0.9...] -> 1
[1. 0.] -> [0.9...] -> 1
[1. 1.] -> [0.0...] -> 0
```

The exact values may vary depending on the weights and training settings.

## Why XOR?

XOR is a simple problem, but it cannot be solved by a single linear neuron.

This makes it a good example for understanding why neural networks need hidden layers and how those layers help the network learn non-linear patterns.

## Project structure

```text
.
├── main.py
└── README.md
```

## What I learned

While making this, I got a better understanding of:

* How neural-network weights and biases work
* What happens during forward propagation
* How errors are propagated backward
* How gradients are used to update weights
* Why hidden layers are useful
* How NumPy can be used to build a neural network without a framework

## Tech used

**Python**
**NumPy**

## Note

This is mainly a learning project. The goal wasn't to build a production-ready neural-network library, but to understand the basic mechanics behind training a neural network.
