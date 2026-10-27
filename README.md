# MNIST Neural Network from Scratch

A simple neural network built from scratch using **NumPy**, trained on the MNIST handwritten digit dataset. The project implements forward propagation, backpropagation, mini-batch gradient descent, and live visualization of how the first-layer weights change during training.

## Overview

The network learns to classify handwritten digits from 0 to 9 using the MNIST dataset.

It uses:
- **NumPy** for matrix operations and neural network computations.
- **Pandas** to load the MNIST CSV files.
- **Matplotlib** to visualize weight updates in real time.
- **tqdm** to display a training progress bar.

No deep learning framework is used.

## Architecture

The network is a fully connected neural network with one hidden layer.

```text
Input layer         Hidden layer        Output layer
  784 neurons  →    128 neurons    →    10 neurons
                    ReLU activation     Logits
```

| Component | Shape | Description |
|---|---|---|
| Input | 784 | Flattened 28 × 28 grayscale image |
| Hidden layer | 128 | Fully connected layer with ReLU |
| Output layer | 10 | One score per digit class |
| Loss | — | Cross-entropy |
| Optimizer | — | Mini-batch gradient descent |

The output scores are converted into probabilities using softmax during training. Predictions are obtained by taking the class with the highest score.

## Dataset

The project uses the MNIST dataset in CSV format:

```text
mnist/
├── mnist_train.csv
└── mnist_test.csv
```

Each CSV row contains:

- The first column: the digit label, from 0 to 9.
- The remaining 784 columns: pixel intensities ranging from 0 to 255.

The training dataset contains 60,000 images, and the test dataset contains 10,000 images.

Pixel values are normalized to the range [0, 1], and training labels are converted to one-hot vectors.

## Installation

Use Python 3 and install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the MNIST CSV files:

```bash
python download_dataset.py
```

## Running the project

Run the Python script from the project root:

```bash
python neural_network.py
```

During training, a Matplotlib window displays the evolution of the first 16 hidden neurons' weight vectors.

## Training configuration

The current implementation uses the following settings:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Learning rate | 0.5 |
| Mini-batch size | 64 |
| Training steps | 3,000 |
| Hidden neurons | 128 |
| Input features | 784 |
| Output classes | 10 |
| Visualization interval | Every 10 steps |

Weights are initialized with small random values, while biases are initialized to zero.

## Live weight visualization

The visualization tracks changes in the first-layer weight matrix, `W1`.

Each of the 16 images represents the weights connecting the 784 input pixels to one hidden neuron, reshaped into a 28 × 28 grid.

Rather than displaying the absolute weights, the plot displays the difference between consecutive updates:

\[
\Delta W_1 = W_1^{(t)} - W_1^{(t-1)}
\]

The `seismic` colormap highlights changes around zero:

- **Red:** positive weight changes.
- **Blue:** negative weight changes.
- **White:** little or no change.

The color scale adapts to the 99th percentile of the absolute weight changes, making small updates easier to see.

The visualization updates every 10 training steps, allowing the learning dynamics to be observed without recreating the figure at every iteration.

## Training process

For each mini-batch, the network performs the following steps:

1. **Forward propagation:** compute hidden activations using ReLU.
2. **Softmax:** convert output logits into class probabilities.
3. **Backpropagation:** calculate gradients for the output and hidden layers.
4. **Parameter update:** update weights and biases using gradient descent.
5. **Visualization:** refresh the weight-change plot periodically.

The model is trained using randomly sampled mini-batches from the training dataset.

## Evaluation

After training, the model predicts the digit class for every test image. Accuracy is calculated as the fraction of correct predictions:

\[
\text{Accuracy} =
\frac{\text{Number of correct predictions}}
{\text{Total number of test images}}
\]

The script prints the resulting test accuracy as a value between 0 and 1. For example, `0.95` corresponds to 95% accuracy.

## Project structure

```text
.
├── download_dataset.py
├── neural_network.py
├── README.md
└── requirements.txt
```

## Possible improvements

- Track training loss and accuracy over time.
- Add a separate visualization of the absolute learned weights.
- Compare different learning rates and mini-batch sizes.
- Save and reload trained model parameters.
- Add validation data and early stopping.
- Experiment with different hidden-layer sizes and activation functions.

## Goal

The goal of this project is to understand how a neural network learns through matrix operations and gradient descent, while making the changes to its internal weights visible during training.
