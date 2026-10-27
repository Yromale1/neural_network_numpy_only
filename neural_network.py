import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

df_train = pd.read_csv("mnist/mnist_train.csv", header=None)
df_train = df_train.to_numpy()

# First column = labels, remaining columns = pixels
y_train = df_train[:, 0].astype(int)
x_train = df_train[:, 1:].astype(np.float32)

# Normalize pixel values to [0, 1]
X = x_train / 255.0

# One-hot encode labels
Y = np.eye(10)[y_train]

print(X.shape)  # (60000, 784)
print(Y.shape)  # (60000, 10)

np.random.seed(42)
lr = 0.5

# Initialisation of weight
W1, b1 = np.random.randn(784, 128) * 0.1, np.zeros(128)
W2, b2 = np.random.randn(128, 10) * 0.1, np.zeros(10)

# Initialisation of plotplt.ion()
seen = W1[:, :16].T.reshape(16, 28, 28)
previous_W1 = W1.copy()

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
images = []

for ax in axes.flat:
    im = ax.imshow(
        np.zeros((28, 28)),
        cmap="seismic",
        vmin=-0.01,
        vmax=0.01,
        interpolation="nearest"
    )
    ax.axis("off")
    images.append(im)

plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

def update_plot(W1):
    global previous_W1

    delta = W1 - previous_W1
    changes = delta[:, :16].T.reshape(16, 28, 28)

    # Adaptive contrast to highlight small changes
    vmax = max(np.percentile(np.abs(changes), 99), 1e-8)

    for j, im in enumerate(images):
        im.set_data(changes[j])
        im.set_clim(-vmax, vmax)

    previous_W1 = W1.copy()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

for step in tqdm(range(3000)):
    i = np.random.randint(0, 60000, 64)
    h = np.maximum(0, X[i] @ W1 + b1)
    e = np.exp(h @ W2 + b2)
    p = e / e.sum(1, keepdims=True)
    d2 = (p - Y[i]) / 64
    d1 = (d2 @ W2.T) * (h > 0)
    W2 -= lr * h.T @ d2
    b2 -= lr * d2.sum(0)
    W1 -= lr * X[i].T @ d1
    b1 -= lr * d1.sum(0)
    if step % 10 == 0:
        update_plot(W1)

df_test = pd.read_csv("mnist/mnist_test.csv", header=None)
df_test = df_test.to_numpy()

# First column = labels, remaining columns = pixels
y_test = df_test[:, 0].astype(int)
x_test = df_test[:, 1:].astype(np.float32)

# Normalize pixel values to [0, 1]
X_test = x_test / 255.0

print(X_test.shape)  # (10000, 784)
print(y_test.shape)  # (10000,)

h = np.maximum(0, X_test @ W1 + b1)
print(((h @ W2 + b2).argmax(1) == y_test).mean())