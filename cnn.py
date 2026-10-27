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
X = (x_train / 255.0).reshape(-1, 28, 28)

# One-hot encode labels
Y = np.eye(10)[y_train]

print(X.shape)  # (60000, 784)
print(Y.shape)  # (60000, 10)

df_test = pd.read_csv("mnist/mnist_test.csv", header=None)
df_test = df_test.to_numpy()

# First column = labels, remaining columns = pixels
y_test = df_test[:, 0].astype(int)
x_test = df_test[:, 1:].astype(np.float32)

# Normalize pixel values to [0, 1]
X_test = (x_test / 255.0).reshape(-1, 28, 28)

print(X_test.shape)  # (10000, 784)
print(y_test.shape)  # (10000,)

rng = np.random.default_rng(42)
lr = 0.1

# Initialisation of filters
K = rng.normal(0, 0.3, (9,8))
W = rng.normal(0, 0.01, (1352, 10))

history_steps = []
history_accuracy = []
history_W_norm = []
history_W_mean = []
history_W_std = []

win = lambda x: np.stack([x[:, a//3:a//3+26, a%3:a%3+26] for a in range(9)], -1)
up = lambda a: a.repeat(2, 1).repeat(2, 2)

for step in tqdm(range(3000)):
    i = rng.integers(0, 60000, 32)
    P = win(X[i])
    c = np.maximum(0, P @ K)
    m = c.reshape(32, 13, 2, 13, 2, 8).max((2, 4))
    z = m.reshape(32, -1) @ W
    p = np.exp(z - z.max(1, keepdims=True))
    dz = (p / p.sum(1, keepdims=True) - Y[i]) / 32
    dc = up((dz @ W.T).reshape(m.shape)) * (c == up(m)) * (c > 0)
    W -= lr * m.reshape(32, -1).T @ dz
    K -= lr * P.reshape(-1, 9).T @ dc.reshape(-1, 8)

    if (step + 1) % 50 == 0:
        # Evaluate on the test set
        c_test = np.maximum(0, win(X_test) @ K)
        m_test = c_test.reshape(-1, 13, 2, 13, 2, 8).max((2, 4))
        z_test = m_test.reshape(len(X_test), -1) @ W

        accuracy = (z_test.argmax(1) == y_test).mean()

        history_steps.append(step + 1)
        history_accuracy.append(accuracy * 100)
        history_W_norm.append(np.linalg.norm(W))
        history_W_mean.append(W.mean())
        history_W_std.append(W.std())

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Test accuracy
axes[0, 0].plot(history_steps, history_accuracy)
axes[0, 0].set_title("Test Accuracy")
axes[0, 0].set_xlabel("Training Step")
axes[0, 0].set_ylabel("Accuracy (%)")
axes[0, 0].grid(True)

# L2 norm of W
axes[0, 1].plot(history_steps, history_W_norm)
axes[0, 1].set_title("Weight Matrix W: L2 Norm")
axes[0, 1].set_xlabel("Training Step")
axes[0, 1].set_ylabel("||W||₂")
axes[0, 1].grid(True)

# Mean of W
axes[1, 0].plot(history_steps, history_W_mean)
axes[1, 0].set_title("Weight Matrix W: Mean")
axes[1, 0].set_xlabel("Training Step")
axes[1, 0].set_ylabel("Mean weight")
axes[1, 0].grid(True)

# Standard deviation of W
axes[1, 1].plot(history_steps, history_W_std)
axes[1, 1].set_title("Weight Matrix W: Standard Deviation")
axes[1, 1].set_xlabel("Training Step")
axes[1, 1].set_ylabel("Std of weights")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

print(f"Final test accuracy: {history_accuracy[-1]:.2f}%")
print(f"Final ||W||₂: {history_W_norm[-1]:.4f}")