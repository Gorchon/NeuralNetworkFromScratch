# Create a complete from-scratch MNIST classifier (NumPy only).
# The script will:
# - Download MNIST (IDX format) directly from Yann LeCun's site the first time it runs
# - Parse the IDX files without external ML libs
# - Build a 784-128-64-10 MLP with ReLU + Softmax
# - Train with mini-batch SGD (optionally Adam)
# - Plot training curves and print test accuracy + confusion matrix
#
# Usage (after creating a venv and installing numpy/matplotlib):
#   python mnist_from_scratch.py --epochs 5 --lr 0.001 --batch_size 128 --optimizer adam
#
# NOTE: Internet is required on first run only (to download MNIST).


import os
import gzip
import math
import time
import struct
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

# -------------------------
# Dataset download & parse
# -------------------------

MNIST_URLS = {
    "train_images": "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def maybe_download_mnist(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        out = data_dir / (name + ".gz")
        if not out.exists():
            print(f"Downloading {name} ...")
            urlretrieve(url, out)
        else:
            print(f"Found {out.name}, skipping download.")

def parse_idx_images(path_gz: Path) -> np.ndarray:
    with gzip.open(path_gz, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic number {magic} for images")
        buf = f.read(rows * cols * num)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num, rows * cols)
        return data.astype(np.float32) / 255.0

def parse_idx_labels(path_gz: Path) -> np.ndarray:
    with gzip.open(path_gz, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic number {magic} for labels")
        buf = f.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels.astype(np.int64)

def load_mnist(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maybe_download_mnist(data_dir)
    X_train = parse_idx_images(data_dir / "train_images.gz")
    y_train = parse_idx_labels(data_dir / "train_labels.gz")
    X_test  = parse_idx_images(data_dir / "test_images.gz")
    y_test  = parse_idx_labels(data_dir / "test_labels.gz")
    return X_train, y_train, X_test, y_test

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

# -------------------------
# Activations & utilities
# -------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def softmax(logits: np.ndarray) -> np.ndarray:
    # stable softmax across rows (batch dimension is 0)
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(probs: np.ndarray, y_true_oh: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(np.sum(y_true_oh * np.log(probs), axis=1)))

def accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = probs.argmax(axis=1)
    return float(np.mean(y_pred == y_true))

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    M = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M

# -------------------------
# MLP from scratch
# -------------------------

class MLP:
    def __init__(self, layer_sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He for ReLU hidden layers; Xavier for last layer (approx with same formula if ReLU everywhere)
            std = np.sqrt(2.0 / fan_in) if i < len(layer_sizes) - 2 else np.sqrt(1.0 / fan_in)
            self.W.append(rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float32))
            self.b.append(np.zeros((1, fan_out), dtype=np.float32))

    def forward(self, X):
        # caches for backprop
        A = X
        pre_acts = []
        acts = [X]

        # Hidden layers (ReLU)
        for i in range(len(self.W) - 1):
            Z = A @ self.W[i] + self.b[i]   # (N, H)
            A = relu(Z)                     # (N, H)
            pre_acts.append(Z); acts.append(A)

        # Output layer (logits -> softmax)
        ZL = A @ self.W[-1] + self.b[-1]    # (N, 10)
        P = softmax(ZL)                     # (N, 10)
        pre_acts.append(ZL); acts.append(P)
        cache = (acts, pre_acts)
        return P, cache

    def backward(self, cache, X, y_oh, l2=0.0):
        acts, pre_acts = cache
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        N = X.shape[0]
        # Output gradient: dL/dZL = (P - Y)/N (softmax + CE combo)
        P = acts[-1]
        dZ = (P - y_oh) / N  # (N, 10)

        # Last layer grads
        A_prev = acts[-2]    # last hidden activation
        grads_W[-1] = A_prev.T @ dZ + l2 * self.W[-1]
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

        # Backprop through hidden layers
        for i in reversed(range(len(self.W) - 1)):
            # dA = dZ @ W^T
            dA = dZ @ self.W[i + 1].T
            # dZ = dA * relu'(Z)
            Z = pre_acts[i]
            dZ = dA * relu_grad(Z)

            A_prev = acts[i]  # activation before layer i (input for layer i)
            grads_W[i] = A_prev.T @ dZ + l2 * self.W[i]
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)

        return grads_W, grads_b

    def step(self, grads_W, grads_b, lr, optimizer_state=None, optimizer="sgd"):
        if optimizer == "sgd":
            for i in range(len(self.W)):
                self.W[i] -= lr * grads_W[i]
                self.b[i] -= lr * grads_b[i]
        elif optimizer == "adam":
            if optimizer_state is None:
                optimizer_state = init_adam_state(self.W, self.b)
            adam_step(self.W, self.b, grads_W, grads_b, optimizer_state, lr)
            return optimizer_state
        else:
            raise ValueError("Unknown optimizer")
        return optimizer_state

# -------------------------
# Adam optimizer (from scratch)
# -------------------------

def init_adam_state(W_list, b_list):
    state = {"mw": [], "vw": [], "mb": [], "vb": [], "t": 0}
    for W, b in zip(W_list, b_list):
        state["mw"].append(np.zeros_like(W))
        state["vw"].append(np.zeros_like(W))
        state["mb"].append(np.zeros_like(b))
        state["vb"].append(np.zeros_like(b))
    return state

def adam_step(W_list, B_list, gW_list, gB_list, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    for i in range(len(W_list)):
        # Weights
        state["mw"][i] = beta1 * state["mw"][i] + (1 - beta1) * gW_list[i]
        state["vw"][i] = beta2 * state["vw"][i] + (1 - beta2) * (gW_list[i] ** 2)
        mw_hat = state["mw"][i] / (1 - beta1 ** t)
        vw_hat = state["vw"][i] / (1 - beta2 ** t)
        W_list[i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)

        # Biases
        state["mb"][i] = beta1 * state["mb"][i] + (1 - beta1) * gB_list[i]
        state["vb"][i] = beta2 * state["vb"][i] + (1 - beta2) * (gB_list[i] ** 2)
        mb_hat = state["mb"][i] / (1 - beta1 ** t)
        vb_hat = state["vb"][i] / (1 - beta2 ** t)
        B_list[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

# -------------------------
# Training loop
# -------------------------

def iterate_minibatches(X, y, batch_size, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, lr=1e-3, batch_size=128, l2=0.0, optimizer="adam"):
    y_train_oh = one_hot(y_train, 10)
    y_val_oh = one_hot(y_val, 10)

    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    opt_state = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        # Training epoch
        losses = []
        accs = []
        for xb, yb in iterate_minibatches(X_train, y_train, batch_size):
            yb_oh = one_hot(yb, 10)
            probs, cache = model.forward(xb)
            loss = cross_entropy(probs, yb_oh) + 0.5 * l2 * sum(np.sum(W * W) for W in model.W)
            losses.append(loss)
            accs.append(accuracy(probs, yb))

            gW, gB = model.backward(cache, xb, yb_oh, l2=l2)
            opt_state = model.step(gW, gB, lr, optimizer_state=opt_state, optimizer=optimizer)

        # Validation
        probs_val, _ = model.forward(X_val)
        val_loss = cross_entropy(probs_val, y_val_oh) + 0.5 * l2 * sum(np.sum(W * W) for W in model.W)
        val_acc = accuracy(probs_val, y_val)

        history["loss"].append(float(np.mean(losses)))
        history["acc"].append(float(np.mean(accs)))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        dt = time.time() - t0
        print(f"Epoch {ep:02d}/{epochs} | loss {history['loss'][-1]:.4f} | acc {history['acc'][-1]*100:5.2f}% | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc*100:5.2f}% | {dt:.1f}s")

    return history

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="MNIST MLP from scratch (NumPy only)")
    parser.add_argument("--data_dir", type=str, default="data", help="Where to store MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden", type=str, default="128,64", help="Comma-separated hidden sizes")
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting (e.g., on headless servers)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    # Split a validation set from train (e.g., 10k)
    val_count = 10_000
    X_val, y_val = X_train[-val_count:], y_train[-val_count:]
    X_tr,  y_tr  = X_train[:-val_count], y_train[:-val_count]

    # Architecture
    hidden_sizes = [int(h) for h in args.hidden.split(",") if h.strip()]
    layer_sizes = [784] + hidden_sizes + [10]
    print("Architecture:", layer_sizes)

    model = MLP(layer_sizes, seed=42)
    hist = train_model(model, X_tr, y_tr, X_val, y_val,
                       epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                       l2=args.l2, optimizer=args.optimizer)

    # Evaluate on test set
    probs_test, _ = model.forward(X_test)
    test_acc = accuracy(probs_test, y_test)
    y_pred = probs_test.argmax(axis=1)
    M = confusion_matrix(y_test, y_pred, 10)

    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    print("Confusion matrix (rows=true, cols=pred):\n", M)

    if not args.no_plots:
        # Plot loss
        plt.figure()
        plt.plot(np.arange(1, len(hist["loss"]) + 1), hist["loss"], label="train")
        plt.plot(np.arange(1, len(hist["val_loss"]) + 1), hist["val_loss"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (CE)")
        plt.title("MNIST MLP: Loss")
        plt.legend()
        plt.show()

        # Plot accuracy
        plt.figure()
        plt.plot(np.arange(1, len(hist["acc"]) + 1), hist["acc"], label="train")
        plt.plot(np.arange(1, len(hist["val_acc"]) + 1), hist["val_acc"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("MNIST MLP: Accuracy")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()



# Mostrar algunos errores comunes
import matplotlib.pyplot as plt
err_idx = np.where(y_pred != y_test)[0][:25]  # primeros 25 errores
if err_idx.size > 0:
    cols = 5
    rows = int(np.ceil(err_idx.size / cols))
    plt.figure(figsize=(10, 2*rows))
    for i, idx in enumerate(err_idx):
        plt.subplot(rows, cols, i+1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"t:{y_test[idx]} p:{y_pred[idx]}")
        plt.axis("off")
    plt.suptitle("Ejemplos mal clasificados")
    plt.show()
