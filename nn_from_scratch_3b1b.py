# We'll implement a tiny neural network from scratch (NumPy only),
# mirroring the core ideas from 3Blue1Brown's neural network series:
# - A 2-2-1 network trained on XOR using gradient descent
# - Binary cross-entropy loss with sigmoid activation
# - Fully vectorized forward and backward passes
# - Clear math-first comments and readable code
#
# We'll also generate two plots:
# (1) Loss vs epochs
# (2) Decision boundary for the trained model
#
# Finally, we will save this code to /mnt/data so you can download it. 

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utilities & activations
# -----------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation σ(z) = 1 / (1 + e^{-z}).
    Interpretable as "squashing" pre-activations into probabilities in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime_from_activation(a: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid wrt pre-activation z, using a = σ(z):
        σ'(z) = a * (1 - a)
    """
    return a * (1.0 - a)

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Binary cross-entropy (BCE) over a batch of size m:
        L = -(1/m) * Σ_i [ y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i) ]
    The gradient wrt the logits when using a sigmoid output simplifies to (ŷ - y).
    We clip to avoid log(0).
    """
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    m = y_true.shape[1]  # using shape (1, m) for labels
    return float(-(1.0 / m) * np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))

# -----------------------------
# Model definition (2-2-1)
# -----------------------------

class TwoLayerNN:
    """
    Two-layer neural network for binary classification:
        Input x ∈ R^{2}
        Hidden layer h ∈ R^{H} with elementwise sigmoid
        Output ŷ ∈ (0,1) with sigmoid
    
    Forward pass (vectorized over a batch of m examples):
        z1 = W1 x + b1        with W1 ∈ R^{H×2},  b1 ∈ R^{H×1}
        a1 = σ(z1)
        z2 = W2 a1 + b2       with W2 ∈ R^{1×H},  b2 ∈ R^{1×1}
        ŷ  = σ(z2)
    
    Loss: Binary cross-entropy L(y, ŷ).
    
    Backprop (using BCE + sigmoid in output => δ2 = ŷ - y):
        δ2 = ŷ - y                     ∈ R^{1×m}
        dW2 = (1/m) * δ2 a1^T          ∈ R^{1×H}
        db2 = (1/m) * Σ_i δ2_i         ∈ R^{1×1}
        δ1 = (W2^T δ2) ⊙ σ'(z1)        ∈ R^{H×m}
        dW1 = (1/m) * δ1 x^T           ∈ R^{H×2}
        db1 = (1/m) * Σ_i δ1_i         ∈ R^{H×1}
    """
    def __init__(self, hidden_size=4, seed=0):
        rng = np.random.default_rng(seed)
        # Heuristic initialization: small random values
        self.W1 = rng.normal(0.0, 1.0, size=(hidden_size, 2)) * 0.5
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = rng.normal(0.0, 1.0, size=(1, hidden_size)) * 0.5
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        """
        X: shape (2, m)
        Returns (cache, Yhat)
        cache stores intermediates for backprop.
        """
        Z1 = self.W1 @ X + self.b1  # (H, m)
        A1 = sigmoid(Z1)            # (H, m)
        Z2 = self.W2 @ A1 + self.b2 # (1, m)
        Yhat = sigmoid(Z2)          # (1, m)
        cache = (X, Z1, A1, Z2, Yhat)
        return cache, Yhat

    def backward(self, cache, Y, l2=0.0):
        """
        Compute gradients via backprop.
        Y: shape (1, m)
        l2: optional L2 regularization coefficient
        
        Returns grads dict.
        """
        X, Z1, A1, Z2, Yhat = cache
        m = X.shape[1]
        
        # Output layer error (BCE + sigmoid => δ2 = ŷ - y)
        delta2 = (Yhat - Y)  # (1, m)

        # Gradients for W2, b2
        dW2 = (1.0 / m) * (delta2 @ A1.T)  # (1, H)
        db2 = (1.0 / m) * np.sum(delta2, axis=1, keepdims=True)  # (1,1)

        # Backprop to hidden
        delta1 = (self.W2.T @ delta2) * sigmoid_prime_from_activation(A1)  # (H, m)

        # Gradients for W1, b1
        dW1 = (1.0 / m) * (delta1 @ X.T)  # (H, 2)
        db1 = (1.0 / m) * np.sum(delta1, axis=1, keepdims=True)  # (H,1)

        # Optional L2 regularization
        if l2 > 0.0:
            dW2 += (l2 / m) * self.W2
            dW1 += (l2 / m) * self.W1

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads, lr=1e-1):
        """Gradient descent parameter update."""
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def predict(self, X, threshold=0.5):
        """Return class predictions in {0,1} based on threshold on ŷ."""
        _, Yhat = self.forward(X)
        return (Yhat >= threshold).astype(np.float64)

# -----------------------------
# Dataset: XOR
# -----------------------------

def make_xor():
    """
    XOR in 2D:
      x in {(0,0), (0,1), (1,0), (1,1)}, y in {0,1,1,0}
    We'll duplicate points with tiny noise for a slightly richer batch.
    """
    base_X = np.array([[0,0,1,1],
                       [0,1,0,1]], dtype=np.float64)  # shape (2, 4)
    base_Y = np.array([[0,1,1,0]], dtype=np.float64)  # shape (1, 4)

    # Expand dataset by sampling each base point with small Gaussian noise
    rng = np.random.default_rng(42)
    reps = 200  # 4 * 200 = 800 samples
    X_list = []
    Y_list = []
    for i in range(base_X.shape[1]):
        x0 = base_X[:, [i]]  # (2,1)
        y0 = base_Y[:, [i]]  # (1,1)
        noise = rng.normal(0.0, 0.05, size=(2, reps))
        X_list.append((x0 @ np.ones((1, reps))) + noise)  # broadcast repeat
        Y_list.append(y0 @ np.ones((1, reps)))
    X = np.concatenate(X_list, axis=1)  # (2, 800)
    Y = np.concatenate(Y_list, axis=1)  # (1, 800)
    return X, Y

# -----------------------------
# Training loop
# -----------------------------

def train_xor(hidden_size=4, lr=0.5, epochs=5000, l2=0.0, seed=0):
    X, Y = make_xor()
    model = TwoLayerNN(hidden_size=hidden_size, seed=seed)
    losses = []

    for ep in range(epochs):
        cache, Yhat = model.forward(X)
        loss = binary_cross_entropy(Y, Yhat)
        losses.append(loss)
        grads = model.backward(cache, Y, l2=l2)
        model.step(grads, lr=lr)

        # Optional: small learning rate decay for stability
        # if (ep+1) % 1000 == 0:
        #     lr *= 0.9

        if (ep + 1) % 500 == 0:
            # Print every 500 epochs
            print(f"Epoch {ep+1:4d} | Loss: {loss:.4f}")

    # Final accuracy
    # Evaluate predictions on the noiseless XOR base points for clarity
    base_X = np.array([[0,0,1,1],
                       [0,1,0,1]], dtype=np.float64)
    base_Y = np.array([[0,1,1,0]], dtype=np.float64)
    preds = model.predict(base_X, threshold=0.5)
    acc = float(np.mean(preds == base_Y))

    return model, np.array(losses), (base_X, base_Y, preds)

# -----------------------------
# Run training and visualize
# -----------------------------

model, losses, eval_triplet = train_xor(hidden_size=4, lr=0.5, epochs=4000, l2=0.0, seed=1)
base_X, base_Y, preds = eval_triplet

# (1) Plot loss curve
plt.figure()
plt.plot(np.arange(1, losses.size+1), losses)
plt.xlabel("Época")
plt.ylabel("Pérdida (BCE)")
plt.title("Entrenando XOR: Pérdida vs Épocas")
plt.show()

# (2) Decision boundary visualization
#    We create a grid over [−0.25, 1.25]^2 and color by predicted class.
xmin, xmax, ymin, ymax = -0.25, 1.25, -0.25, 1.25
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])  # (2, 200*200)
probs = model.forward(grid)[1].reshape(xx.shape)

plt.figure()
# Contourf for probability (default colormap; we don't specify colors)
cs = plt.contourf(xx, yy, probs, levels=50, alpha=0.8)
# Scatter XOR base points (defaults for marker colors)
plt.scatter(base_X[0, base_Y[0]==0], base_X[1, base_Y[0]==0], marker='o', label='Clase 0')
plt.scatter(base_X[0, base_Y[0]==1], base_X[1, base_Y[0]==1], marker='s', label='Clase 1')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Frontera de decisión aprendida (XOR)")
plt.legend()
plt.show()

# Print simple evaluation
print("Base points predictions (XOR):")
for i in range(base_X.shape[1]):
    x = base_X[:, i]
    y_true = int(base_Y[0, i])
    y_pred = int(preds[0, i])
    print(f"  x=({x[0]:.0f},{x[1]:.0f})  y_true={y_true}  y_pred={y_pred}")

# -----------------------------
# Save the code to a .py file
# -----------------------------
code_str = r'''import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime_from_activation(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    m = y_true.shape[1]
    return float(-(1.0 / m) * np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))

class TwoLayerNN:
    def __init__(self, hidden_size=4, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0, size=(hidden_size, 2)) * 0.5
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = rng.normal(0.0, 1.0, size=(1, hidden_size)) * 0.5
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        Z1 = self.W1 @ X + self.b1
        A1 = sigmoid(Z1)
        Z2 = self.W2 @ A1 + self.b2
        Yhat = sigmoid(Z2)
        cache = (X, Z1, A1, Z2, Yhat)
        return cache, Yhat

    def backward(self, cache, Y, l2=0.0):
        X, Z1, A1, Z2, Yhat = cache
        m = X.shape[1]
        delta2 = (Yhat - Y)
        dW2 = (1.0 / m) * (delta2 @ A1.T)
        db2 = (1.0 / m) * np.sum(delta2, axis=1, keepdims=True)
        delta1 = (self.W2.T @ delta2) * sigmoid_prime_from_activation(A1)
        dW1 = (1.0 / m) * (delta1 @ X.T)
        db1 = (1.0 / m) * np.sum(delta1, axis=1, keepdims=True)
        if l2 > 0.0:
            dW2 += (l2 / m) * self.W2
            dW1 += (l2 / m) * self.W1
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads, lr=1e-1):
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

    def predict(self, X, threshold=0.5):
        _, Yhat = self.forward(X)
        return (Yhat >= threshold).astype(np.float64)

def make_xor():
    base_X = np.array([[0,0,1,1],
                       [0,1,0,1]], dtype=np.float64)
    base_Y = np.array([[0,1,1,0]], dtype=np.float64)
    rng = np.random.default_rng(42)
    reps = 200
    X_list = []
    Y_list = []
    for i in range(base_X.shape[1]):
        x0 = base_X[:, [i]]
        y0 = base_Y[:, [i]]
        noise = rng.normal(0.0, 0.05, size=(2, reps))
        X_list.append((x0 @ np.ones((1, reps))) + noise)
        Y_list.append(y0 @ np.ones((1, reps)))
    X = np.concatenate(X_list, axis=1)
    Y = np.concatenate(Y_list, axis=1)
    return X, Y

def train_xor(hidden_size=4, lr=0.5, epochs=4000, l2=0.0, seed=1):
    X, Y = make_xor()
    model = TwoLayerNN(hidden_size=hidden_size, seed=seed)
    losses = []
    for ep in range(epochs):
        cache, Yhat = model.forward(X)
        loss = binary_cross_entropy(Y, Yhat)
        losses.append(loss)
        grads = model.backward(cache, Y, l2=l2)
        model.step(grads, lr=lr)
        if (ep + 1) % 500 == 0:
            print(f"Epoch {ep+1:4d} | Loss: {loss:.4f}")
    base_X = np.array([[0,0,1,1],
                       [0,1,0,1]], dtype=np.float64)
    base_Y = np.array([[0,1,1,0]], dtype=np.float64)
    preds = model.predict(base_X, threshold=0.5)
    acc = float(np.mean(preds == base_Y))
    return model, np.array(losses), (base_X, base_Y, preds), acc

if __name__ == "__main__":
    model, losses, eval_triplet, acc = train_xor()
    base_X, base_Y, preds = eval_triplet
    print(f"Accuracy on base XOR points: {acc*100:.1f}%")
    # Loss plot
    plt.figure()
    plt.plot(np.arange(1, losses.size+1), losses)
    plt.xlabel("Época")
    plt.ylabel("Pérdida (BCE)")
    plt.title("Entrenando XOR: Pérdida vs Épocas")
    plt.show()
    # Decision boundary
    xmin, xmax, ymin, ymax = -0.25, 1.25, -0.25, 1.25
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    probs = model.forward(grid)[1].reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, probs, levels=50, alpha=0.8)
    plt.scatter(base_X[0, base_Y[0]==0], base_X[1, base_Y[0]==0], marker='o', label='Clase 0')
    plt.scatter(base_X[0, base_Y[0]==1], base_X[1, base_Y[0]==1], marker='s', label='Clase 1')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Frontera de decisión aprendida (XOR)")
    plt.legend()
    plt.show()'''

with open("/mnt/data/nn_from_scratch_3b1b.py", "w", encoding="utf-8") as f:
    f.write(code_str)

"/mnt/data/nn_from_scratch_3b1b.py"
