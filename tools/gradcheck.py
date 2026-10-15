"""Finite-difference gradient check for train_nnue_numpy.py's manual backward pass."""
import numpy as np
from train_nnue_numpy import forward, backward, INPUT_SIZE, HIDDEN_SIZE, MAX_ACTIVE, PAD_ROW

rng = np.random.default_rng(1)
batch = 6

W1 = rng.normal(0, 0.3, size=(INPUT_SIZE + 1, HIDDEN_SIZE)).astype(np.float64)
W1[PAD_ROW] = 0.0
b1 = rng.normal(0, 0.3, size=(HIDDEN_SIZE,)).astype(np.float64)
W2 = rng.normal(0, 0.3, size=(2 * HIDDEN_SIZE,)).astype(np.float64)
b2 = np.float64(0.2)

us_idx = np.array([rng.choice(INPUT_SIZE, size=MAX_ACTIVE, replace=False) for _ in range(batch)])
them_idx = np.array([rng.choice(INPUT_SIZE, size=MAX_ACTIVE, replace=False) for _ in range(batch)])
target = rng.normal(0, 1, size=(batch,)).astype(np.float64)


def loss_of(W1, b1, W2, b2):
    pred, _ = forward(W1, b1, W2, b2, us_idx, them_idx)
    return np.mean((pred - target) ** 2)


pred, cache = forward(W1, b1, W2, b2, us_idx, them_idx)
dW1, db1, dW2, db2 = backward(W2, pred, target, cache)

eps = 1e-4


def check(name, param, grad, n_samples=12):
    flat = param.reshape(-1)
    grad_flat = grad.reshape(-1)
    idxs = rng.choice(flat.shape[0], size=min(n_samples, flat.shape[0]), replace=False)
    max_err = 0.0
    for i in idxs:
        orig = flat[i]
        flat[i] = orig + eps
        l1 = loss_of(W1, b1, W2, b2)
        flat[i] = orig - eps
        l2 = loss_of(W1, b1, W2, b2)
        flat[i] = orig
        numeric = (l1 - l2) / (2 * eps)
        analytic = grad_flat[i]
        err = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-8)
        max_err = max(max_err, err)
    print(f"{name}: max relative error over {len(idxs)} samples = {max_err:.6f}")
    return max_err


e1 = check("dW1", W1[:INPUT_SIZE], dW1)
e2 = check("db1", b1, db1)
e3 = check("dW2", W2, dW2)

# b2 is a scalar
b2_p, b2_m = b2 + eps, b2 - eps
l1 = loss_of(W1, b1, W2, b2_p)
l2 = loss_of(W1, b1, W2, b2_m)
numeric_db2 = (l1 - l2) / (2 * eps)
e4 = abs(numeric_db2 - db2) / (abs(numeric_db2) + abs(db2) + 1e-8)
print(f"db2: relative error = {e4:.6f}")

worst = max(e1, e2, e3, e4)
print("WORST RELATIVE ERROR:", worst)
print("PASS" if worst < 1e-3 else "FAIL")
