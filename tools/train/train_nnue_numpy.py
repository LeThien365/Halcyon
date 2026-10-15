"""
Pure-NumPy trainer for the Halcyon NNUE network. Written specifically because
this sandbox has no PyTorch (disk-space constrained) and no GPU. Implements
the forward pass, manual backpropagation, and the Adam optimizer by hand for
this one small architecture — no autograd framework involved.

Architecture (must match include/nnue.h / src/nnue.cpp / tools/train_nnue.py):
    768 -> shared Linear(768,256) -> ReLU   (applied once per perspective)
    concat(us, them) = 512 -> Linear(512,1) -> score (centipawns / 100)

Sparse features are handled via a padded-index representation: each position
has at most MAX_ACTIVE active features per perspective (<=32, one per piece
on the board); padding uses an index pointing at an always-zero extra row of
W1 so batches can be processed as fixed-size arrays without ragged lists.

Usage:
    python3 train_nnue_numpy.py --data data.csv --epochs 10 --out net.hnn
"""
import argparse
import csv
import time

import numpy as np
import chess

from nnue_format import save_hnn, load_hnn, INPUT_SIZE, HIDDEN_SIZE, board_features

MAX_ACTIVE = 32          # max pieces ever on a board
PAD_ROW = INPUT_SIZE     # index of the extra always-zero row appended to W1


def load_dataset(csv_path, cp_clip=1000):
    us_all, them_all, targets = [], [], []
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for fen, cp, result in reader:
            board = chess.Board(fen)
            us_idx, them_idx = board_features(board)
            n = len(us_idx)  # same count for us/them by construction
            us_all.append(us_idx + [PAD_ROW] * (MAX_ACTIVE - n))
            them_all.append(them_idx + [PAD_ROW] * (MAX_ACTIVE - n))
            cp_val = max(-cp_clip, min(cp_clip, int(cp)))
            targets.append(cp_val / 100.0)
    return (np.array(us_all, dtype=np.int32),
            np.array(them_all, dtype=np.int32),
            np.array(targets, dtype=np.float32))


class Adam:
    def __init__(self, shape, lr):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.lr = lr

    def step(self, param, grad, t, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (grad * grad)
        mhat = self.m / (1 - beta1 ** t)
        vhat = self.v / (1 - beta2 ** t)
        return param - self.lr * mhat / (np.sqrt(vhat) + eps)


def forward(W1, b1, W2, b2, us_idx, them_idx):
    # W1 has an extra zero row at PAD_ROW so padded slots contribute nothing.
    z1_us = W1[us_idx].sum(axis=1) + b1          # [batch, hidden]
    a1_us = np.maximum(z1_us, 0.0)
    z1_them = W1[them_idx].sum(axis=1) + b1
    a1_them = np.maximum(z1_them, 0.0)
    concat = np.concatenate([a1_us, a1_them], axis=1)  # [batch, 2*hidden]
    pred = concat @ W2 + b2
    cache = (us_idx, them_idx, z1_us, a1_us, z1_them, a1_them, concat)
    return pred, cache


def backward(W2, pred, target, cache):
    us_idx, them_idx, z1_us, a1_us, z1_them, a1_them, concat = cache
    batch = pred.shape[0]

    dpred = 2.0 * (pred - target) / batch           # [batch]
    dW2 = concat.T @ dpred                            # [2*hidden]
    db2 = dpred.sum()

    dconcat = np.outer(dpred, W2)                     # [batch, 2*hidden]
    dA1_us, dA1_them = dconcat[:, :HIDDEN_SIZE], dconcat[:, HIDDEN_SIZE:]
    dz1_us = dA1_us * (z1_us > 0)
    dz1_them = dA1_them * (z1_them > 0)

    # Every active feature slot in a row received the full row's upstream
    # gradient in the forward pass (the accumulator is a plain sum), so each
    # of the MAX_ACTIVE index columns scatter-adds the same dz1 row.
    dW1 = np.zeros((INPUT_SIZE + 1, HIDDEN_SIZE), dtype=np.float32)
    db1 = dz1_us.sum(axis=0) + dz1_them.sum(axis=0)
    for k in range(us_idx.shape[1]):
        np.add.at(dW1, us_idx[:, k], dz1_us)
        np.add.at(dW1, them_idx[:, k], dz1_them)

    return dW1[:INPUT_SIZE], db1, dW2, db2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", default="net.hnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weights", default=None, help="resume training from an existing .hnn file")
    ap.add_argument("--cache", default=None, help="cache the parsed dataset as .npz for fast reload across resumed runs")
    args = ap.parse_args()

    print("Loading dataset ...")
    t0 = time.time()
    if args.cache:
        import os
        if os.path.exists(args.cache):
            npz = np.load(args.cache)
            us_all, them_all, targets = npz["us"], npz["them"], npz["y"]
            print(f"Loaded {len(targets)} samples from cache in {time.time()-t0:.1f}s")
        else:
            us_all, them_all, targets = load_dataset(args.data)
            np.savez(args.cache, us=us_all, them=them_all, y=targets)
            print(f"Loaded {len(targets)} samples in {time.time()-t0:.1f}s (cached to {args.cache})")
    else:
        us_all, them_all, targets = load_dataset(args.data)
        print(f"Loaded {len(targets)} samples in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    n = len(targets)
    perm = rng.permutation(n)  # lightweight index array (n int64), not a copy of the data
    n_val = max(1, int(n * args.val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    # Only the (small) validation slice is materialized as its own array;
    # training stays indexed into the original us_all/them_all/targets so we
    # never hold a second full-size copy of a multi-million-row dataset in
    # memory at once (that duplication is what caused an OOM kill at 2M rows).
    us_val, them_val, y_val = us_all[val_idx], them_all[val_idx], targets[val_idx]

    W1 = rng.normal(0, (2.0 / INPUT_SIZE) ** 0.5, size=(INPUT_SIZE + 1, HIDDEN_SIZE)).astype(np.float32)
    W1[PAD_ROW] = 0.0
    b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    W2 = rng.normal(0, (2.0 / (2 * HIDDEN_SIZE)) ** 0.5, size=(2 * HIDDEN_SIZE,)).astype(np.float32)
    b2 = np.float32(0.0)

    if args.init_weights:
        w1_loaded, b1_loaded, w2_loaded, b2_loaded = load_hnn(args.init_weights)
        W1[:INPUT_SIZE] = w1_loaded
        b1 = b1_loaded
        W2 = w2_loaded
        b2 = np.float32(b2_loaded)
        print(f"Resumed weights from {args.init_weights}")

    opt_W1 = Adam(W1[:INPUT_SIZE].shape, args.lr)
    opt_b1 = Adam(b1.shape, args.lr)
    opt_W2 = Adam(W2.shape, args.lr)
    opt_b2 = Adam((), args.lr)

    n_tr = len(tr_idx)
    step = 0
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        epoch_order = rng.permutation(n_tr)  # shuffles which indices we visit, not the data itself
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_tr, args.batch_size):
            idx = tr_idx[epoch_order[i:i + args.batch_size]]
            bu, bt, by = us_all[idx], them_all[idx], targets[idx]

            pred, cache = forward(W1, b1, W2, b2, bu, bt)
            loss = np.mean((pred - by) ** 2)
            dW1, db1, dW2, db2 = backward(W2, pred, by, cache)

            step += 1
            W1[:INPUT_SIZE] = opt_W1.step(W1[:INPUT_SIZE], dW1, step)
            b1 = opt_b1.step(b1, db1, step)
            W2 = opt_W2.step(W2, dW2, step)
            b2 = float(opt_b2.step(np.float32(b2), np.float32(db2), step))

            epoch_loss += loss
            n_batches += 1

        pred_val, val_loss = None, 0.0
        n_val_batches = 0
        for i in range(0, len(y_val), args.batch_size):
            bu, bt, by = us_val[i:i + args.batch_size], them_val[i:i + args.batch_size], y_val[i:i + args.batch_size]
            p, _ = forward(W1, b1, W2, b2, bu, bt)
            val_loss += np.mean((p - by) ** 2)
            n_val_batches += 1
        val_loss /= max(1, n_val_batches)
        print(f"epoch {epoch}/{args.epochs}  train_mse={epoch_loss/n_batches:.4f}  "
              f"val_mse={val_loss:.4f}  ({time.time()-t_epoch:.1f}s)")

    save_hnn(args.out, W1[:INPUT_SIZE], b1, W2, b2)
    print(f"Exported trained network to {args.out}")


if __name__ == "__main__":
    main()
