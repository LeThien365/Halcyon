"""
Trains the Halcyon NNUE network on self-play data and exports it to the
custom .hnn format that src/nnue.cpp loads.

Requires PyTorch (pip install torch) — this is meant to run on YOUR machine,
ideally with a GPU. It is not runnable inside the sandbox that produced this
project (no PyTorch / no GPU there), so treat this as a carefully-written
starting point to validate and extend rather than a black box.

Usage:
    python3 train_nnue.py --data selfplay_data.csv --epochs 20 --out net.hnn
    (then in the engine)  setoption name NNUEFile value net.hnn

Architecture (must match include/nnue.h / src/nnue.cpp exactly):
    768 -> shared Linear(768,256) -> ReLU     (applied once per perspective)
    concat(us, them) = 512 -> Linear(512,1)   -> score (centipawns / 100)
"""
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import chess
from nnue_format import save_hnn, INPUT_SIZE, HIDDEN_SIZE, board_features


class PositionDataset(Dataset):
    def __init__(self, csv_path, cp_clip=1000):
        self.rows = []
        with open(csv_path) as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # header
            for fen, cp, result in reader:
                cp_val = max(-cp_clip, min(cp_clip, int(cp)))
                self.rows.append((fen, cp_val, float(result)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fen, cp, result = self.rows[idx]
        board = chess.Board(fen)
        us_idx, them_idx = board_features(board)

        us_vec = np.zeros(INPUT_SIZE, dtype=np.float32)
        us_vec[us_idx] = 1.0
        them_vec = np.zeros(INPUT_SIZE, dtype=np.float32)
        them_vec[them_idx] = 1.0

        # Training target matches the C++ inference scale (raw_output * 100 = cp).
        target = cp / 100.0
        return us_vec, them_vec, np.float32(target)


class HalcyonNNUE(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.ft = nn.Linear(input_size, hidden_size)       # shared feature transformer
        self.output = nn.Linear(2 * hidden_size, 1)

    def forward(self, us, them):
        acc_us = torch.relu(self.ft(us))
        acc_them = torch.relu(self.ft(them))
        x = torch.cat([acc_us, acc_them], dim=1)
        return self.output(x).squeeze(-1)


def export(model, path):
    ft_w = model.ft.weight.detach().cpu().numpy()        # [hidden, input]
    ft_b = model.ft.bias.detach().cpu().numpy()           # [hidden]
    out_w = model.output.weight.detach().cpu().numpy()    # [1, 2*hidden]
    out_b = model.output.bias.detach().cpu().numpy()      # [1]

    W1 = ft_w.T.astype(np.float32)          # -> [input, hidden], matches nnue.cpp row-major layout
    b1 = ft_b.astype(np.float32)
    W2 = out_w.reshape(-1).astype(np.float32)  # -> [2*hidden]
    b2 = float(out_b[0])

    save_hnn(path, W1, b1, W2, b2)
    print(f"Exported trained network to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV produced by selfplay_gen.py (fen;cp;result)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="net.hnn")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    dataset = PositionDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = HalcyonNNUE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        total_loss, n_batches = 0.0, 0
        for us_vec, them_vec, target in loader:
            us_vec, them_vec, target = us_vec.to(device), them_vec.to(device), target.to(device)
            pred = model(us_vec, them_vec)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch+1}/{args.epochs}  mse={total_loss/max(1,n_batches):.4f}")

    export(model, args.out)


if __name__ == "__main__":
    main()
