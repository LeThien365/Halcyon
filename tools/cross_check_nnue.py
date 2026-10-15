import subprocess, sys
import numpy as np
import chess
from nnue_format import save_hnn, INPUT_SIZE, HIDDEN_SIZE, board_features

rng = np.random.default_rng(42)
W1 = rng.normal(0, 0.05, size=(INPUT_SIZE, HIDDEN_SIZE)).astype(np.float32)
b1 = rng.normal(0, 0.05, size=(HIDDEN_SIZE,)).astype(np.float32)
W2 = rng.normal(0, 0.05, size=(2 * HIDDEN_SIZE,)).astype(np.float32)
b2 = np.float32(0.1)

path = "/tmp/dummy.hnn"
save_hnn(path, W1, b1, W2, b2)


def nnue_eval_from_indices(us_idx, them_idx):
    acc_us = b1.copy()
    for i in us_idx:
        acc_us += W1[i]
    acc_us = np.maximum(acc_us, 0.0)

    acc_them = b1.copy()
    for i in them_idx:
        acc_them += W1[i]
    acc_them = np.maximum(acc_them, 0.0)

    out = b2 + acc_us @ W2[:HIDDEN_SIZE] + acc_them @ W2[HIDDEN_SIZE:]
    return int(out * 100.0)


test_positions = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                             # ep torture
]

all_ok = True
for fen in test_positions:
    board = chess.Board(fen)
    us_idx, them_idx = board_features(board)
    expected = nnue_eval_from_indices(us_idx, them_idx)

    proc = subprocess.run(
        ["./halcyon"],
        input=f"setoption name NNUEFile value {path}\nposition fen {fen}\neval\nquit\n",
        capture_output=True, text=True, timeout=10, cwd="/home/claude/halcyon"
    )
    cpp_val = None
    for line in proc.stdout.splitlines():
        if "static eval" in line:
            cpp_val = int(line.strip().split(":")[-1])

    match = (cpp_val == expected)
    all_ok &= match
    print(f"FEN: {fen}\n  python={expected}  cpp={cpp_val}  MATCH={match}")

sys.exit(0 if all_ok else 1)
