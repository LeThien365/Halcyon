"""
Halcyon .hnn weight file format — a small custom binary format, NOT Stockfish's
.nnue format. Mirrors src/nnue.cpp exactly:

    uint32 magic   = 0x484E4E31 ("HNN1")
    int32  inputSize  (must be 768)
    int32  hiddenSize (must be 256)
    float32[inputSize*hiddenSize]  W1   (row-major: W1[feature][hidden])
    float32[hiddenSize]            b1
    float32[2*hiddenSize]          W2   (first half = us-accumulator weights, second = them)
    float32                        b2

Kept dependency-free (numpy + struct only) so it can be unit-tested without
PyTorch, and reused by both the trainer's exporter and the self-play tooling.
"""
import struct
import numpy as np

MAGIC = 0x484E4E31
INPUT_SIZE = 768
HIDDEN_SIZE = 256


def save_hnn(path, W1, b1, W2, b2):
    W1 = np.asarray(W1, dtype=np.float32)
    b1 = np.asarray(b1, dtype=np.float32)
    W2 = np.asarray(W2, dtype=np.float32)
    assert W1.shape == (INPUT_SIZE, HIDDEN_SIZE), f"W1 shape {W1.shape}"
    assert b1.shape == (HIDDEN_SIZE,), f"b1 shape {b1.shape}"
    assert W2.shape == (2 * HIDDEN_SIZE,), f"W2 shape {W2.shape}"
    with open(path, "wb") as f:
        f.write(struct.pack("<Iii", MAGIC, INPUT_SIZE, HIDDEN_SIZE))
        f.write(np.ascontiguousarray(W1).tobytes(order="C"))
        f.write(np.ascontiguousarray(b1).tobytes())
        f.write(np.ascontiguousarray(W2).tobytes())
        f.write(struct.pack("<f", float(b2)))


def load_hnn(path):
    with open(path, "rb") as f:
        magic, in_size, hid_size = struct.unpack("<Iii", f.read(12))
        if magic != MAGIC or in_size != INPUT_SIZE or hid_size != HIDDEN_SIZE:
            raise ValueError("not a valid Halcyon .hnn file (bad header)")
        w1 = np.frombuffer(f.read(4 * INPUT_SIZE * HIDDEN_SIZE), dtype=np.float32).reshape(INPUT_SIZE, HIDDEN_SIZE).copy()
        b1 = np.frombuffer(f.read(4 * HIDDEN_SIZE), dtype=np.float32).copy()
        w2 = np.frombuffer(f.read(4 * 2 * HIDDEN_SIZE), dtype=np.float32).copy()
        (b2,) = struct.unpack("<f", f.read(4))
    return w1, b1, w2, b2


def feature_index(piece_type, square, is_own, perspective_white):
    """piece_type: 0=P,1=N,2=B,3=R,4=Q,5=K. square: 0..63, a1=0 h8=63 (python-chess convention)."""
    s = square if perspective_white else (square ^ 56)
    base = 0 if is_own else 384
    return base + piece_type * 64 + s


def board_features(board):
    """board: a python-chess Board. Returns (us_indices, them_indices) — the
    active feature indices (0/1 sparse) for the side-to-move perspective and
    the opposing perspective, respectively."""
    import chess
    white_to_move = board.turn == chess.WHITE
    us_idx, them_idx = [], []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt = piece.piece_type - 1  # chess.PAWN=1..chess.KING=6 -> 0..5
        piece_is_white = piece.color == chess.WHITE

        is_own_for_us = (piece_is_white == white_to_move)
        us_idx.append(feature_index(pt, sq, is_own_for_us, white_to_move))

        is_own_for_them = (piece_is_white != white_to_move)
        them_idx.append(feature_index(pt, sq, is_own_for_them, not white_to_move))
    return us_idx, them_idx


if __name__ == "__main__":
    # Round-trip self-test: no PyTorch required.
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, 0.05, size=(INPUT_SIZE, HIDDEN_SIZE)).astype(np.float32)
    b1 = rng.normal(0, 0.05, size=(HIDDEN_SIZE,)).astype(np.float32)
    W2 = rng.normal(0, 0.05, size=(2 * HIDDEN_SIZE,)).astype(np.float32)
    b2 = np.float32(0.25)
    save_hnn("/tmp/roundtrip_test.hnn", W1, b1, W2, b2)
    W1b, b1b, W2b, b2b = load_hnn("/tmp/roundtrip_test.hnn")
    ok = np.allclose(W1, W1b) and np.allclose(b1, b1b) and np.allclose(W2, W2b) and abs(b2 - b2b) < 1e-6
    print("round-trip OK:", ok)
