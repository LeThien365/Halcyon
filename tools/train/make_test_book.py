import struct
import chess
import chess.polyglot as pg

def encode_move(uci_from, uci_to, promo=None):
    f = chess.parse_square(uci_from)
    t = chess.parse_square(uci_to)
    raw = (chess.square_file(t)) | (chess.square_rank(t) << 3) | \
          (chess.square_file(f) << 6) | (chess.square_rank(f) << 9)
    promo_map = {None: 0, "n": 1, "b": 2, "r": 3, "q": 4}
    raw |= promo_map[promo] << 12
    return raw

entries = []

# startpos: two candidate moves with different weights
b = chess.Board()
key = pg.zobrist_hash(b)
entries.append((key, encode_move("e2", "e4"), 50))
entries.append((key, encode_move("d2", "d4"), 30))

# after 1.e4 e5
b2 = chess.Board()
b2.push_uci("e2e4")
b2.push_uci("e7e5")
key2 = pg.zobrist_hash(b2)
entries.append((key2, encode_move("g1", "f3"), 10))

# castling test position: white can castle both sides
b3 = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
key3 = pg.zobrist_hash(b3)
entries.append((key3, encode_move("e1", "h1"), 5))   # polyglot quirk: king "captures" own rook = kingside castle

entries.sort(key=lambda e: e[0])

with open("/tmp/test_book.bin", "wb") as f:
    for k, m, w in entries:
        f.write(struct.pack(">QHHI", k, m, w, 0))

print("wrote", len(entries), "entries")
print("startpos key:", hex(key))
print("after e4 e5 key:", hex(key2))
print("castle test key:", hex(key3))
