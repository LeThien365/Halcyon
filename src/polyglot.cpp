#include "polyglot.h"
#include "polyglot_random.h"
#include <fstream>
#include <algorithm>

namespace halcyon {

PolyglotBook g_book;

static inline int polyglot_piece_index(PieceType pt, Color c) {
    // Polyglot's own convention (independent of our internal Color enum):
    // the color pivot is 0 for Black, 1 for White.
    int pivot = (c == WHITE) ? 1 : 0;
    return int(pt) * 2 + pivot;
}

uint64_t polyglot_key(const Position& pos) {
    uint64_t key = 0;

    for (int p = 0; p < PIECE_NB; p++) {
        Piece piece = Piece(p);
        PieceType pt = type_of(piece);
        Color c = color_of(piece);
        int pieceIndex = polyglot_piece_index(pt, c);
        Bitboard bb = pos.pieceBB[piece];
        while (bb) {
            Square sq = pop_lsb(bb);
            key ^= POLYGLOT_RANDOM64[64 * pieceIndex + sq];
        }
    }

    if (pos.castling & WK_SIDE) key ^= POLYGLOT_RANDOM64[768];
    if (pos.castling & WQ_SIDE) key ^= POLYGLOT_RANDOM64[769];
    if (pos.castling & BK_SIDE) key ^= POLYGLOT_RANDOM64[770];
    if (pos.castling & BQ_SIDE) key ^= POLYGLOT_RANDOM64[771];

    if (pos.epSquare != NO_SQUARE) {
        int f = file_of(pos.epSquare);
        Color us = pos.sideToMove;
        int capRank = (us == WHITE) ? rank_of(pos.epSquare) - 1 : rank_of(pos.epSquare) + 1;
        bool canCapture = false;
        if (capRank >= 0 && capRank <= 7) {
            for (int df : {-1, 1}) {
                int cf = f + df;
                if (cf < 0 || cf > 7) continue;
                Square s = make_square(cf, capRank);
                if (pos.board[s] == make_piece(us, PAWN)) { canCapture = true; break; }
            }
        }
        if (canCapture) key ^= POLYGLOT_RANDOM64[772 + f];
    }

    if (pos.sideToMove == WHITE) key ^= POLYGLOT_RANDOM64[780];

    return key;
}

bool PolyglotBook::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { loaded = false; return false; }

    entries.clear();
    unsigned char buf[16];
    while (f.read(reinterpret_cast<char*>(buf), 16)) {
        PolyglotEntry e;
        e.key = 0;
        for (int i = 0; i < 8; i++) e.key = (e.key << 8) | buf[i];
        e.move = (uint16_t(buf[8]) << 8) | buf[9];
        e.weight = (uint16_t(buf[10]) << 8) | buf[11];
        e.learn = (uint32_t(buf[12]) << 24) | (uint32_t(buf[13]) << 16) | (uint32_t(buf[14]) << 8) | buf[15];
        entries.push_back(e);
    }
    if (entries.empty()) { loaded = false; return false; }

    std::sort(entries.begin(), entries.end(), [](const PolyglotEntry& a, const PolyglotEntry& b) {
        return a.key < b.key;
    });
    loaded = true;
    return true;
}

static Move decode_polyglot_move(const Position& pos, uint16_t raw, const MoveList& legalMoves) {
    int toFile = raw & 0x7;
    int toRank = (raw >> 3) & 0x7;
    int fromFile = (raw >> 6) & 0x7;
    int fromRank = (raw >> 9) & 0x7;
    int promo = (raw >> 12) & 0x7; // 0=none,1=N,2=B,3=R,4=Q

    Square from = make_square(fromFile, fromRank);
    Square to = make_square(toFile, toRank);

    // Polyglot's own quirk: castling is encoded as "king captures own rook".
    if (pos.board[from] != NO_PIECE && type_of(pos.board[from]) == KING) {
        if (from == E1 && to == H1) to = G1;
        else if (from == E1 && to == A1) to = C1;
        else if (from == E8 && to == H8) to = G8;
        else if (from == E8 && to == A8) to = C8;
    }

    PieceType promoPt = NO_PIECE_TYPE;
    switch (promo) {
        case 1: promoPt = KNIGHT; break;
        case 2: promoPt = BISHOP; break;
        case 3: promoPt = ROOK; break;
        case 4: promoPt = QUEEN; break;
        default: break;
    }

    for (Move m : legalMoves)
        if (from_sq(m) == from && to_sq(m) == to && promo_type(m) == promoPt)
            return m;
    return MOVE_NONE;
}

Move PolyglotBook::probe(Position& pos, std::mt19937_64& rng) const {
    if (!loaded) return MOVE_NONE;

    uint64_t key = polyglot_key(pos);
    auto lo = std::lower_bound(entries.begin(), entries.end(), key,
        [](const PolyglotEntry& e, uint64_t k) { return e.key < k; });

    std::vector<const PolyglotEntry*> matches;
    for (auto it = lo; it != entries.end() && it->key == key; ++it) matches.push_back(&*it);
    if (matches.empty()) return MOVE_NONE;

    MoveList legal;
    generate_legal_moves(pos, legal);

    // Weighted-random primary pick; if it fails to decode to a legal move,
    // fall back to trying the rest in descending weight order.
    uint64_t total = 0;
    for (auto* e : matches) total += e->weight ? e->weight : 1;
    uint64_t r = total ? (rng() % total) : 0;
    uint64_t acc = 0;
    for (auto* e : matches) {
        acc += e->weight ? e->weight : 1;
        if (r < acc) {
            Move m = decode_polyglot_move(pos, e->move, legal);
            if (m != MOVE_NONE) return m;
            break;
        }
    }

    std::vector<const PolyglotEntry*> byWeight = matches;
    std::sort(byWeight.begin(), byWeight.end(), [](const PolyglotEntry* a, const PolyglotEntry* b) {
        return a->weight > b->weight;
    });
    for (auto* e : byWeight) {
        Move m = decode_polyglot_move(pos, e->move, legal);
        if (m != MOVE_NONE) return m;
    }
    return MOVE_NONE;
}

} // namespace halcyon
