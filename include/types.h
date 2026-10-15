#pragma once
#include <cstdint>
#include <string>
#include <cctype>

namespace halcyon {

using Bitboard = uint64_t;
using Square   = int;
using Move     = uint32_t;

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NB = 2 };
inline Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    PIECE_TYPE_NB, NO_PIECE_TYPE = PIECE_TYPE_NB
};

enum Piece : int {
    WP = 0, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK,
    NO_PIECE = 12, PIECE_NB = 12
};

inline Color     color_of(Piece p) { return Color(p / 6); }
inline PieceType type_of(Piece p)  { return PieceType(p % 6); }
inline Piece make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }

enum Squares : int {
    A1,B1,C1,D1,E1,F1,G1,H1,
    A2,B2,C2,D2,E2,F2,G2,H2,
    A3,B3,C3,D3,E3,F3,G3,H3,
    A4,B4,C4,D4,E4,F4,G4,H4,
    A5,B5,C5,D5,E5,F5,G5,H5,
    A6,B6,C6,D6,E6,F6,G6,H6,
    A7,B7,C7,D7,E7,F7,G7,H7,
    A8,B8,C8,D8,E8,F8,G8,H8,
    NO_SQUARE = 64, SQUARE_NB = 64
};

inline int file_of(Square s) { return s & 7; }
inline int rank_of(Square s) { return s >> 3; }
inline Square make_square(int f, int r) { return Square(r * 8 + f); }

inline std::string square_to_str(Square s) {
    if (s == NO_SQUARE) return "-";
    std::string r;
    r += char('a' + file_of(s));
    r += char('1' + rank_of(s));
    return r;
}

inline Square str_to_square(const std::string& s) {
    if (s.size() < 2 || s == "-") return NO_SQUARE;
    int f = s[0] - 'a';
    int r = s[1] - '1';
    if (f < 0 || f > 7 || r < 0 || r > 7) return NO_SQUARE;
    return make_square(f, r);
}

enum CastleRight : int { WK_SIDE = 1, WQ_SIDE = 2, BK_SIDE = 4, BQ_SIDE = 8 };

// ---- Move encoding (32-bit) ----
// bits 0-5 from | 6-11 to | 12-15 moved piece | 16-19 promo piece-type
// bit 20 capture | 21 double-push | 22 en-passant | 23 castle
constexpr Move MOVE_NONE = 0;

inline Move make_move(Square from, Square to, Piece piece, PieceType promo = NO_PIECE_TYPE,
                       bool capture = false, bool doublePush = false, bool ep = false, bool castle = false) {
    return Move(from) | Move(to) << 6 | Move(piece) << 12 | Move(promo) << 16 |
           Move(capture) << 20 | Move(doublePush) << 21 | Move(ep) << 22 | Move(castle) << 23;
}
inline Square    from_sq(Move m)      { return Square(m & 0x3F); }
inline Square    to_sq(Move m)        { return Square((m >> 6) & 0x3F); }
inline Piece     moved_piece(Move m)  { return Piece((m >> 12) & 0xF); }
inline PieceType promo_type(Move m)   { return PieceType((m >> 16) & 0xF); }
inline bool      is_capture(Move m)   { return (m >> 20) & 1; }
inline bool      is_double_push(Move m){ return (m >> 21) & 1; }
inline bool      is_ep(Move m)        { return (m >> 22) & 1; }
inline bool      is_castle(Move m)    { return (m >> 23) & 1; }

inline char piece_to_char(Piece p) {
    static const char* s = "PNBRQKpnbrqk";
    if (p == NO_PIECE) return '.';
    return s[p];
}

inline std::string move_to_uci(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s = square_to_str(from_sq(m)) + square_to_str(to_sq(m));
    if (promo_type(m) != NO_PIECE_TYPE) {
        static const char pc[6] = {' ', 'n', 'b', 'r', 'q', ' '};
        s += pc[promo_type(m)];
    }
    return s;
}

} // namespace halcyon
