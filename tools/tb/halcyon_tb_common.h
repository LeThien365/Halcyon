#pragma once
#include "types.h"   // halcyon::Color, WHITE/BLACK -- reused as-is, it's just an enum
#include <cstdint>

namespace halcyon { namespace tb {

// A bare (square-only) state for a 3-piece K+X vs K table. Side to move is
// tracked separately (as an index dimension), not stored in the struct.
struct State3 {
    int wk, bk, x;
};

enum PieceKind : int { PK_QUEEN = 0, PK_ROOK, PK_BISHOP, PK_KNIGHT, PK_NB };

inline char piece_kind_letter(PieceKind k) {
    switch (k) { case PK_QUEEN: return 'Q'; case PK_ROOK: return 'R'; case PK_BISHOP: return 'B'; case PK_KNIGHT: return 'N'; default: return '?'; }
}

// Result from the perspective of the side to move in that state.
enum class Result : uint8_t { UNRESOLVED = 0, WIN = 1, LOSS = 2, DRAW = 3, ILLEGAL = 4 };

} } // namespace halcyon::tb
