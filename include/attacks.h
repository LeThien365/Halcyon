#pragma once
#include "types.h"
#include "bitboard.h"

namespace halcyon {

extern Bitboard PAWN_ATTACKS[COLOR_NB][SQUARE_NB];
extern Bitboard KNIGHT_ATTACKS[SQUARE_NB];
extern Bitboard KING_ATTACKS[SQUARE_NB];

// Must be called once at program startup before any move generation / eval / search.
void init_attacks();

Bitboard get_bishop_attacks(Square sq, Bitboard occupancy);
Bitboard get_rook_attacks(Square sq, Bitboard occupancy);
inline Bitboard get_queen_attacks(Square sq, Bitboard occupancy) {
    return get_bishop_attacks(sq, occupancy) | get_rook_attacks(sq, occupancy);
}

} // namespace halcyon
