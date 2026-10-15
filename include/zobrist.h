#pragma once
#include "types.h"

namespace halcyon {

extern Bitboard ZOBRIST_PIECE[PIECE_NB][SQUARE_NB];
extern Bitboard ZOBRIST_CASTLE[16];
extern Bitboard ZOBRIST_EP_FILE[8];
extern Bitboard ZOBRIST_SIDE;

void init_zobrist();

} // namespace halcyon
