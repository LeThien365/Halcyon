#include "zobrist.h"
#include <random>

namespace halcyon {

Bitboard ZOBRIST_PIECE[PIECE_NB][SQUARE_NB];
Bitboard ZOBRIST_CASTLE[16];
Bitboard ZOBRIST_EP_FILE[8];
Bitboard ZOBRIST_SIDE;

void init_zobrist() {
    std::mt19937_64 rng(20260917ULL);
    for (int p = 0; p < PIECE_NB; p++)
        for (int s = 0; s < SQUARE_NB; s++)
            ZOBRIST_PIECE[p][s] = rng();
    for (int i = 0; i < 16; i++) ZOBRIST_CASTLE[i] = rng();
    for (int i = 0; i < 8; i++)  ZOBRIST_EP_FILE[i] = rng();
    ZOBRIST_SIDE = rng();
}

} // namespace halcyon
