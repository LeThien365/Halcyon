#pragma once
#include "types.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace halcyon {

inline void   set_bit(Bitboard& bb, Square s)   { bb |= (1ULL << s); }
inline void   clear_bit(Bitboard& bb, Square s) { bb &= ~(1ULL << s); }
inline bool   get_bit(Bitboard bb, Square s)    { return (bb >> s) & 1ULL; }

inline int popcount(Bitboard bb) {
#if defined(_MSC_VER)
    return (int)__popcnt64(bb);
#else
    return __builtin_popcountll(bb);
#endif
}

inline Square lsb(Bitboard bb) {
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanForward64(&idx, bb);
    return Square(idx);
#else
    return Square(__builtin_ctzll(bb));
#endif
}

inline Square pop_lsb(Bitboard& bb) { Square s = lsb(bb); bb &= bb - 1; return s; }

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard RANK_4 = 0x00000000FF000000ULL;
constexpr Bitboard RANK_5 = 0x000000FF00000000ULL;
constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;
constexpr Bitboard NOT_FILE_A = ~FILE_A;
constexpr Bitboard NOT_FILE_H = ~FILE_H;

void print_bitboard(Bitboard bb);

} // namespace halcyon
