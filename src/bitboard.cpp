#include "bitboard.h"
#include <iostream>

namespace halcyon {

void print_bitboard(Bitboard bb) {
    for (int r = 7; r >= 0; r--) {
        std::cout << (r + 1) << "  ";
        for (int f = 0; f < 8; f++) {
            Square s = make_square(f, r);
            std::cout << (get_bit(bb, s) ? "1 " : ". ");
        }
        std::cout << "\n";
    }
    std::cout << "   a b c d e f g h\n";
}

} // namespace halcyon
