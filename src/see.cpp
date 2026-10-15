#include "see.h"
#include "attacks.h"
#include "bitboard.h"
#include <algorithm>

namespace halcyon {

static Bitboard attackers_to(const Position& pos, Square sq, Bitboard occ) {
    Bitboard attackers = 0;
    attackers |= PAWN_ATTACKS[BLACK][sq] & pos.pieceBB[WP];
    attackers |= PAWN_ATTACKS[WHITE][sq] & pos.pieceBB[BP];
    attackers |= KNIGHT_ATTACKS[sq] & (pos.pieceBB[WN] | pos.pieceBB[BN]);
    attackers |= KING_ATTACKS[sq] & (pos.pieceBB[WK] | pos.pieceBB[BK]);
    Bitboard bishopsQueens = pos.pieceBB[WB] | pos.pieceBB[BB] | pos.pieceBB[WQ] | pos.pieceBB[BQ];
    attackers |= get_bishop_attacks(sq, occ) & bishopsQueens;
    Bitboard rooksQueens = pos.pieceBB[WR] | pos.pieceBB[BR] | pos.pieceBB[WQ] | pos.pieceBB[BQ];
    attackers |= get_rook_attacks(sq, occ) & rooksQueens;
    return attackers;
}

int see(const Position& pos, Move m) {
    static const int PIECE_VAL[6] = {100, 325, 335, 500, 975, 20000};

    Square to = to_sq(m);
    Square from = from_sq(m);
    Piece attacker = moved_piece(m);
    Color side = pos.sideToMove;

    Bitboard occ = pos.occAll;
    Piece captured;

    if (is_ep(m)) {
        Square capSq = make_square(file_of(to), rank_of(from));
        captured = make_piece(~side, PAWN);
        occ &= ~(1ULL << capSq);
    } else {
        captured = pos.board[to];
    }

    int gain[32];
    int d = 0;
    gain[0] = (captured == NO_PIECE) ? 0 : PIECE_VAL[type_of(captured)];

    occ &= ~(1ULL << from);
    Bitboard attackers = attackers_to(pos, to, occ);

    Bitboard bishopsQueens = pos.pieceBB[WB] | pos.pieceBB[BB] | pos.pieceBB[WQ] | pos.pieceBB[BQ];
    Bitboard rooksQueens = pos.pieceBB[WR] | pos.pieceBB[BR] | pos.pieceBB[WQ] | pos.pieceBB[BQ];
    Bitboard mayXray = pos.pieceBB[WP] | pos.pieceBB[BP] | bishopsQueens | rooksQueens;

    Piece currentAttacker = attacker;
    side = ~side;

    while (true) {
        d++;
        gain[d] = PIECE_VAL[type_of(currentAttacker)] - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) break;

        attackers &= occ;
        Bitboard sideAttackers = attackers & pos.occByColor[side];
        if (!sideAttackers) break;

        Piece lva = NO_PIECE;
        Square lvaSq = NO_SQUARE;
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = sideAttackers & pos.pieceBB[make_piece(side, PieceType(pt))];
            if (bb) { lvaSq = lsb(bb); lva = make_piece(side, PieceType(pt)); break; }
        }
        occ &= ~(1ULL << lvaSq);
        // Removing a blocker can discover a slider. Recompute only sliding rays.
        if (get_bit(mayXray, lvaSq)) {
            attackers |= get_bishop_attacks(to, occ) & bishopsQueens;
            attackers |= get_rook_attacks(to, occ) & rooksQueens;
        }

        currentAttacker = lva;
        side = ~side;
    }

    while (--d) gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0];
}

} // namespace halcyon
