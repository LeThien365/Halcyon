#include "eval.h"
#include "tablebase.h"
#include "attacks.h"
#include "bitboard.h"
#include "nnue.h"
#include <algorithm>
#include <cstdlib>

// Every improvement added on top of the original evaluation can be switched
// off individually at build time (-DEVAL_DISABLE_MASK=<bits>), so each one can
// be A/B-tested in isolation instead of trusting that a bundle of changes is
// better as a whole. 0 (default) = everything on.
//   1  safe+centred mobility      2  blocked/free passed pawns
//   4  unstoppable passed pawn    8  king-attack weight needs a queen
//  16  drawish-ending scaling    32  fifty-move damping
#ifndef EVAL_DISABLE_MASK
#define EVAL_DISABLE_MASK 0
#endif
#define EVAL_ON(bit) (((EVAL_DISABLE_MASK) & (bit)) == 0)

namespace halcyon {

// Piece-square tables, written from White's point of view with rank 8 as the
// top row (index 0) and rank 1 as the bottom row (index 63), mirrored for Black.
// Piece values + PSTs tuned toward PeSTO / modern HCE (MG/EG tapered).
// Values in centipawns; board orientation: rank 8 at index 0 (White's view).
static const int PAWN_MG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
};
static const int PAWN_EG[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};
static const int KNIGHT_MG[64] = {
   -167, -89, -34, -49,  61, -97, -15,-107,
    -73, -41,  72,  36,  23,  62,   7, -17,
    -47,  60,  37,  65,  84, 129,  73,  44,
     -9,  17,  19,  53,  37,  69,  18,  22,
    -13,   4,  16,  13,  28,  19,  21,  -8,
    -23,  -9,  12,  10,  19,  17,  25, -16,
    -29, -53, -12,  -3,  -1,  18, -14, -19,
   -105, -21, -58, -33, -17, -28, -19, -23,
};
static const int KNIGHT_EG[64] = {
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
};
static const int BISHOP_MG[64] = {
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
};
static const int BISHOP_EG[64] = {
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
};
static const int ROOK_MG[64] = {
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
};
static const int ROOK_EG[64] = {
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
};
static const int QUEEN_MG[64] = {
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
};
static const int QUEEN_EG[64] = {
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
};
static const int KING_MG[64] = {
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
};
static const int KING_EG[64] = {
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
};

static const int* MG_TABLE[6] = { PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG };
static const int* EG_TABLE[6] = { PAWN_EG, KNIGHT_EG, BISHOP_EG, ROOK_EG, QUEEN_EG, KING_EG };

// PeSTO-style material (tapered).
static const int PIECE_VALUE_MG[6] = {  82, 337, 365, 477, 1025, 0 };
static const int PIECE_VALUE_EG[6] = {  94, 281, 297, 512,  936, 0 };
static const int PHASE_WEIGHT[6]   = { 0, 1, 1, 2, 4, 0 };
static const int MAX_PHASE = 24;
static const int TEMPO = 30;

static inline int pst_index(Square sq, Color c) {
    int f = file_of(sq), r = rank_of(sq);
    return (c == WHITE) ? (7 - r) * 8 + f : r * 8 + f;
}

// All squares attacked by a set of pawns of color `c`.
static inline Bitboard pawn_attack_bb(Bitboard pawns, Color c) {
    if (c == WHITE) return ((pawns & NOT_FILE_A) << 7) | ((pawns & NOT_FILE_H) << 9);
    return ((pawns & NOT_FILE_H) >> 7) | ((pawns & NOT_FILE_A) >> 9);
}

// Non-pawn, non-king attackers of `sq` belonging to color `by` (used for a
// lightweight static "is this piece adequately defended/attacked" check --
// not a full SEE, just attacker/defender counting for the threats term).
static int count_attackers(const Position& pos, Square sq, Color by) {
    int n = 0;
    if (PAWN_ATTACKS[~by][sq] & pos.pieceBB[make_piece(by, PAWN)]) n++;
    n += popcount(KNIGHT_ATTACKS[sq] & pos.pieceBB[make_piece(by, KNIGHT)]);
    n += popcount(KING_ATTACKS[sq] & pos.pieceBB[make_piece(by, KING)]);
    Bitboard bishopsQueens = pos.pieceBB[make_piece(by, BISHOP)] | pos.pieceBB[make_piece(by, QUEEN)];
    n += popcount(get_bishop_attacks(sq, pos.occAll) & bishopsQueens);
    Bitboard rooksQueens = pos.pieceBB[make_piece(by, ROOK)] | pos.pieceBB[make_piece(by, QUEEN)];
    n += popcount(get_rook_attacks(sq, pos.occAll) & rooksQueens);
    return n;
}

// ---- precomputed masks for passed pawns / outposts, filled by init_eval() ----
static Bitboard ADJACENT_AND_OWN_FILES[8];
static Bitboard PASSED_MASK[COLOR_NB][SQUARE_NB];

static inline Bitboard rank_mask_of(int r) { return 0xFFULL << (r * 8); }

void init_eval() {
    for (int f = 0; f < 8; f++) {
        Bitboard m = FILE_A << f;
        if (f > 0) m |= FILE_A << (f - 1);
        if (f < 7) m |= FILE_A << (f + 1);
        ADJACENT_AND_OWN_FILES[f] = m;
    }
    for (Square sq = 0; sq < 64; sq++) {
        int f = file_of(sq), r = rank_of(sq);
        Bitboard ahead = 0;
        for (int rr = r + 1; rr < 8; rr++) ahead |= rank_mask_of(rr);
        PASSED_MASK[WHITE][sq] = ahead & ADJACENT_AND_OWN_FILES[f];
        ahead = 0;
        for (int rr = r - 1; rr >= 0; rr--) ahead |= rank_mask_of(rr);
        PASSED_MASK[BLACK][sq] = ahead & ADJACENT_AND_OWN_FILES[f];
    }
}

static const int PASSED_MG[8] = {0, 10, 17, 29, 48, 86, 140, 0};
static const int PASSED_EG[8] = {0, 18, 32, 58, 96, 160, 260, 0};

constexpr Bitboard LONG_DIAG_A1H8 = 0x8040201008040201ULL;
constexpr Bitboard LONG_DIAG_H1A8 = 0x0102040810204080ULL;
constexpr Bitboard CENTER4 = (1ULL << D4) | (1ULL << E4) | (1ULL << D5) | (1ULL << E5);

int evaluate_hce(const Position& pos) {
    int mg[2] = {0, 0}, eg[2] = {0, 0};

    // material + piece-square tables + bishop pair
    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pos.pieceBB[make_piece(col, PieceType(pt))];
            while (bb) {
                Square sq = pop_lsb(bb);
                int idx = pst_index(sq, col);
                mg[c] += PIECE_VALUE_MG[pt] + MG_TABLE[pt][idx];
                eg[c] += PIECE_VALUE_EG[pt] + EG_TABLE[pt][idx];
            }
        }
        if (popcount(pos.pieceBB[make_piece(col, BISHOP)]) >= 2) { mg[c] += 40; eg[c] += 65; }
    }

    // mobility. Only squares a piece can actually use count: not occupied by
    // its own men and not covered by an enemy pawn (a knight "mobile" onto a
    // square where a pawn takes it is not mobile). Each count is measured
    // against a typical value for that piece, so a boxed-in piece is
    // penalised and a free one rewarded, instead of every move adding a flat
    // amount (which just made the side with more pieces look better).
    if (EVAL_ON(1)) {
        for (int c = 0; c < 2; c++) {
            Color col = Color(c);
            const Bitboard occ = pos.occAll;
            const Bitboard area = ~pos.occByColor[col] &
                                  ~pawn_attack_bb(pos.pieceBB[make_piece(~col, PAWN)], ~col);
            Bitboard bb = pos.pieceBB[make_piece(col, KNIGHT)];
            while (bb) { Square s2 = pop_lsb(bb); int m = popcount(KNIGHT_ATTACKS[s2] & area) - 4;
                         mg[c] += 3 * m; eg[c] += 3 * m; }
            bb = pos.pieceBB[make_piece(col, BISHOP)];
            while (bb) { Square s2 = pop_lsb(bb); int m = popcount(get_bishop_attacks(s2, occ) & area) - 6;
                         mg[c] += 3 * m; eg[c] += 3 * m; }
            bb = pos.pieceBB[make_piece(col, ROOK)];
            while (bb) { Square s2 = pop_lsb(bb); int m = popcount(get_rook_attacks(s2, occ) & area) - 7;
                         mg[c] += 2 * m; eg[c] += 3 * m; }
            bb = pos.pieceBB[make_piece(col, QUEEN)];
            while (bb) { Square s2 = pop_lsb(bb);
                         int m = popcount((get_bishop_attacks(s2, occ) | get_rook_attacks(s2, occ)) & area) - 14;
                         mg[c] += m; eg[c] += 2 * m; }
        }
    } else {
    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        Bitboard own = pos.occByColor[col];
        Bitboard occ = pos.occAll;
        Bitboard bb;

        bb = pos.pieceBB[make_piece(col, KNIGHT)];
        while (bb) { Square s = pop_lsb(bb); int mob = popcount(KNIGHT_ATTACKS[s] & ~own); mg[c] += mob * 2; eg[c] += mob * 2; }
        bb = pos.pieceBB[make_piece(col, BISHOP)];
        while (bb) { Square s = pop_lsb(bb); int mob = popcount(get_bishop_attacks(s, occ) & ~own); mg[c] += mob * 3; eg[c] += mob * 3; }
        bb = pos.pieceBB[make_piece(col, ROOK)];
        while (bb) { Square s = pop_lsb(bb); int mob = popcount(get_rook_attacks(s, occ) & ~own); mg[c] += mob * 2; eg[c] += mob * 2; }
        bb = pos.pieceBB[make_piece(col, QUEEN)];
        while (bb) { Square s = pop_lsb(bb); int mob = popcount((get_bishop_attacks(s, occ) | get_rook_attacks(s, occ)) & ~own); mg[c] += mob; eg[c] += mob; }
    }
    }

    // doubled / isolated / passed pawns
    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        Bitboard ownPawns = pos.pieceBB[make_piece(col, PAWN)];
        Bitboard enemyPawns = pos.pieceBB[make_piece(~col, PAWN)];

        bool isolatedFile[8] = {false};
        for (int f = 0; f < 8; f++) {
            Bitboard fileMask = FILE_A << f;
            int cnt = popcount(ownPawns & fileMask);
            if (cnt > 1) { mg[c] -= 12 * (cnt - 1); eg[c] -= 20 * (cnt - 1); }
            if (cnt > 0) {
                bool isolated = true;
                if (f > 0 && (ownPawns & (FILE_A << (f - 1)))) isolated = false;
                if (f < 7 && (ownPawns & (FILE_A << (f + 1)))) isolated = false;
                if (isolated) { mg[c] -= 10; eg[c] -= 15; }
                isolatedFile[f] = isolated;
            }
        }

        // A passed pawn that nothing but the enemy king can ever stop is worth
        // nearly a queen once the enemy has no pieces left. Tracked per side
        // (best one only, so two such pawns don't double-count).
        const bool enemyBare = !(pos.pieceBB[make_piece(~col, KNIGHT)] | pos.pieceBB[make_piece(~col, BISHOP)] |
                                 pos.pieceBB[make_piece(~col, ROOK)]   | pos.pieceBB[make_piece(~col, QUEEN)]);
        int unstoppableBonus = 0;

        Bitboard bb = ownPawns;
        while (bb) {
            Square sq = pop_lsb(bb);
            if (!(enemyPawns & PASSED_MASK[col][sq])) {
                int advRank = (col == WHITE) ? rank_of(sq) : (7 - rank_of(sq));
                int pm = PASSED_MG[advRank], pe = PASSED_EG[advRank];
                if (EVAL_ON(2)) {
                    // The square directly in front decides how real the pawn is:
                    // a piece standing there blocks it (worth much less), while a
                    // free, uncontested square means it can simply walk.
                    const Square stopSq = sq + ((col == WHITE) ? 8 : -8);
                    if (get_bit(pos.occAll, stopSq)) { pm = pm * 3 / 4; pe = pe * 3 / 4; }
                    else if (count_attackers(pos, stopSq, ~col) == 0) { pm += pm / 4; pe += pe / 4; }
                }
                mg[c] += pm;
                eg[c] += pe;

                // Luật hình vuông / khoảng cách vua tới tốt thông (Reti/Tarrasch):
                // trong tàn cuộc, khả năng một tốt thông phong cấp được hay không
                // phụ thuộc rất nhiều vào vua đối phương có kịp vào "hình vuông"
                // để chặn hay không, và vua bên có tốt có hộ tống được nó hay
                // không. Đây là kiến thức tàn cuộc kinh điển mà PST tĩnh (không
                // phụ thuộc vị trí vua) không thể nắm bắt. Chỉ tính ở endgame
                // (nhân với egPhaseWeight ở cuối hàm qua thang eg[], không mg[])
                // vì ở middlegame quân còn nhiều, vua chưa thể tự do đuổi tốt.
                Square ownK = lsb(pos.pieceBB[make_piece(col, KING)]);
                Square enemyK = lsb(pos.pieceBB[make_piece(~col, KING)]);
                int f = file_of(sq);
                int promoRank = (col == WHITE) ? 7 : 0;
                Square promoSq = make_square(f, promoRank);
                int ownDist = std::max(std::abs(file_of(ownK) - f), std::abs(rank_of(ownK) - promoRank));
                int enemyDist = std::max(std::abs(file_of(enemyK) - f), std::abs(rank_of(enemyK) - promoRank));
                // Vua địch càng xa ô phong cấp so với vua mình, tốt thông càng
                // có giá trị thực tế (dễ đi lọt); thang điểm nhỏ theo rank để
                // tốt càng xa mới càng đáng kể (tốt ở hàng 2 thì vua ở đâu cũng
                // còn nhiều thời gian can thiệp).
                eg[c] += (enemyDist - ownDist) * (2 + advRank);

                if (EVAL_ON(4) && enemyBare) {
                    // Pure race: the pawn needs `dEff` moves (2 less than the
                    // rank distance on its first move if it may double-step),
                    // the enemy king needs `enemyDist`. The extra margin keeps
                    // the new queen from being taken the moment it appears --
                    // so this is deliberately stricter than the textbook square.
                    const int step = (col == WHITE) ? 8 : -8;
                    bool clear = true;
                    for (Square t = sq + step;; t += step) {
                        if (get_bit(pos.occAll, t)) { clear = false; break; }
                        if (t == promoSq) break;
                    }
                    if (clear) {
                        const int d = (col == WHITE) ? (7 - rank_of(sq)) : rank_of(sq);
                        const bool onStartRank = (col == WHITE) ? (rank_of(sq) == 1) : (rank_of(sq) == 6);
                        const int dEff = (onStartRank && d >= 2) ? d - 1 : d;
                        const bool attackerToMove = (pos.sideToMove == col);
                        if (attackerToMove ? (enemyDist >= dEff + 1) : (enemyDist >= dEff + 2))
                            unstoppableBonus = std::max(unstoppableBonus, 450 + 30 * advRank);
                    }
                }
            }

            // Connected pawns: a phalanx partner beside it, or a pawn defending
            // it diagonally from behind. Either shape makes the pawn harder to
            // win outright and easier to advance safely, so both earn a bonus.
            int f = file_of(sq), r = rank_of(sq);
            bool phalanx = (f > 0 && get_bit(ownPawns, make_square(f - 1, r))) ||
                           (f < 7 && get_bit(ownPawns, make_square(f + 1, r)));
            bool defended = PAWN_ATTACKS[~col][sq] & ownPawns;
            if (phalanx || defended) {
                int advRank = (col == WHITE) ? rank_of(sq) : (7 - rank_of(sq));
                int bonus = 5 + advRank * 3;
                mg[c] += bonus;
                eg[c] += bonus + (defended ? 4 : 0);
            }
            // Backward pawn: no neighbour ever advanced far enough to defend
            // it (phalanx/defended already ruled that out above), AND its
            // very next square is already covered by an enemy pawn -- it
            // can't push without being lost, so it just sits as a fixed
            // target. Isolated pawns are excluded (already penalized their
            // own way, and "backward" implies neighbours exist).
            if (!phalanx && !defended && !isolatedFile[f]) {
                int dir = (col == WHITE) ? 1 : -1;
                int stopR = r + dir;
                if (stopR >= 0 && stopR <= 7) {
                    Square stop = make_square(f, stopR);
                    if (PAWN_ATTACKS[~col][stop] & enemyPawns) {
                        mg[c] -= 8;
                        eg[c] -= 12;
                    }
                }
            }
        }
        if (unstoppableBonus) { mg[c] += unstoppableBonus; eg[c] += unstoppableBonus; }
    }

    // rook on open / semi-open file, rook on 7th rank
    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        Bitboard ownPawns = pos.pieceBB[make_piece(col, PAWN)];
        Bitboard enemyPawns = pos.pieceBB[make_piece(~col, PAWN)];
        int seventh = (col == WHITE) ? 6 : 1;
        Bitboard bb = pos.pieceBB[make_piece(col, ROOK)];
        while (bb) {
            Square sq = pop_lsb(bb);
            Bitboard fileMask = FILE_A << file_of(sq);
            bool ownOnFile = ownPawns & fileMask;
            bool enemyOnFile = enemyPawns & fileMask;
            if (!ownOnFile && !enemyOnFile) { mg[c] += 20; eg[c] += 10; }
            else if (!ownOnFile)            { mg[c] += 10; eg[c] += 5; }
            if (rank_of(sq) == seventh)     { mg[c] += 15; eg[c] += 25; }

            // Xe bị nhốt góc (trapped rook): mẫu hình kinh điển khi vua chưa
            // nhập thành hoặc vừa nhập thành nhưng chưa "mở đường" cho xe --
            // xe nằm ở file biên (a/h), gần hàng nhà, và gần như không còn ô
            // nào để di chuyển (mobility thực tế rất thấp). Đây là một trong
            // những lỗi khai cuộc/trung cuộc kinh điển nhất (ví dụ Rh1 bị Kg1
            // nhốt sau khi nhập thành cánh vua mà xe chưa kịp ra cột mở) và
            // các engine mạnh luôn phạt nó vì PST tĩnh không thấy được sự phụ
            // thuộc vào occupancy thực tế quanh xe. Dùng thẳng mobility thực
            // (đã tính bằng get_rook_attacks, xuyên occupancy thật) làm điều
            // kiện chính thay vì đoán vị trí vua bằng ngưỡng file cố định --
            // cách này tự động đúng trong mọi cấu hình quân cản đường, không
            // chỉ riêng trường hợp vua chặn, và tránh được lỗi phạt nhầm một
            // xe vẫn còn thừa nhiều ô trống để lùi ra (đã từng xảy ra khi
            // dùng ngưỡng file: vua ở e1 bị tính nhầm là "chặn" xe h1 dù cách
            // nhau 3 ô trống hoàn toàn thoáng).
            // Chỉ xét middlegame -- tàn cuộc quân thưa, xe hiếm khi thực sự
            // bị nhốt theo nghĩa này.
            {
                int f = file_of(sq), r = rank_of(sq);
                int homeRank = (col == WHITE) ? 0 : 7;
                bool nearHome = (r == homeRank) || (r == homeRank + (col == WHITE ? 1 : -1));
                if ((f == 0 || f == 7) && nearHome) {
                    int mob = popcount(get_rook_attacks(sq, pos.occAll) & ~pos.occByColor[col]);
                    // mob==0: hoàn toàn kẹt, không đi được ô nào -- phạt nặng.
                    // mob==1: gần như kẹt (chỉ 1 ô thoát) -- phạt vừa.
                    // mob>=2: đã có đủ lựa chọn, không còn coi là "trapped" theo
                    // nghĩa nguy hiểm thực sự (một xe với 2+ ô đi vẫn linh hoạt).
                    if (mob == 0) mg[c] -= 40;
                    else if (mob == 1) mg[c] -= 20;
                }
            }
        }
    }

    // knight outposts: defended by own pawn, can never be chased off by an enemy pawn
    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        Bitboard ownPawns = pos.pieceBB[make_piece(col, PAWN)];
        Bitboard enemyPawns = pos.pieceBB[make_piece(~col, PAWN)];
        Bitboard bb = pos.pieceBB[make_piece(col, KNIGHT)];
        while (bb) {
            Square sq = pop_lsb(bb);
            int advRank = (col == WHITE) ? rank_of(sq) : (7 - rank_of(sq));
            if (advRank >= 3 && advRank <= 5) {
                bool defended = PAWN_ATTACKS[~col][sq] & ownPawns;
                Bitboard adjEnemyAhead = enemyPawns & PASSED_MASK[col][sq] & ~(FILE_A << file_of(sq));
                if (defended && !adjEnemyAhead) { mg[c] += 20; eg[c] += 10; }
            }
        }
    }

    // bishop controlling an open long diagonal towards the centre
    for (int c = 0; c < 2; c++) {
        Bitboard bb = pos.pieceBB[make_piece(Color(c), BISHOP)];
        while (bb) {
            Square sq = pop_lsb(bb);
            if (get_bit(LONG_DIAG_A1H8, sq) || get_bit(LONG_DIAG_H1A8, sq)) {
                if (get_bishop_attacks(sq, pos.occAll) & CENTER4) { mg[c] += 10; eg[c] += 5; }
            }
        }
    }

    // "Bad bishop": tượng bị chính tốt nhà mình đứng trên cùng màu ô che chắn.
    // Đây là khái niệm rất kinh điển trong lý thuyết cờ vua (tượng cùng màu ô
    // với tốt nhà không thể tấn công/phòng thủ những ô đó, và dễ bị nhốt sau
    // chuỗi tốt của chính mình). Phạt tỉ lệ với số tốt nhà đứng trên ô cùng
    // màu với tượng; nặng hơn ở tàn cuộc vì lúc đó không còn đủ quân khác để
    // bù đắp sự thụ động của tượng.
    {
        constexpr Bitboard LIGHT_SQUARES = 0x55AA55AA55AA55AAULL;
        constexpr Bitboard DARK_SQUARES  = ~LIGHT_SQUARES;
        for (int c = 0; c < 2; c++) {
            Color col = Color(c);
            Bitboard ownPawns = pos.pieceBB[make_piece(col, PAWN)];
            Bitboard bb = pos.pieceBB[make_piece(col, BISHOP)];
            while (bb) {
                Square sq = pop_lsb(bb);
                Bitboard sameColorSquares = get_bit(LIGHT_SQUARES, sq) ? LIGHT_SQUARES : DARK_SQUARES;
                int blockedPawns = popcount(ownPawns & sameColorSquares);
                mg[c] -= blockedPawns * 3;
                eg[c] -= blockedPawns * 5;
            }
        }
    }


    // king tropism: queens and knights posted close to the enemy king carry
    // real attacking potential even before any concrete threat exists, so
    // give a small, distance-scaled nudge towards keeping them nearby.
    // Middlegame only -- proximity to the enemy king stops meaning the same
    // thing once mating nets give way to endgame technique.
    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        Square enemyKing = lsb(pos.pieceBB[make_piece(~col, KING)]);
        int ekf = file_of(enemyKing), ekr = rank_of(enemyKing);
        Bitboard bb = pos.pieceBB[make_piece(col, QUEEN)];
        while (bb) {
            Square sq = pop_lsb(bb);
            int dist = std::max(std::abs(file_of(sq) - ekf), std::abs(rank_of(sq) - ekr));
            mg[c] += (7 - dist) * 3;
        }
        bb = pos.pieceBB[make_piece(col, KNIGHT)];
        while (bb) {
            Square sq = pop_lsb(bb);
            int dist = std::max(std::abs(file_of(sq) - ekf), std::abs(rank_of(sq) - ekr));
            mg[c] += (7 - dist) * 2;
        }
    }

    // king safety: pawn shield + open files near king + nearby enemy attackers
    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        Square ksq = lsb(pos.pieceBB[make_piece(col, KING)]);
        int kf = file_of(ksq), kr = rank_of(ksq);
        Bitboard ownPawns = pos.pieceBB[make_piece(col, PAWN)];
        Bitboard enemyPawns = pos.pieceBB[make_piece(~col, PAWN)];
        int dir = (col == WHITE) ? 1 : -1;

        for (int df = -1; df <= 1; df++) {
            int f = kf + df;
            if (f < 0 || f > 7) continue;
            Bitboard fileMask = FILE_A << f;
            if (!(ownPawns & fileMask)) {
                mg[c] -= (enemyPawns & fileMask) ? 12 : 25;
            } else {
                bool near = false;
                for (int step = 1; step <= 2 && !near; step++) {
                    int rr = kr + dir * step;
                    if (rr < 0 || rr > 7) break;
                    if (get_bit(ownPawns, make_square(f, rr))) near = true;
                }
                if (near) mg[c] += 10;
            }
        }

        Bitboard kingZone = KING_ATTACKS[ksq];
        Bitboard occ = pos.occAll;
        int attackWeight = 0;
        Bitboard bb = pos.pieceBB[make_piece(~col, KNIGHT)];
        while (bb) { Square s = pop_lsb(bb); if (KNIGHT_ATTACKS[s] & kingZone) attackWeight += 2; }
        bb = pos.pieceBB[make_piece(~col, BISHOP)];
        while (bb) { Square s = pop_lsb(bb); if (get_bishop_attacks(s, occ) & kingZone) attackWeight += 2; }
        bb = pos.pieceBB[make_piece(~col, ROOK)];
        while (bb) { Square s = pop_lsb(bb); if (get_rook_attacks(s, occ) & kingZone) attackWeight += 3; }
        bb = pos.pieceBB[make_piece(~col, QUEEN)];
        while (bb) { Square s = pop_lsb(bb); if ((get_bishop_attacks(s, occ) | get_rook_attacks(s, occ)) & kingZone) attackWeight += 5; }
        int attackPenalty = (attackWeight * attackWeight) / 2;
        // Without an enemy queen, the same minor-piece/rook pressure on the king
        // is much less dangerous (no mating net), so it counts for half.
        if (EVAL_ON(8) && !pos.pieceBB[make_piece(~col, QUEEN)]) attackPenalty /= 2;
        mg[c] -= attackPenalty;

        // "Virtual queen mobility": đặt một hậu ảo ngay tại ô vua và đếm số ô
        // nó khống chế được xuyên qua occupancy thật của bàn cờ. Đây là thước
        // đo độ an toàn vua kinh điển (rộng rãi hơn nhiều so với chỉ đếm pawn
        // shield): một vua có nhiều đường chéo/dọc/ngang thoáng xung quanh --
        // dù hiện tại chưa có quân địch nào đứng trên đó -- vẫn dễ bị xâm
        // nhập/chiếu hơn một vua bị bịt kín, vì bất kỳ quân địch nào ghé vào
        // các đường đó đều lập tức uy hiếp trực tiếp. Ngược lại pawn shield
        // tĩnh không phân biệt được "3 tốt còn nguyên nhưng cột bên cạnh mở
        // toang" với "3 tốt còn nguyên và mọi thứ xung quanh đều kín".
        // Chỉ áp dụng middlegame (khi hậu còn trên bàn của ít nhất một bên --
        // nếu cả hai đã đổi hết hậu thì khái niệm này gần như vô nghĩa).
        if (pos.pieceBB[WQ] | pos.pieceBB[BQ]) {
            Bitboard virtualQueenReach = get_bishop_attacks(ksq, occ) | get_rook_attacks(ksq, occ);
            int openLines = popcount(virtualQueenReach);
            mg[c] -= openLines / 2;
        }
    }

    // space: safe, non-pawn-attacked squares in one's own half of the centre
    // files, weighted by how many pieces are actually around to use them.
    // Only matters in the middlegame -- an empty board has no use for space.
    {
        constexpr Bitboard SPACE_FILES = (FILE_A << 2) | (FILE_A << 3) | (FILE_A << 4) | (FILE_A << 5);
        Bitboard spaceZone[2] = {
            SPACE_FILES & (rank_mask_of(1) | rank_mask_of(2) | rank_mask_of(3)), // White: ranks 2-4
            SPACE_FILES & (rank_mask_of(4) | rank_mask_of(5) | rank_mask_of(6)), // Black: ranks 5-7
        };
        for (int c = 0; c < 2; c++) {
            Color col = Color(c);
            Bitboard ownPawns = pos.pieceBB[make_piece(col, PAWN)];
            Bitboard enemyPawnAttacks = pawn_attack_bb(pos.pieceBB[make_piece(~col, PAWN)], ~col);
            Bitboard safe = spaceZone[c] & ~ownPawns & ~enemyPawnAttacks;
            int nonPawnPieces = popcount(pos.occByColor[col] & ~ownPawns & ~pos.pieceBB[make_piece(col, KING)]);
            mg[c] += (popcount(safe) * nonPawnPieces) / 4;
        }
    }

    // threats: pieces sitting on a square a cheaper enemy piece (especially a
    // pawn) attacks, or with more attackers than defenders -- both are signs
    // the position is about to lose material even if no capture has happened
    // yet, which plain material+PST scoring can't see on its own.
    for (int c = 0; c < 2; c++) {
        Color col = Color(c);
        Bitboard enemyPawnAttacks = pawn_attack_bb(pos.pieceBB[make_piece(~col, PAWN)], ~col);
        for (int pt = KNIGHT; pt <= QUEEN; pt++) {
            Bitboard bb = pos.pieceBB[make_piece(col, PieceType(pt))];
            while (bb) {
                Square sq = pop_lsb(bb);
                if (get_bit(enemyPawnAttacks, sq)) {
                    mg[c] -= PIECE_VALUE_MG[pt] / 8;
                    eg[c] -= PIECE_VALUE_EG[pt] / 8;
                } else if (count_attackers(pos, sq, ~col) > count_attackers(pos, sq, col)) {
                    mg[c] -= PIECE_VALUE_MG[pt] / 10;
                    eg[c] -= PIECE_VALUE_EG[pt] / 10;
                }
            }
        }
    }

    int phase = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = KNIGHT; pt <= QUEEN; pt++)
            phase += PHASE_WEIGHT[pt] * popcount(pos.pieceBB[make_piece(Color(c), PieceType(pt))]);
    if (phase > MAX_PHASE) phase = MAX_PHASE;

    int mgScore = mg[WHITE] - mg[BLACK];
    int egScore = eg[WHITE] - eg[BLACK];
    int score = (mgScore * phase + egScore * (MAX_PHASE - phase)) / MAX_PHASE;

    return ((pos.sideToMove == WHITE) ? score : -score) + TEMPO;
}

// ---------------------------------------------------------------------------
// Final adjustment applied to BOTH evaluators (hand-crafted and NNUE).
// ---------------------------------------------------------------------------
static int non_pawn_material(const Position& pos, Color c) {
    return 337 * popcount(pos.pieceBB[make_piece(c, KNIGHT)]) + 365 * popcount(pos.pieceBB[make_piece(c, BISHOP)]) +
           477 * popcount(pos.pieceBB[make_piece(c, ROOK)])   + 1025 * popcount(pos.pieceBB[make_piece(c, QUEEN)]);
}

// How much of the advantage of `strong` survives, in 64ths. Material alone
// overrates endings that are theoretically drawn or nearly so.
static int scale_factor(const Position& pos, Color strong) {
    const Color weak = ~strong;
    const int pawnsS = popcount(pos.pieceBB[make_piece(strong, PAWN)]);
    const int pawnsW = popcount(pos.pieceBB[make_piece(weak, PAWN)]);
    const int nS = popcount(pos.pieceBB[make_piece(strong, KNIGHT)]), nW = popcount(pos.pieceBB[make_piece(weak, KNIGHT)]);
    const int bS = popcount(pos.pieceBB[make_piece(strong, BISHOP)]), bW = popcount(pos.pieceBB[make_piece(weak, BISHOP)]);
    const int rS = popcount(pos.pieceBB[make_piece(strong, ROOK)]),   rW = popcount(pos.pieceBB[make_piece(weak, ROOK)]);
    const int qS = popcount(pos.pieceBB[make_piece(strong, QUEEN)]),  qW = popcount(pos.pieceBB[make_piece(weak, QUEEN)]);
    const int npmS = non_pawn_material(pos, strong), npmW = non_pawn_material(pos, weak);
    int scale = 64;

    if (pawnsS == 0) {
        // Two knights cannot force mate against a bare king.
        if (npmW == 0 && pawnsW == 0 && nS == 2 && bS == 0 && rS == 0 && qS == 0) return 2;
        // No pawns and at most a minor piece ahead: cannot be converted
        // (R vs B, R vs N, R+B vs R, B vs N ...). Limited to small endgames so a
        // middlegame attack with queens still on is not flattened.
        // Was 8/64 — too aggressive; flattened many convertible edges into near-draws.
        if (npmS - npmW <= 380 && npmS + npmW <= 1684) scale = std::min(scale, 24);
    }

    // Bishops of opposite colours (alone, or with one rook each) are famously drawish.
    if (bS == 1 && bW == 1 && nS == 0 && nW == 0 && qS == 0 && qW == 0) {
        const Square sb = lsb(pos.pieceBB[make_piece(strong, BISHOP)]);
        const Square wb = lsb(pos.pieceBB[make_piece(weak, BISHOP)]);
        if (((file_of(sb) + rank_of(sb)) & 1) != ((file_of(wb) + rank_of(wb)) & 1)) {
            if (rS == 0 && rW == 0)      scale = std::min(scale, std::min(64, 32 + 8 * std::max(0, pawnsS - pawnsW - 1)));
            else if (rS == 1 && rW == 1) scale = std::min(scale, 48);
        }
    }
    return scale;
}

int evaluate(const Position& pos) {
    if (popcount(pos.occAll) <= 3 && tablebases_available()) {
        TBProbeResult tb;
        if (probe_tb(pos, tb)) {
            const int scoreWhitePerspective = tb_score_for(tb.result, tb.dtm);
            return (pos.sideToMove == WHITE) ? scoreWhitePerspective : -scoreWhitePerspective;
        }
    }

    int v;
    if (g_nnue.is_loaded()) {
        v = g_nnue.evaluate(pos);
        // Small / generic nets often drift in sparse endgames. Blend handcrafted
        // eval in when few pieces remain so conversion does not "fade".
        const int pcs = popcount(pos.occAll);
        // Blend HCE earlier (≤10 pieces). Small nets drift in sparse endings;
        // classical terms (passed pawns, king activity, material) stay reliable.
        if (pcs <= 10) {
            const int h = evaluate_hce(pos);
            // pcs=10 → 20% HCE, pcs=8 → 35%, pcs=6 → 55%, pcs≤4 → 80%
            int wH;
            if      (pcs <= 4) wH = 80;
            else if (pcs <= 6) wH = 55;
            else if (pcs <= 8) wH = 35;
            else               wH = 20;
            v = (v * (100 - wH) + h * wH) / 100;
        }
    } else {
        v = evaluate_hce(pos);
    }

    if (EVAL_ON(16) && v != 0) {
        const Color strong = (v > 0) ? pos.sideToMove : ~pos.sideToMove;
        const int sc = scale_factor(pos, strong);
        if (sc < 64) v = v * sc / 64;
    }
    if (EVAL_ON(32)) {
        // The closer a game is to the fifty-move draw, the less any advantage
        // is worth; without this the engine happily shuffles past the limit
        // while "winning".
        const int hm = std::min(pos.halfmoveClock, 100);
        v -= v * hm / 200;
    }
    return v;
}

} // namespace halcyon
