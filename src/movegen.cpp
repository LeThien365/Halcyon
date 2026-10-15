#include "movegen.h"
#include "attacks.h"

namespace halcyon {

static void emit_pawn_set(MoveList& moves, Bitboard bb, int delta, Piece myPawn, bool capture) {
    while (bb) {
        Square to = pop_lsb(bb);
        Square from = Square(to - delta);
        moves.push_back(make_move(from, to, myPawn, NO_PIECE_TYPE, capture));
    }
}

static void add_pawn_moves(const Position& pos, MoveList& moves, bool capturesOnly) {
    Color us = pos.sideToMove, them = ~us;
    Piece myPawn = make_piece(us, PAWN);
    Bitboard pawns = pos.pieceBB[myPawn];
    Bitboard empty = ~pos.occAll;
    Bitboard enemy = pos.occByColor[them];
    constexpr Bitboard RANK3 = 0x0000000000FF0000ULL;
    constexpr Bitboard RANK6 = 0x0000FF0000000000ULL;

    if (us == WHITE) {
        Bitboard promos = pawns & RANK_7;
        Bitboard rest = pawns & ~RANK_7;
        if (!capturesOnly) {
            Bitboard single = (rest << 8) & empty;
            Bitboard dbl = ((single & RANK3) << 8) & empty;
            emit_pawn_set(moves, single, 8, myPawn, false);
            while (dbl) {
                Square to = pop_lsb(dbl);
                moves.push_back(make_move(Square(to - 16), to, myPawn, NO_PIECE_TYPE, false, true));
            }
        }
        Bitboard capL = ((rest & NOT_FILE_A) << 7) & enemy;
        Bitboard capR = ((rest & NOT_FILE_H) << 9) & enemy;
        emit_pawn_set(moves, capL, 7, myPawn, true);
        emit_pawn_set(moves, capR, 9, myPawn, true);
        while (promos) {
            Square from = pop_lsb(promos);
            Square to1 = Square(from + 8);
            if (!capturesOnly && get_bit(empty, to1))
                for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT})
                    moves.push_back(make_move(from, to1, myPawn, pt));
            Bitboard atk = PAWN_ATTACKS[us][from] & enemy;
            while (atk) {
                Square to = pop_lsb(atk);
                for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT})
                    moves.push_back(make_move(from, to, myPawn, pt, true));
            }
        }
    } else {
        Bitboard promos = pawns & RANK_2;
        Bitboard rest = pawns & ~RANK_2;
        if (!capturesOnly) {
            Bitboard single = (rest >> 8) & empty;
            Bitboard dbl = ((single & RANK6) >> 8) & empty;
            emit_pawn_set(moves, single, -8, myPawn, false);
            while (dbl) {
                Square to = pop_lsb(dbl);
                moves.push_back(make_move(Square(to + 16), to, myPawn, NO_PIECE_TYPE, false, true));
            }
        }
        Bitboard capL = ((rest & NOT_FILE_H) >> 7) & enemy;
        Bitboard capR = ((rest & NOT_FILE_A) >> 9) & enemy;
        emit_pawn_set(moves, capL, -7, myPawn, true);
        emit_pawn_set(moves, capR, -9, myPawn, true);
        while (promos) {
            Square from = pop_lsb(promos);
            Square to1 = Square(from - 8);
            if (!capturesOnly && to1 >= 0 && get_bit(empty, to1))
                for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT})
                    moves.push_back(make_move(from, to1, myPawn, pt));
            Bitboard atk = PAWN_ATTACKS[us][from] & enemy;
            while (atk) {
                Square to = pop_lsb(atk);
                for (PieceType pt : {QUEEN, ROOK, BISHOP, KNIGHT})
                    moves.push_back(make_move(from, to, myPawn, pt, true));
            }
        }
    }

    if (pos.epSquare != NO_SQUARE) {
        Bitboard src = PAWN_ATTACKS[them][pos.epSquare] & pawns;
        while (src) {
            Square from = pop_lsb(src);
            moves.push_back(make_move(from, pos.epSquare, myPawn, NO_PIECE_TYPE, true, false, true));
        }
    }
}

static void add_piece_moves(const Position& pos, MoveList& moves, PieceType pt, bool capturesOnly) {
    Color us = pos.sideToMove;
    Piece piece = make_piece(us, pt);
    Bitboard bb = pos.pieceBB[piece];
    Bitboard own = pos.occByColor[us];
    Bitboard target = capturesOnly ? pos.occByColor[~us] : ~own;

    while (bb) {
        Square from = pop_lsb(bb);
        Bitboard attacks;
        switch (pt) {
            case KNIGHT: attacks = KNIGHT_ATTACKS[from]; break;
            case BISHOP: attacks = get_bishop_attacks(from, pos.occAll); break;
            case ROOK:   attacks = get_rook_attacks(from, pos.occAll); break;
            case QUEEN:  attacks = get_bishop_attacks(from, pos.occAll) | get_rook_attacks(from, pos.occAll); break;
            case KING:   attacks = KING_ATTACKS[from]; break;
            default:     attacks = 0;
        }
        attacks &= target;
        while (attacks) {
            Square to = pop_lsb(attacks);
            bool cap = get_bit(pos.occByColor[~us], to);
            moves.push_back(make_move(from, to, piece, NO_PIECE_TYPE, cap));
        }
    }
}

static void add_castle_moves(const Position& pos, MoveList& moves) {
    Color us = pos.sideToMove;
    Bitboard occ = pos.occAll;
    if (us == WHITE) {
        if ((pos.castling & WK_SIDE) && !get_bit(occ, F1) && !get_bit(occ, G1) &&
            !pos.square_attacked(E1, BLACK) && !pos.square_attacked(F1, BLACK) && !pos.square_attacked(G1, BLACK))
            moves.push_back(make_move(E1, G1, WK, NO_PIECE_TYPE, false, false, false, true));
        if ((pos.castling & WQ_SIDE) && !get_bit(occ, D1) && !get_bit(occ, C1) && !get_bit(occ, B1) &&
            !pos.square_attacked(E1, BLACK) && !pos.square_attacked(D1, BLACK) && !pos.square_attacked(C1, BLACK))
            moves.push_back(make_move(E1, C1, WK, NO_PIECE_TYPE, false, false, false, true));
    } else {
        if ((pos.castling & BK_SIDE) && !get_bit(occ, F8) && !get_bit(occ, G8) &&
            !pos.square_attacked(E8, WHITE) && !pos.square_attacked(F8, WHITE) && !pos.square_attacked(G8, WHITE))
            moves.push_back(make_move(E8, G8, BK, NO_PIECE_TYPE, false, false, false, true));
        if ((pos.castling & BQ_SIDE) && !get_bit(occ, D8) && !get_bit(occ, C8) && !get_bit(occ, B8) &&
            !pos.square_attacked(E8, WHITE) && !pos.square_attacked(D8, WHITE) && !pos.square_attacked(C8, WHITE))
            moves.push_back(make_move(E8, C8, BK, NO_PIECE_TYPE, false, false, false, true));
    }
}

void generate_pseudo_moves(const Position& pos, MoveList& moves) {
    moves.clear();
    add_pawn_moves(pos, moves, false);
    add_piece_moves(pos, moves, KNIGHT, false);
    add_piece_moves(pos, moves, BISHOP, false);
    add_piece_moves(pos, moves, ROOK, false);
    add_piece_moves(pos, moves, QUEEN, false);
    add_piece_moves(pos, moves, KING, false);
    add_castle_moves(pos, moves);
}

void generate_captures(const Position& pos, MoveList& moves) {
    moves.clear();
    add_pawn_moves(pos, moves, true);
    add_piece_moves(pos, moves, KNIGHT, true);
    add_piece_moves(pos, moves, BISHOP, true);
    add_piece_moves(pos, moves, ROOK, true);
    add_piece_moves(pos, moves, QUEEN, true);
    add_piece_moves(pos, moves, KING, true);
}

// Kiểm tra ô `sq` có bị phe `by` tấn công hay không với occupancy tuỳ ý `occ`.
// `ignore` là tập ô chứa quân địch sắp bị ăn (không được tính là quân tấn công).
// Hàm thuần tuý: không đụng vào trạng thái Position nên không cần make/unmake.
static inline bool attacked_with_occ(const Position& pos, Square sq, Color by, Bitboard occ, Bitboard ignore) {
    const Bitboard keep = ~ignore;
    if (PAWN_ATTACKS[~by][sq] & pos.pieceBB[make_piece(by, PAWN)] & keep) return true;
    if (KNIGHT_ATTACKS[sq] & pos.pieceBB[make_piece(by, KNIGHT)] & keep) return true;
    if (KING_ATTACKS[sq] & pos.pieceBB[make_piece(by, KING)]) return true; // vua không bao giờ bị ăn
    const Bitboard diag = (pos.pieceBB[make_piece(by, BISHOP)] | pos.pieceBB[make_piece(by, QUEEN)]) & keep;
    if (diag && (get_bishop_attacks(sq, occ) & diag)) return true;
    const Bitboard orth = (pos.pieceBB[make_piece(by, ROOK)] | pos.pieceBB[make_piece(by, QUEEN)]) & keep;
    if (orth && (get_rook_attacks(sq, occ) & orth)) return true;
    return false;
}

// Kiểm tra tính hợp lệ (vua không bị chiếu sau nước đi) mà KHÔNG cần make/unmake.
// Trước đây mỗi nước đi giả-hợp-lệ phải make_move + unmake_move (cập nhật 12 bitboard,
// khoá Zobrist, đẩy/lấy vector lịch sử). Bản này chỉ tính lại occupancy trên một biến cục bộ
// nên nhanh hơn nhiều lần, kết quả hoàn toàn tương đương (đã đối chiếu bằng perft).
bool move_is_legal(Position& pos, Move m) {
    const Color us = pos.sideToMove, them = ~us;
    const Square from = from_sq(m), to = to_sq(m);
    const Bitboard fromBB = 1ULL << from, toBB = 1ULL << to;

    if (type_of(moved_piece(m)) == KING) {
        // Nhập thành: movegen đã kiểm tra E/F/G (hoặc E/D/C) không bị tấn công.
        if (is_castle(m)) return true;
        // Vua rời ô `from` nên các quân trượt được "nhìn xuyên" qua ô đó.
        const Bitboard occ = pos.occAll ^ fromBB;
        return !attacked_with_occ(pos, to, them, occ, toBB);
    }

    const Square ksq = lsb(pos.pieceBB[make_piece(us, KING)]);
    Bitboard occ = pos.occAll;
    Bitboard captured = 0;
    if (is_ep(m)) {
        captured = 1ULL << make_square(file_of(to), rank_of(from)); // tốt bị ăn qua đường
        occ ^= captured;
    } else if (is_capture(m)) {
        captured = toBB;
    }
    occ = (occ ^ fromBB) | toBB;
    return !attacked_with_occ(pos, ksq, them, occ, captured);
}

void generate_legal_moves(Position& pos, MoveList& moves) {
    MoveList pseudo;
    generate_pseudo_moves(pos, pseudo);
    moves.clear();
    for (Move m : pseudo)
        if (move_is_legal(pos, m)) moves.push_back(m);
}

} // namespace halcyon
