#include "position.h"
#include "attacks.h"
#include "zobrist.h"
#include "nnue.h"
#include <sstream>
#include <iostream>
#include <algorithm>

namespace halcyon {

static Piece char_to_piece(char c) {
    switch (c) {
        case 'P': return WP; case 'N': return WN; case 'B': return WB;
        case 'R': return WR; case 'Q': return WQ; case 'K': return WK;
        case 'p': return BP; case 'n': return BN; case 'b': return BB;
        case 'r': return BR; case 'q': return BQ; case 'k': return BK;
        default:  return NO_PIECE;
    }
}

void Position::update_occupancies() {
    occByColor[WHITE] = occByColor[BLACK] = 0;
    for (int p = WP; p <= WK; p++) occByColor[WHITE] |= pieceBB[p];
    for (int p = BP; p <= BK; p++) occByColor[BLACK] |= pieceBB[p];
    occAll = occByColor[WHITE] | occByColor[BLACK];
}

void Position::set_startpos() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Position::set_fen(const std::string& fen) {
    for (int i = 0; i < PIECE_NB; i++) pieceBB[i] = 0;
    for (int i = 0; i < 64; i++) board[i] = NO_PIECE;

    std::istringstream iss(fen);
    std::string boardPart, sideStr, castleStr, epStr;
    int halfmove = 0, fullmove = 1;
    iss >> boardPart >> sideStr >> castleStr >> epStr;
    if (!(iss >> halfmove)) halfmove = 0;
    if (!(iss >> fullmove)) fullmove = 1;

    int rank = 7, file = 0;
    for (char c : boardPart) {
        if (c == '/') { rank--; file = 0; }
        else if (isdigit((unsigned char)c)) { file += c - '0'; }
        else {
            Square sq = make_square(file, rank);
            Piece p = char_to_piece(c);
            if (p != NO_PIECE) { board[sq] = p; set_bit(pieceBB[p], sq); }
            file++;
        }
    }

    sideToMove = (sideStr == "w") ? WHITE : BLACK;
    castling = 0;
    for (char c : castleStr) {
        if (c == 'K') castling |= WK_SIDE;
        if (c == 'Q') castling |= WQ_SIDE;
        if (c == 'k') castling |= BK_SIDE;
        if (c == 'q') castling |= BQ_SIDE;
    }
    epSquare = str_to_square(epStr);
    halfmoveClock = halfmove;
    fullmoveNumber = fullmove;

    update_occupancies();
    key = compute_key();
    histCount = 0;
    keyCount = 0;
    keyHistory[keyCount++] = key;

    // A FEN load has no "previous ply" accumulator to update incrementally
    // from -- this is the one place a full recompute is unavoidable. Every
    // make_move/unmake_move from here on keeps nnueAcc current for free.
    // refresh() marks the accumulator computed itself -- and deliberately leaves
    // it "not computed" when no network is loaded yet, so make_move() never
    // touches an empty weight table. Do not force the flag here.
    g_nnue.refresh(*this, nnueAcc);
}

std::string Position::get_fen() const {
    std::string fen;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            Piece p = board[make_square(f, r)];
            if (p == NO_PIECE) empty++;
            else {
                if (empty) { fen += char('0' + empty); empty = 0; }
                fen += piece_to_char(p);
            }
        }
        if (empty) fen += char('0' + empty);
        if (r > 0) fen += '/';
    }
    fen += ' ';
    fen += (sideToMove == WHITE) ? 'w' : 'b';
    fen += ' ';
    std::string cr;
    if (castling & WK_SIDE) cr += 'K';
    if (castling & WQ_SIDE) cr += 'Q';
    if (castling & BK_SIDE) cr += 'k';
    if (castling & BQ_SIDE) cr += 'q';
    fen += cr.empty() ? "-" : cr;
    fen += ' ';
    fen += square_to_str(epSquare);
    fen += ' ' + std::to_string(halfmoveClock);
    fen += ' ' + std::to_string(fullmoveNumber);
    return fen;
}

bool Position::square_attacked(Square sq, Color by) const {
    if (PAWN_ATTACKS[~by][sq] & pieceBB[make_piece(by, PAWN)]) return true;
    if (KNIGHT_ATTACKS[sq] & pieceBB[make_piece(by, KNIGHT)]) return true;
    if (KING_ATTACKS[sq] & pieceBB[make_piece(by, KING)]) return true;
    Bitboard bishopsQueens = pieceBB[make_piece(by, BISHOP)] | pieceBB[make_piece(by, QUEEN)];
    if (get_bishop_attacks(sq, occAll) & bishopsQueens) return true;
    Bitboard rooksQueens = pieceBB[make_piece(by, ROOK)] | pieceBB[make_piece(by, QUEEN)];
    if (get_rook_attacks(sq, occAll) & rooksQueens) return true;
    return false;
}

Bitboard Position::compute_key() const {
    Bitboard k = 0;
    for (int p = 0; p < PIECE_NB; p++) {
        Bitboard bb = pieceBB[p];
        while (bb) k ^= ZOBRIST_PIECE[p][pop_lsb(bb)];
    }
    k ^= ZOBRIST_CASTLE[castling];
    if (epSquare != NO_SQUARE) k ^= ZOBRIST_EP_FILE[file_of(epSquare)];
    if (sideToMove == BLACK) k ^= ZOBRIST_SIDE;
    return k;
}

void Position::make_move(Move m) {
    StateInfo st;
    st.castling = castling;
    st.epSquare = epSquare;
    st.halfmoveClock = halfmoveClock;
    st.key = key;
    st.captured = NO_PIECE;
    // Incremental NNUE: apply feature add/sub only (no full accumulator copy).
    const bool nnueTracked = nnueAcc.computed;

    Square from = from_sq(m), to = to_sq(m);
    Piece piece = moved_piece(m);
    Color us = sideToMove, them = ~us;

    // King-dependent nets (HalfKP / HalfKAv2) key each side's features off that
    // side's OWN king square. The only king that can move here is the mover's,
    // so: the mover's perspective is rebuilt from scratch once the move is
    // fully applied (skipPersp), while the opponent's perspective -- whose king
    // is unchanged -- keeps updating incrementally. The pre-move king squares
    // are therefore the right ones for every perspective updated below.
    // (For the 768-input net king_dependent() is false: nothing special.)
    Square wk = A1, bk = A1;
    int skipPersp = -1;
    if (nnueTracked) {
        if (pieceBB[WK]) wk = lsb(pieceBB[WK]);
        if (pieceBB[BK]) bk = lsb(pieceBB[BK]);
        if (type_of(piece) == KING && g_nnue.king_dependent()) skipPersp = int(us);
    }

    if (epSquare != NO_SQUARE) key ^= ZOBRIST_EP_FILE[file_of(epSquare)];

    if (is_ep(m)) {
        Square capSq = make_square(file_of(to), rank_of(from));
        st.captured = board[capSq];
        clear_bit(pieceBB[st.captured], capSq);
        board[capSq] = NO_PIECE;
        key ^= ZOBRIST_PIECE[st.captured][capSq];
        if (nnueTracked) g_nnue.update_sub(nnueAcc, st.captured, capSq, wk, bk, skipPersp);
    } else if (is_capture(m)) {
        st.captured = board[to];
        clear_bit(pieceBB[st.captured], to);
        key ^= ZOBRIST_PIECE[st.captured][to];
        if (nnueTracked) g_nnue.update_sub(nnueAcc, st.captured, to, wk, bk, skipPersp);
    }

    clear_bit(pieceBB[piece], from);
    board[from] = NO_PIECE;
    key ^= ZOBRIST_PIECE[piece][from];
    if (nnueTracked) g_nnue.update_sub(nnueAcc, piece, from, wk, bk, skipPersp);

    Piece placed = piece;
    if (promo_type(m) != NO_PIECE_TYPE) placed = make_piece(us, promo_type(m));
    set_bit(pieceBB[placed], to);
    board[to] = placed;
    key ^= ZOBRIST_PIECE[placed][to];
    if (nnueTracked) g_nnue.update_add(nnueAcc, placed, to, wk, bk, skipPersp);

    if (is_castle(m)) {
        Square rFrom, rTo;
        if (to == make_square(6, rank_of(from))) { rFrom = make_square(7, rank_of(from)); rTo = make_square(5, rank_of(from)); }
        else                                     { rFrom = make_square(0, rank_of(from)); rTo = make_square(3, rank_of(from)); }
        Piece rook = make_piece(us, ROOK);
        clear_bit(pieceBB[rook], rFrom); board[rFrom] = NO_PIECE; key ^= ZOBRIST_PIECE[rook][rFrom];
        set_bit(pieceBB[rook], rTo);     board[rTo] = rook;       key ^= ZOBRIST_PIECE[rook][rTo];
        if (nnueTracked) { g_nnue.update_sub(nnueAcc, rook, rFrom, wk, bk, skipPersp); g_nnue.update_add(nnueAcc, rook, rTo, wk, bk, skipPersp); }
    }

    if (skipPersp >= 0) g_nnue.refresh_perspective(*this, nnueAcc, us);

    key ^= ZOBRIST_CASTLE[castling];
    if (piece == WK) castling &= ~(WK_SIDE | WQ_SIDE);
    if (piece == BK) castling &= ~(BK_SIDE | BQ_SIDE);
    if (from == A1 || to == A1) castling &= ~WQ_SIDE;
    if (from == H1 || to == H1) castling &= ~WK_SIDE;
    if (from == A8 || to == A8) castling &= ~BQ_SIDE;
    if (from == H8 || to == H8) castling &= ~BK_SIDE;
    key ^= ZOBRIST_CASTLE[castling];

    epSquare = NO_SQUARE;
    if (is_double_push(m)) {
        epSquare = Square((from + to) / 2);
        key ^= ZOBRIST_EP_FILE[file_of(epSquare)];
    }

    if (type_of(piece) == PAWN || is_capture(m)) halfmoveClock = 0;
    else halfmoveClock++;

    if (us == BLACK) fullmoveNumber++;

    sideToMove = them;
    key ^= ZOBRIST_SIDE;

    update_occupancies();
    if (histCount < kMaxHistory) history[histCount++] = st;
    if (keyCount < kMaxHistory) keyHistory[keyCount++] = key;
}

void Position::unmake_move(Move m) {
    StateInfo st = history[--histCount];
    --keyCount;

    sideToMove = ~sideToMove;
    Color us = sideToMove;
    Color them = ~us;
    Square from = from_sq(m), to = to_sq(m);
    Piece piece = moved_piece(m);
    const bool nnueTracked = nnueAcc.computed;

    // King squares AFTER the board is restored (for reverse NNUE updates).
    // For king moves on king-dependent nets we full-refresh that perspective
    // after the board is correct instead of reverse-incrementing.
    const bool kingMove = (type_of(piece) == KING);
    const bool needKingRefresh = nnueTracked && kingMove && g_nnue.king_dependent();

    Piece placed = board[to];
    clear_bit(pieceBB[placed], to);
    set_bit(pieceBB[piece], from);
    board[from] = piece;
    board[to] = NO_PIECE;

    Square rFrom = A1, rTo = A1;
    bool didCastle = false;
    if (is_castle(m)) {
        didCastle = true;
        if (to == make_square(6, rank_of(from))) { rFrom = make_square(7, rank_of(from)); rTo = make_square(5, rank_of(from)); }
        else                                     { rFrom = make_square(0, rank_of(from)); rTo = make_square(3, rank_of(from)); }
        Piece rook = make_piece(us, ROOK);
        clear_bit(pieceBB[rook], rTo); set_bit(pieceBB[rook], rFrom);
        board[rTo] = NO_PIECE; board[rFrom] = rook;
    }

    Square epCapSq = A1;
    bool didEp = false;
    if (is_ep(m)) {
        didEp = true;
        epCapSq = make_square(file_of(to), rank_of(from));
        board[epCapSq] = st.captured;
        set_bit(pieceBB[st.captured], epCapSq);
    } else if (st.captured != NO_PIECE) {
        board[to] = st.captured;
        set_bit(pieceBB[st.captured], to);
    }

    castling = st.castling;
    epSquare = st.epSquare;
    halfmoveClock = st.halfmoveClock;
    key = st.key;
    if (us == BLACK) fullmoveNumber--;

    update_occupancies();

    // Reverse NNUE feature updates (mirror of make_move). On a king-dependent
    // net the mover's own perspective is rebuilt from scratch at the end, but
    // the OPPONENT's perspective was updated incrementally by make_move (captured
    // piece, king/rook squares) and must be reversed here too -- hence skip=us.
    if (nnueTracked) {
        const Square wk = pieceBB[WK] ? lsb(pieceBB[WK]) : A1;
        const Square bk = pieceBB[BK] ? lsb(pieceBB[BK]) : A1;
        const int skip = needKingRefresh ? int(us) : -1;
        // make: sub cap, sub piece@from, add placed@to, [castle rook]
        // unmake reverse: sub placed@to, add piece@from, [castle rook reverse], add cap
        g_nnue.update_sub(nnueAcc, placed, to, wk, bk, skip);
        g_nnue.update_add(nnueAcc, piece, from, wk, bk, skip);
        if (didCastle) {
            Piece rook = make_piece(us, ROOK);
            g_nnue.update_sub(nnueAcc, rook, rTo, wk, bk, skip);
            g_nnue.update_add(nnueAcc, rook, rFrom, wk, bk, skip);
        }
        if (didEp)
            g_nnue.update_add(nnueAcc, st.captured, epCapSq, wk, bk, skip);
        else if (st.captured != NO_PIECE)
            g_nnue.update_add(nnueAcc, st.captured, to, wk, bk, skip);
        if (needKingRefresh) g_nnue.refresh_perspective(*this, nnueAcc, us);
    }
}

void Position::make_null_move() {
    StateInfo st;
    st.castling = castling; st.epSquare = epSquare; st.halfmoveClock = halfmoveClock; st.key = key; st.captured = NO_PIECE;
    history[histCount++] = st;
    if (epSquare != NO_SQUARE) { key ^= ZOBRIST_EP_FILE[file_of(epSquare)]; epSquare = NO_SQUARE; }
    sideToMove = ~sideToMove;
    key ^= ZOBRIST_SIDE;
    keyHistory[keyCount++] = key;
}

void Position::unmake_null_move() {
    StateInfo st = history[--histCount];
    --keyCount;
    sideToMove = ~sideToMove;
    castling = st.castling; epSquare = st.epSquare; halfmoveClock = st.halfmoveClock; key = st.key;
}

// Call once after replaying "position ... moves ...": the search never unmakes
// past its root, so the StateInfo stack can restart at 0, and only the keys
// within the 50-move window can ever matter for repetition detection.
void Position::rebase() {
    const int keep = std::min(std::min(keyCount, halfmoveClock + 1), 101);
    for (int i = 0; i < keep; i++) keyHistory[i] = keyHistory[keyCount - keep + i];
    keyCount = keep;
    histCount = 0;
}

bool Position::is_repetition() const {
    int cnt = keyCount;
    int limit = std::min(halfmoveClock, cnt - 1);
    for (int i = 4; i <= limit; i += 2)
        if (keyHistory[cnt - 1 - i] == keyHistory[cnt - 1]) return true;
    return false;
}

bool Position::is_draw() const {
    if (halfmoveClock >= 100) return true;
    if (is_repetition()) return true;
    return false;
}

void Position::print() const {
    for (int r = 7; r >= 0; r--) {
        std::cout << (r + 1) << "  ";
        for (int f = 0; f < 8; f++) std::cout << piece_to_char(board[make_square(f, r)]) << " ";
        std::cout << "\n";
    }
    std::cout << "   a b c d e f g h\n";
    std::cout << "FEN: " << get_fen() << "\n";
}

} // namespace halcyon
