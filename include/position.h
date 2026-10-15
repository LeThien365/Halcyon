#pragma once
#include "types.h"
#include "bitboard.h"
#include "nnue_types.h"
#include <string>

namespace halcyon {

struct StateInfo {
    int     castling;
    Square  epSquare;
    int     halfmoveClock;
    Bitboard key;
    Piece   captured;
    // NNUE is maintained incrementally: make_move applies add/sub feature
    // updates; unmake_move reverses them. No full accumulator is stored here
    // (that was ~4KB/ply at hidden=1024 and dominated make/unmake cost).
};

class Position {
public:
    Bitboard pieceBB[PIECE_NB];
    Bitboard occByColor[COLOR_NB];
    Bitboard occAll;
    Piece    board[SQUARE_NB];
    Color    sideToMove;
    int      castling;
    Square   epSquare;
    int      halfmoveClock;
    int      fullmoveNumber;
    Bitboard key;

    // The *current* position's NNUE accumulator, kept incrementally in sync
    // by make_move()/unmake_move() below (true O(1) update: no bitboard
    // diffing, no periodic full recompute). Marked `computed = false`
    // whenever it hasn't been initialized yet (e.g. right after set_fen,
    // before anything has called refresh_nnue()); NNUE::evaluate() checks
    // this flag and refreshes from scratch exactly once in that cold-start
    // case, then every subsequent make_move/unmake_move keeps it current
    // for free. `mutable` because it is a derived cache of the piece
    // placement (like a memoized hash), not part of the position's logical
    // state -- evaluate() takes `const Position&` and still needs to
    // refresh this on a cold start or after a weights reload.
    mutable NNUEAccumulator nnueAcc;

    // Fixed stacks: search + game length stay well under this, and make/unmake
    // no longer touch the heap.
    static constexpr int kMaxHistory = 256;
    StateInfo history[kMaxHistory];
    int histCount = 0;
    Bitboard keyHistory[kMaxHistory];
    int keyCount = 0;

    void set_startpos();
    void set_fen(const std::string& fen);
    std::string get_fen() const;

    bool square_attacked(Square sq, Color by) const;
    bool in_check(Color c) const {
        return square_attacked(lsb(pieceBB[make_piece(c, KING)]), ~c);
    }

    void make_move(Move m);
    void unmake_move(Move m);
    void make_null_move();
    void unmake_null_move();

    void rebase();   // reset undo stack + trim key history after replaying a game
    bool is_repetition() const;
    bool is_draw() const;

    Bitboard compute_key() const;
    void print() const;

private:
    void update_occupancies();
};

} // namespace halcyon
