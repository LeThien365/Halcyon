#pragma once
#include "position.h"
#include <vector>

namespace halcyon {

struct MoveList {
    static constexpr int CAP = 256;
    Move m[CAP];
    int n = 0;
    void clear() { n = 0; }
    void push_back(Move x) { if (n < CAP) m[n++] = x; }
    int size() const { return n; }
    bool empty() const { return n == 0; }
    Move* begin() { return m; }
    Move* end() { return m + n; }
    const Move* begin() const { return m; }
    const Move* end() const { return m + n; }
    Move& operator[](int i) { return m[i]; }
    const Move& operator[](int i) const { return m[i]; }
};

void generate_pseudo_moves(const Position& pos, MoveList& moves);
void generate_captures(const Position& pos, MoveList& moves);
void generate_legal_moves(Position& pos, MoveList& moves);
bool move_is_legal(Position& pos, Move m);

} // namespace halcyon
