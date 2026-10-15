// halcyon_tb_gen3.cpp -- Halcyon's own 3-piece endgame tablebase generator.
//
// Generates exact win/draw/loss (WDL) and distance-to-mate (DTM) tables for
// K+X vs K endings (X in {Q,R,B,N}) via retrograde analysis, entirely
// independent of Syzygy or any other engine's tablebase code or format.
//
// ---------------------------------------------------------------------------
// State space
// ---------------------------------------------------------------------------
// A position is (wk, bk, x, stm): the two king squares, the strong side's
// extra piece square, and side to move. Both kings and the extra piece are
// always White's in this generator's convention; to probe a real position
// where Black has the extra piece, the caller mirrors colours first (see
// the probing code added to the engine later).
//
// Kings can never be adjacent and can never occupy the same square as each
// other or as X -- those combinations are simply illegal and are skipped
// when the table is filled (marked ILLEGAL, never reached at runtime).
//
// ---------------------------------------------------------------------------
// Retrograde analysis, from scratch
// ---------------------------------------------------------------------------
// 1. Enumerate every (wk, bk, x, stm) that is not illegal (kings adjacent,
//    kings coincide, X coincides with a king).
// 2. Seed: any position with stm=BLACK, Black to move, Black has no legal
//    moves, and Black's king is in check -> checkmate, DTM=0, WHITE_WIN.
//    Any position with stm=BLACK, no legal moves, not in check -> stalemate,
//    DRAW (a a terminal draw, distance irrelevant).
// 3. Backward step, repeated to a fixed point:
//      For every not-yet-resolved position P with side S to move:
//        - if every legal move from P leads to an already-resolved position
//          that is a win for the opponent of S, then P is a LOSS for S
//          (mate distance = 1 + min over moves of the opponent's win distance)
//        - if at least one legal move from P leads to an already-resolved
//          position that is a LOSS for the opponent of S, then P is a WIN
//          for S (mate distance = 1 + that move's loss distance) -- among
//          all such moves we keep the SHORTEST one (fastest mate)
//        - otherwise P is still unresolved this round
//    This is done by iterating outward from DTM=0 (mate positions) exactly
//    like the classic Bellman/Zermelo backward induction/retrograde
//    algorithm for solving finite two-player games -- textbook game theory,
//    not anyone's specific tablebase implementation.
// 4. Positions never resolved after the fixed point (every move stays inside
//    the unresolved set forever) are DRAWs.
//
// To make step 3 efficient we don't rescan every position every round:
// instead of "forward" retrograde (from a position, find its predecessors),
// this generator uses the simpler PLAIN retrograde: repeatedly sweep the
// whole table and try to resolve any still-unknown position by looking at
// ITS OWN legal moves (whose results, if already known, are looked up
// directly). This is O(positions * moves) per sweep and needs at most
// (board_squares) sweeps to converge for a 3-piece table -- cheap enough
// that the extra simplicity is worth the (small, for 3 pieces) cost versus
// a true predecessor-indexed retrograde sweep.
//
// ---------------------------------------------------------------------------
// Output format -- Halcyon's own, not Syzygy's
// ---------------------------------------------------------------------------
// See halcyon_tb_format.h for the exact on-disk layout.
#include "halcyon_tb_common.h"
#include "halcyon_tb_format.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace halcyon;
using namespace halcyon::tb;

// ---------------------------------------------------------------------------
// Move generation for the tiny 3-piece state space. Deliberately
// self-contained (doesn't reuse the engine's own movegen) so the tablebase
// generator has zero dependency on -- and can independently cross-check --
// the main engine's move generator.
// ---------------------------------------------------------------------------
static const int KING_DELTAS[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
static const int KNIGHT_DELTAS[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};

static inline bool on_board(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }
static inline int mk_sq(int f, int r) { return r * 8 + f; }
static inline int f_of(int s) { return s & 7; }
static inline int r_of(int s) { return s >> 3; }

static void king_attacks(int sq, int* out, int& n) {
    n = 0;
    int f = f_of(sq), r = r_of(sq);
    for (auto& d : KING_DELTAS) { int nf = f + d[0], nr = r + d[1]; if (on_board(nf, nr)) { assert(n < 8); out[n++] = mk_sq(nf, nr); } }
}
static void knight_attacks(int sq, int* out, int& n) {
    n = 0;
    int f = f_of(sq), r = r_of(sq);
    for (auto& d : KNIGHT_DELTAS) { int nf = f + d[0], nr = r + d[1]; if (on_board(nf, nr)) { assert(n < 8); out[n++] = mk_sq(nf, nr); } }
}
static void slider_attacks(int sq, bool diag, bool orth, uint64_t occ, int* out, int& n) {
    n = 0;
    // Hard cap so a logic error here fails loudly (assert) instead of silently
    // overrunning the caller's fixed-size buffer -- exactly the class of bug
    // that caused the first run of this generator to crash.
    const int CAP = 27;
    int f = f_of(sq), r = r_of(sq);
    static const int DIAG[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static const int ORTH[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    if (diag) for (auto& d : DIAG) { int nf=f+d[0], nr=r+d[1]; while (on_board(nf,nr)) { int s=mk_sq(nf,nr); assert(n < CAP); out[n++]=s; if (occ & (1ULL<<s)) break; nf+=d[0]; nr+=d[1]; } }
    if (orth) for (auto& d : ORTH) { int nf=f+d[0], nr=r+d[1]; while (on_board(nf,nr)) { int s=mk_sq(nf,nr); assert(n < CAP); out[n++]=s; if (occ & (1ULL<<s)) break; nf+=d[0]; nr+=d[1]; } }
}

static bool king_adjacent(int a, int b) {
    int df = std::abs(f_of(a) - f_of(b)), dr = std::abs(r_of(a) - r_of(b));
    return df <= 1 && dr <= 1;
}

static bool square_attacked_by_king(int ksq, int target) { return king_adjacent(ksq, target); }
static bool square_attacked_by_knight(int nsq, int target) {
    int tmp[8], n; knight_attacks(nsq, tmp, n);
    for (int i = 0; i < n; i++) if (tmp[i] == target) return true;
    return false;
}
static bool square_attacked_by_slider(int ssq, bool diag, bool orth, uint64_t occ, int target) {
    int tmp[28], n; slider_attacks(ssq, diag, orth, occ, tmp, n);
    for (int i = 0; i < n; i++) if (tmp[i] == target) return true;
    return false;
}

// Is `target` attacked by White's piece(s) in this 3-piece state?
static bool attacked_by_white(const State3& s, PieceKind kind, int target) {
    if (square_attacked_by_king(s.wk, target)) return true;
    uint64_t occ = (1ULL << s.wk) | (1ULL << s.bk) | (1ULL << s.x);
    switch (kind) {
        case PK_QUEEN:  return square_attacked_by_slider(s.x, true, true, occ, target);
        case PK_ROOK:   return square_attacked_by_slider(s.x, false, true, occ, target);
        case PK_BISHOP: return square_attacked_by_slider(s.x, true, false, occ, target);
        case PK_KNIGHT: return square_attacked_by_knight(s.x, target);
        default: return false;
    }
}

// Enumerate every legal successor of `s` (mover's own move) into `out`.
// `side` = who is to move in `s` (WHITE means White's K [+ X] moves; BLACK
// means Black's lone king moves). Returns the count.
static int gen_moves(const State3& s, PieceKind kind, Color side, State3* out) {
    int n = 0;
    uint64_t occ = (1ULL << s.wk) | (1ULL << s.bk) | (1ULL << s.x);
    if (side == WHITE) {
        // King moves
        int tmp[8], cnt; king_attacks(s.wk, tmp, cnt);
        for (int i = 0; i < cnt; i++) {
            int to = tmp[i];
            if (to == s.x) continue;                 // own piece
            if (king_adjacent(to, s.bk)) continue;    // can't move next to enemy king
            State3 ns{to, s.bk, s.x};
            out[n++] = ns;
        }
        // X moves
        int xtmp[27], xcnt = 0;
        switch (kind) {
            case PK_QUEEN:  slider_attacks(s.x, true, true, occ, xtmp, xcnt); break;
            case PK_ROOK:   slider_attacks(s.x, false, true, occ, xtmp, xcnt); break;
            case PK_BISHOP: slider_attacks(s.x, true, false, occ, xtmp, xcnt); break;
            case PK_KNIGHT: knight_attacks(s.x, xtmp, xcnt); break;
            default: break;
        }
        for (int i = 0; i < xcnt; i++) {
            int to = xtmp[i];
            if (to == s.wk) continue;
            if (to == s.bk) continue;   // capturing the lone black king can't happen legally, but guard anyway
            State3 ns{s.wk, s.bk, to};
            out[n++] = ns;
        }
    } else {
        int tmp[8], cnt; king_attacks(s.bk, tmp, cnt);
        for (int i = 0; i < cnt; i++) {
            int to = tmp[i];
            if (to == s.wk || to == s.x) continue;
            if (king_adjacent(to, s.wk)) continue;
            if (attacked_by_white(State3{s.wk, to, s.x}, kind, to)) continue;
            State3 ns{s.wk, to, s.x};
            out[n++] = ns;
        }
    }
    return n;
}

static bool is_legal_state(const State3& s) {
    if (s.wk == s.bk || s.wk == s.x || s.bk == s.x) return false;
    if (king_adjacent(s.wk, s.bk)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Table storage: index by (wk, bk, x, stm). No symmetry reduction in this
// first version -- correctness first, compaction later once the algorithm
// is verified. 64*64*64*2 = 524288 entries per piece kind; trivial to hold
// fully in memory as plain arrays.
// ---------------------------------------------------------------------------
static inline size_t index_of(const State3& s, Color stm) {
    return (((size_t)s.wk * 64 + s.bk) * 64 + s.x) * 2 + (size_t)stm;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: halcyon_tb_gen3 <Q|R|B|N> <out.htbz>\n";
        return 2;
    }
    PieceKind kind;
    switch (argv[1][0]) {
        case 'Q': kind = PK_QUEEN; break;
        case 'R': kind = PK_ROOK; break;
        case 'B': kind = PK_BISHOP; break;
        case 'N': kind = PK_KNIGHT; break;
        default: std::cerr << "unknown piece kind '" << argv[1] << "'\n"; return 2;
    }
    const std::string outPath = argv[2];

    const size_t N = 64u * 64u * 64u * 2u;
    std::vector<Result> table(N, Result::UNRESOLVED);
    std::vector<uint16_t> dtm(N, 0);

    // Mark illegal states once, up front.
    size_t legalCount = 0;
    for (int wk = 0; wk < 64; wk++)
        for (int bk = 0; bk < 64; bk++)
            for (int x = 0; x < 64; x++) {
                State3 s{wk, bk, x};
                if (!is_legal_state(s)) {
                    table[index_of(s, WHITE)] = Result::ILLEGAL;
                    table[index_of(s, BLACK)] = Result::ILLEGAL;
                } else {
                    legalCount += 2;
                }
            }
    std::cout << "legal (state,stm) pairs: " << legalCount << "\n";

    // Seed: Black to move, no legal moves.
    auto t0 = std::chrono::steady_clock::now();
    size_t seededMate = 0, seededStale = 0;
    State3 succ[40];  // worst case: king (<=8) + queen (<=27) = 35, +margin
    for (int wk = 0; wk < 64; wk++)
        for (int bk = 0; bk < 64; bk++)
            for (int x = 0; x < 64; x++) {
                State3 s{wk, bk, x};
                size_t idx = index_of(s, BLACK);
                if (table[idx] != Result::UNRESOLVED) continue;
                int n = gen_moves(s, kind, BLACK, succ);
                if (n > 0) continue;   // has replies -> not terminal
                bool inCheck = attacked_by_white(s, kind, s.bk);
                if (inCheck) { table[idx] = Result::LOSS; dtm[idx] = 0; seededMate++; }
                else         { table[idx] = Result::DRAW; seededStale++; }
            }
    std::cout << "seeded checkmates: " << seededMate << "  stalemates: " << seededStale << "\n";

    // Retrograde fixed-point sweep.
    int round = 0;
    size_t resolvedThisRound = 1;
    while (resolvedThisRound > 0) {
        resolvedThisRound = 0;
        round++;
        for (int wk = 0; wk < 64; wk++)
            for (int bk = 0; bk < 64; bk++)
                for (int x = 0; x < 64; x++) {
                    State3 s{wk, bk, x};
                    for (int stmInt = 0; stmInt < 2; stmInt++) {
                        Color stm = Color(stmInt);
                        size_t idx = index_of(s, stm);
                        if (table[idx] != Result::UNRESOLVED) continue;
                        int n = gen_moves(s, kind, stm, succ);
                        if (n == 0) continue;   // side to move here always has moves except the BLACK-seed case already handled

                        bool anyLoss = false; uint16_t bestWinDist = UINT16_MAX;
                        bool allWin = true; uint16_t worstLossDist = 0;
                        for (int i = 0; i < n; i++) {
                            size_t cidx = index_of(succ[i], ~stm);
                            Result r = table[cidx];
                            if (r == Result::LOSS) {           // opponent loses after our move => we win
                                anyLoss = true;
                                bestWinDist = std::min(bestWinDist, (uint16_t)(dtm[cidx] + 1));
                            } else if (r == Result::WIN) {
                                worstLossDist = std::max(worstLossDist, (uint16_t)(dtm[cidx] + 1));
                            } else {
                                allWin = false;                // unresolved or draw child => can't conclude LOSS this round
                            }
                        }
                        if (anyLoss) {
                            table[idx] = Result::WIN;
                            dtm[idx] = bestWinDist;
                            resolvedThisRound++;
                        } else if (allWin) {
                            table[idx] = Result::LOSS;
                            dtm[idx] = worstLossDist;
                            resolvedThisRound++;
                        }
                        // else: stays UNRESOLVED, revisited next round
                    }
                }
        std::cout << "round " << round << ": resolved " << resolvedThisRound << " this round\n";
    }

    // Anything still unresolved is a draw (infinite play, neither side can force a result).
    size_t draws = 0;
    for (size_t i = 0; i < N; i++)
        if (table[i] == Result::UNRESOLVED) { table[i] = Result::DRAW; draws++; }
    std::cout << "final draws (incl. stalemate seeds): " << draws + seededStale << "\n";

    auto t1 = std::chrono::steady_clock::now();
    std::cout << "generation time: " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    write_table3(outPath, kind, table, dtm);
    std::cout << "wrote " << outPath << "\n";
    return 0;
}
