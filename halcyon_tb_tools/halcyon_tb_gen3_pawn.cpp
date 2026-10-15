// halcyon_tb_gen3_pawn.cpp -- Halcyon's own K+P vs K endgame tablebase
// generator. Separate from halcyon_tb_gen3.cpp because a pawn's rules are
// fundamentally different from every other piece: it moves one way only,
// it has a two-square first move, and above all it can PROMOTE -- which
// means this generator's retrograde analysis needs the *already-solved*
// KQvK table as an input, not just its own state space, to know the result
// of a position the instant the pawn queens.
//
// ---------------------------------------------------------------------------
// State space
// ---------------------------------------------------------------------------
// (wk, bk, p, stm): two king squares, White's pawn square, side to move.
// The pawn can never be on rank 1 (impossible starting square for a piece
// that only moves forward) or rank 8 (that square means it has already
// promoted -- promotion is a MOVE that transitions out of this table
// entirely, not a resting state within it). So p ranges over squares with
// rank 2..7 inclusive: 48 possible squares.
//
// No square-symmetry reduction is applied (same choice as the other 3-piece
// generator): correctness first. A left-right mirror symmetry does exist
// for a pawn (advancing straight up the board doesn't care about a1-h1
// mirroring) but is deliberately not exploited yet.
//
// ---------------------------------------------------------------------------
// Handling promotion
// ---------------------------------------------------------------------------
// When White's pawn moves from rank 7 to rank 8, the position is no longer
// representable in this table's state space (there's no "queen square" or
// "which piece it promoted to" dimension here) -- instead, the move's
// result is looked up directly in the pre-generated KQvK table (promotion
// to queen is what matters for the WIN/LOSS/DRAW result; under-promotion
// can only ever be equal or worse for White and is not needed to determine
// whether the pawn endgame itself is won, drawn, or lost).
//
// This makes promoting moves act exactly like "terminal" edges in the
// retrograde sweep: their result is known immediately (from the KQvK
// table), rather than needing to be resolved by the fixed-point iteration
// -- so they're folded in as fixed facts from round 0, the same way
// checkmate/stalemate seeds are for the other tables.
//
// ---------------------------------------------------------------------------
// Output format
// ---------------------------------------------------------------------------
// New variant of Halcyon's own format: "HTBP" (3-piece pawn table). See
// halcyon_tb_format.h for the exact layout.
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

static const int KD[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
static inline bool ob(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }
static inline int f_of(int s) { return s & 7; }
static inline int r_of(int s) { return s >> 3; }
static inline int mk_sq(int f, int r) { return r * 8 + f; }
static inline bool adj(int a, int b) { return std::abs(f_of(a)-f_of(b)) <= 1 && std::abs(r_of(a)-r_of(b)) <= 1; }

static void king_attacks(int sq, int* out, int& n) {
    n = 0; int f = f_of(sq), r = r_of(sq);
    for (auto& d : KD) { int nf=f+d[0], nr=r+d[1]; if (ob(nf,nr)) { assert(n<8); out[n++]=mk_sq(nf,nr); } }
}

// ---------------------------------------------------------------------------
// A KPvK state. `p` = White pawn's square (rank 2..7). stm tracked
// separately as an index dimension exactly as in halcyon_tb_gen3.cpp.
// ---------------------------------------------------------------------------
struct PState { int wk, bk, p; };

static bool is_legal_pstate(const PState& s) {
    if (s.wk == s.bk || s.wk == s.p || s.bk == s.p) return false;
    if (adj(s.wk, s.bk)) return false;
    int pr = r_of(s.p);
    if (pr < 1 || pr > 6) return false;   // rank 2..7 only (0-indexed: 1..6)
    return true;
}

// Successors of a WHITE move (king or pawn). Pawn pushes/captures can't
// capture anything here (Black has no piece besides the king, and the pawn
// can never capture the enemy king), so the only special pawn move besides
// a normal advance is promotion, handled by the caller via a separate
// "is this a promotion?" check before calling this. `out` must hold >= 16.
// Returns the number of NON-PROMOTING successors; sets `promotes` true if
// pushing the pawn one square would place it on rank 8 (the caller looks
// that case up in the KQvK table instead of recursing further here).
static int gen_white_moves(const PState& s, PState* out, bool& canPromote) {
    int n = 0;
    canPromote = false;
    // King moves
    int tmp[8], cnt; king_attacks(s.wk, tmp, cnt);
    for (int i = 0; i < cnt; i++) {
        int to = tmp[i];
        if (to == s.p) continue;              // king can't step on its own pawn
        if (adj(to, s.bk)) continue;           // can't move next to the enemy king
        out[n++] = PState{to, s.bk, s.p};
    }
    // Pawn moves (White pawns advance toward rank 8 in this generator's
    // convention -- see the note in main() about how a real Black-pawn
    // position is mirrored before probing).
    int pf = f_of(s.p), pr = r_of(s.p);
    int oneSq = mk_sq(pf, pr + 1);
    bool oneBlocked = (oneSq == s.wk) || (oneSq == s.bk);
    if (pr + 1 == 7) {
        // Pushing here lands on rank 8 next move -- that's promotion, handled
        // by the caller, NOT added to `out` (it isn't a same-table successor).
        if (!oneBlocked) canPromote = true;
    } else if (!oneBlocked) {
        out[n++] = PState{s.wk, s.bk, oneSq};
        if (pr == 1) {   // still on rank 2 -> two-square first move available
            int twoSq = mk_sq(pf, pr + 2);
            bool twoBlocked = (twoSq == s.wk) || (twoSq == s.bk);
            if (!twoBlocked) out[n++] = PState{s.wk, s.bk, twoSq};
        }
    }
    return n;
}

// Is `target` attacked by White (king or pawn) in state s?
static bool attacked_by_white(const PState& s, int target) {
    if (adj(s.wk, target)) return true;
    int pf = f_of(s.p), pr = r_of(s.p);
    // A pawn on rank pr attacks the two diagonal squares on rank pr+1.
    if (r_of(target) == pr + 1 && std::abs(f_of(target) - pf) == 1) return true;
    return false;
}

// Successors of a BLACK move (king only). Filters out moves into check.
static int gen_black_moves(const PState& s, PState* out) {
    int n = 0;
    int tmp[8], cnt; king_attacks(s.bk, tmp, cnt);
    for (int i = 0; i < cnt; i++) {
        int to = tmp[i];
        if (to == s.wk || to == s.p) continue;
        if (adj(to, s.wk)) continue;
        PState cand{s.wk, to, s.p};
        if (attacked_by_white(cand, to)) continue;
        out[n++] = cand;
    }
    return n;
}

static inline size_t index_of(const PState& s, Color stm) {
    return (((size_t)s.wk * 64 + s.bk) * 64 + s.p) * 2 + (size_t)stm;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: halcyon_tb_gen3_pawn <kqvk_table.htbz> <out.htbp>\n";
        return 2;
    }
    Table3 kq;
    if (!read_table3(argv[1], kq) || kq.kind != PK_QUEEN) {
        std::cerr << "failed to load a valid KQvK table from " << argv[1] << "\n";
        return 1;
    }
    const std::string outPath = argv[2];

    const size_t N = 64u * 64u * 64u * 2u;
    std::vector<Result> table(N, Result::ILLEGAL);
    std::vector<uint16_t> dtm(N, 0);

    size_t legalCount = 0;
    for (int wk = 0; wk < 64; wk++)
        for (int bk = 0; bk < 64; bk++)
            for (int p = 0; p < 64; p++) {
                PState s{wk, bk, p};
                if (is_legal_pstate(s)) {
                    table[index_of(s, WHITE)] = Result::UNRESOLVED;
                    table[index_of(s, BLACK)] = Result::UNRESOLVED;
                    legalCount += 2;
                }
            }
    std::cout << "legal (state,stm) pairs: " << legalCount << "\n";

    auto t0 = std::chrono::steady_clock::now();

    // ---- Seed 1: Black to move, no legal replies (mate or stalemate). ----
    size_t seededMate = 0, seededStale = 0;
    PState succ[16];
    for (int wk = 0; wk < 64; wk++)
        for (int bk = 0; bk < 64; bk++)
            for (int p = 0; p < 64; p++) {
                PState s{wk, bk, p};
                if (!is_legal_pstate(s)) continue;
                size_t idx = index_of(s, BLACK);
                int n = gen_black_moves(s, succ);
                if (n > 0) continue;
                bool inCheck = attacked_by_white(s, s.bk);
                if (inCheck) { table[idx] = Result::LOSS; dtm[idx] = 0; seededMate++; }
                else         { table[idx] = Result::DRAW; seededStale++; }
            }
    std::cout << "seeded checkmates: " << seededMate << "  stalemates: " << seededStale << "\n";

    // ---- Seed 2: White to move, promoting is available and immediately
    // decisive per the KQvK table (a WIN there is trivially a WIN here too,
    // one ply further out; a DRAW/LOSS there does not by itself resolve
    // this position, since White might have a non-promoting alternative --
    // so that case is left for the ordinary retrograde sweep below, which
    // will correctly see the promotion edge as leading to a DRAW/LOSS child
    // and look for a better move instead.) ----
    // To let the retrograde sweep treat promotion uniformly with every other
    // move, we don't special-case it there; instead we precompute, for every
    // state where White CAN promote this move, what result/dtm that edge
    // provides, and feed it into the sweep as an extra pseudo-successor.
    auto promotion_result = [&](const PState& s, uint16_t& outDtm) -> Result {
        // After promotion the pawn becomes a queen on file pf, rank 8(=7
        // zero-indexed), and it becomes BLACK's move (White just moved).
        int pf = f_of(s.p);
        int qSq = mk_sq(pf, 7);
        Result r = kq.at(s.wk, s.bk, qSq, BLACK);
        outDtm = kq.dtm_at(s.wk, s.bk, qSq, BLACK);
        return r;
    };

    // ---- Retrograde fixed point, extended to also consider the promotion
    // edge (computed on the fly via promotion_result) for White-to-move
    // states where the pawn is on rank 7. ----
    int round = 0;
    size_t resolvedThisRound = 1;
    while (resolvedThisRound > 0) {
        resolvedThisRound = 0;
        round++;
        for (int wk = 0; wk < 64; wk++)
            for (int bk = 0; bk < 64; bk++)
                for (int p = 0; p < 64; p++) {
                    PState s{wk, bk, p};
                    if (!is_legal_pstate(s)) continue;

                    // ---- Black to move ----
                    {
                        size_t idx = index_of(s, BLACK);
                        if (table[idx] == Result::UNRESOLVED) {
                            int n = gen_black_moves(s, succ);
                            bool anyLoss = false; uint16_t bestWinDist = UINT16_MAX;
                            bool allWin = true; uint16_t worstLossDist = 0;
                            bool anyUnknown = (n == 0);   // n==0 handled by seeding already; guard anyway
                            for (int i = 0; i < n; i++) {
                                size_t cidx = index_of(succ[i], WHITE);
                                Result r = table[cidx];
                                if (r == Result::LOSS) { anyLoss = true; bestWinDist = std::min(bestWinDist, (uint16_t)(dtm[cidx]+1)); }
                                else if (r == Result::WIN) { worstLossDist = std::max(worstLossDist, (uint16_t)(dtm[cidx]+1)); }
                                else { allWin = false; anyUnknown = true; }
                            }
                            if (n > 0) {
                                if (anyLoss) { table[idx]=Result::WIN; dtm[idx]=bestWinDist; resolvedThisRound++; }
                                else if (allWin && !anyUnknown) { table[idx]=Result::LOSS; dtm[idx]=worstLossDist; resolvedThisRound++; }
                            }
                        }
                    }
                    // ---- White to move (may include a promotion edge) ----
                    {
                        size_t idx = index_of(s, WHITE);
                        if (table[idx] == Result::UNRESOLVED) {
                            bool canPromote = false;
                            int n = gen_white_moves(s, succ, canPromote);
                            bool anyLoss = false; uint16_t bestWinDist = UINT16_MAX;
                            bool allWin = true; uint16_t worstLossDist = 0;
                            bool anyUnknown = false;
                            for (int i = 0; i < n; i++) {
                                size_t cidx = index_of(succ[i], BLACK);
                                Result r = table[cidx];
                                if (r == Result::LOSS) { anyLoss = true; bestWinDist = std::min(bestWinDist, (uint16_t)(dtm[cidx]+1)); }
                                else if (r == Result::WIN) { worstLossDist = std::max(worstLossDist, (uint16_t)(dtm[cidx]+1)); }
                                else { allWin = false; anyUnknown = true; }
                            }
                            if (canPromote) {
                                uint16_t pd; Result pr = promotion_result(s, pd);
                                // From White's point of view: pr is Black's
                                // result right after promotion (Black to move
                                // against a queen). If pr==LOSS (i.e. Black
                                // loses that KQvK position), promoting is a
                                // WIN for White, 1 ply further out.
                                if (pr == Result::LOSS) { anyLoss = true; bestWinDist = std::min(bestWinDist, (uint16_t)(pd+1)); }
                                else if (pr == Result::WIN) { worstLossDist = std::max(worstLossDist, (uint16_t)(pd+1)); }
                                else { allWin = false; }   // DRAW after promotion: doesn't resolve LOSS for White this round
                            } else {
                                // No promotion available at all this move,
                                // and n==0 real moves -> stalemate for White
                                // (extremely rare in KPvK but must be handled).
                            }
                            int totalEdges = n + (canPromote ? 1 : 0);
                            if (totalEdges == 0) {
                                // White has literally no legal move: stalemate.
                                table[idx] = Result::DRAW;
                                resolvedThisRound++;
                            } else if (anyLoss) {
                                table[idx] = Result::WIN; dtm[idx] = bestWinDist; resolvedThisRound++;
                            } else if (allWin && !anyUnknown) {
                                table[idx] = Result::LOSS; dtm[idx] = worstLossDist; resolvedThisRound++;
                            }
                        }
                    }
                }
        std::cout << "round " << round << ": resolved " << resolvedThisRound << " this round\n";
    }

    size_t draws = 0;
    for (int wk = 0; wk < 64; wk++)
        for (int bk = 0; bk < 64; bk++)
            for (int p = 0; p < 64; p++) {
                PState s{wk, bk, p};
                if (!is_legal_pstate(s)) continue;
                for (int stmI = 0; stmI < 2; stmI++) {
                    size_t idx = index_of(s, Color(stmI));
                    if (table[idx] == Result::UNRESOLVED) { table[idx] = Result::DRAW; draws++; }
                }
            }
    std::cout << "final additional draws from fixed point: " << draws << "\n";

    auto t1 = std::chrono::steady_clock::now();
    std::cout << "generation time: " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    write_table3_pawn(outPath, table, dtm);
    std::cout << "wrote " << outPath << "\n";
    return 0;
}
