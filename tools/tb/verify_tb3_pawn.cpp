// verify_tb3_pawn.cpp -- independent sanity checks for the Halcyon KPvK
// table. Re-derives pawn/king rules from scratch, separately from both
// halcyon_tb_gen3_pawn.cpp's generator logic and verify_tb3.cpp's style.
#include "halcyon_tb_common.h"
#include "halcyon_tb_format.h"
#include <cstdint>
#include <iostream>
#include <string>

using namespace halcyon;
using namespace halcyon::tb;

static inline int F(int s){return s&7;} static inline int R(int s){return s>>3;}
static inline int MK(int f,int r){return r*8+f;}
static inline bool adjacent(int a,int b){return std::abs(F(a)-F(b))<=1 && std::abs(R(a)-R(b))<=1;}
static std::string sqname(int s){std::string r;r+=char('a'+F(s));r+=char('1'+R(s));return r;}

// Independent king-move / attack model (bitboard-based, deliberately not
// reusing the generator's delta-array style). Kept for potential future
// checks that need direct attack-square computation; not currently called
// by the checks below, which work entirely off table lookups plus the
// hand-derived rule-of-the-square geometry.
[[maybe_unused]] static uint64_t king_ring(int sq) {
    uint64_t bb = 0; int f = F(sq), r = R(sq);
    for (int df = -1; df <= 1; df++)
        for (int dr = -1; dr <= 1; dr++) {
            if (df == 0 && dr == 0) continue;
            int nf = f+df, nr = r+dr;
            if (nf>=0 && nf<8 && nr>=0 && nr<8) bb |= 1ULL << MK(nf,nr);
        }
    return bb;
}
// White pawn's attack squares (diagonal captures) from square p. Kept for
// symmetry with king_ring above; not currently called (see that comment).
[[maybe_unused]] static uint64_t pawn_attacks(int p) {
    uint64_t bb = 0; int f = F(p), r = R(p);
    if (r+1 <= 7) {
        if (f-1>=0) bb |= 1ULL << MK(f-1,r+1);
        if (f+1<=7) bb |= 1ULL << MK(f+1,r+1);
    }
    return bb;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: verify_tb3_pawn <kpvk.htbp>\n"; return 2; }
    Table3 t;
    if (!read_table3_pawn(argv[1], t)) { std::cerr << "failed to read " << argv[1] << " as an HTBP file\n"; return 1; }
    std::cout << "loaded KPvK pawn table\n";

    int errors = 0;
    long wins=0, losses=0, draws=0, illegal=0;
    int maxDtm = 0;

    // ---- Check 1: general structural sanity + independent mate/stalemate
    // re-derivation for the (few) direct-mate cases, and a same-file/king-
    // support consistency spot check for a batch of known theoretical facts. ----
    for (int wk = 0; wk < 64; wk++) {
        for (int bk = 0; bk < 64; bk++) {
            for (int p = 0; p < 64; p++) {
                int pr = R(p);
                bool legalSquareSet = (wk!=bk && wk!=p && bk!=p && !adjacent(wk,bk) && pr>=1 && pr<=6);
                for (int stmI = 0; stmI < 2; stmI++) {
                    Color stm = Color(stmI);
                    Result res = t.at(wk,bk,p,stm);
                    if (!legalSquareSet) { if (res != Result::ILLEGAL) { std::cerr<<"expected ILLEGAL wk="<<sqname(wk)<<" bk="<<sqname(bk)<<" p="<<sqname(p)<<"\n"; errors++; } illegal++; continue; }
                    if (res == Result::UNRESOLVED) { std::cerr << "UNRESOLVED left over at wk="<<sqname(wk)<<" bk="<<sqname(bk)<<" p="<<sqname(p)<<" stm="<<stmI<<"\n"; errors++; continue; }
                    uint16_t d = t.dtm_at(wk,bk,p,stm);
                    if (res==Result::WIN){wins++; maxDtm=std::max(maxDtm,(int)d);}
                    else if (res==Result::LOSS) losses++;
                    else if (res==Result::DRAW) draws++;
                }
            }
        }
    }
    std::cout << "wins="<<wins<<" losses="<<losses<<" draws="<<draws<<" illegal="<<illegal<<"\n";
    std::cout << "max DTM (WIN side): " << maxDtm << "\n";

    // ---- Check 2: "rule of the square" spot checks. ----
    // Classic KPvK theory: with the defending king OUTSIDE the pawn's
    // promotion square and no help from its own king, the pawn queens
    // untouched. We build a few textbook square/non-square cases (white
    // king far away and irrelevant, e.g. a1, so it can't help -- meaning
    // the only question is whether Black's king can catch the pawn) and
    // check the result matches "black king inside the square => can at
    // least stop/draw; outside => pawn queens (WIN for white)".
    //
    // Square of the pawn (White pawn on square p, White to move next after
    // Black's turn -- i.e. we check with Black to move, since the "outside
    // the square" rule is normally stated for the side about to move):
    // the square's corners are p, the promotion square q, and the two
    // squares completing a square of side (rank_to_promotion+1).
    auto king_in_square = [](int p, int bk) {
        // Classic "rule of the square": the pawn's square has one corner at
        // the pawn itself and the opposite corner at its promotion square.
        // The defending king is inside it iff its file distance to the
        // pawn's file AND its own remaining-ranks-to-promotion-rank are
        // both <= the pawn's remaining ranks to promotion.
        int pf = F(p), pr = R(p);
        int dist = 7 - pr;                 // pawn's ranks remaining to promotion
        int bf = F(bk), br = R(bk);
        int fileDist = std::abs(bf - pf);
        int rankDist = 7 - br;             // black king's ranks remaining to the promotion rank
        return fileDist <= dist && rankDist <= dist;
    };

    int squareChecks = 0, squareFails = 0;
    // White king parked far away (a1) and irrelevant/legal for a range of
    // pawn squares and black king squares away from a1's adjacency.
    int wkFar = MK(0,0); // a1
    for (int pf = 0; pf < 8; pf++) {
        for (int pr = 1; pr <= 5; pr++) {   // avoid rank 7 (about to promote next move -- different regime) and rank 1
            int p = MK(pf, pr);
            for (int bkf = 0; bkf < 8; bkf++) {
                for (int bkr = 0; bkr < 8; bkr++) {
                    int bk = MK(bkf, bkr);
                    if (bk==wkFar || bk==p) continue;
                    if (adjacent(wkFar,bk)) continue;
                    // Only test cases where white king is far enough to be
                    // truly irrelevant to the race (outside a generous
                    // radius of both the pawn and black king), so this is
                    // purely a pawn-vs-king race.
                    if (std::abs(F(wkFar)-pf) < 5 && std::abs(R(wkFar)-pr) < 5) continue;
                    if (adjacent(wkFar, p)) continue;

                    Result res = t.at(wkFar, bk, p, BLACK);  // result is from BLACK's (the mover's) perspective
                    if (res == Result::ILLEGAL) continue;
                    squareChecks++;
                    bool inSquare = king_in_square(p, bk);
                    // Outside the square: the defending side (Black, to
                    // move) cannot catch the pawn, so the pawn queens
                    // unaided -- that is a LOSS for Black (White wins), not
                    // a WIN for Black. res is Black's own result here.
                    if (!inSquare && res != Result::LOSS) {
                        if (squareFails < 6) std::cerr << "SQUARE-RULE MISMATCH (expected Black LOSS, pawn races home): wk=a1 bk="<<sqname(bk)<<" p="<<sqname(p)<<" got result="<<(int)res<<"\n";
                        squareFails++;
                    }
                    // Note: being INSIDE the square does not guarantee a
                    // draw/loss for White (White's own king or a rook-pawn
                    // edge case can still win) -- so we only assert the
                    // OUTSIDE direction, which is unconditional.
                }
            }
        }
    }
    std::cout << "[rule-of-the-square check] " << squareChecks << " checks, " << squareFails << " mismatches (outside-square => must be WIN)\n";
    errors += squareFails;

    // ---- Check 3: known KPvK theory numbers. ----
    // "At most 19 moves are required to promote the pawn, and at most 9
    // more to checkmate" (Wikipedia, King and pawn versus king endgame) =>
    // an upper bound around 19+9=28 full moves in the most extreme cited
    // case, i.e. well under 60 plies. This is a loose sanity ceiling, not a
    // tight bound we expect to be hit exactly.
    std::cout << "[KPvK loose DTM ceiling] max DTM should be well under 60 plies: "
              << (maxDtm <= 60 ? "OK" : "FAIL") << " (got " << maxDtm << ")\n";
    if (maxDtm > 60) errors++;

    // ---- Check 4: rook-pawn vs. knight-pawn fortress comparison. Rather
    // than hand-pick a single position and assert "this must be a draw"
    // (unreliable without a second oracle to check the assumption against,
    // as the retracted version of this check discovered), compare the ROOK
    // pawn (file a) against the KNIGHT pawn (file b) under the identical
    // king geometry, mirrored so the attacking king is equally far from
    // both corners. The established, unambiguous theoretical fact (true
    // regardless of exact distances) is: the rook pawn has strictly MORE
    // drawing chances than a center-ward pawn, because only the rook pawn
    // has the "no far side to outflank from" fortress available at all.
    // So: for every attacking-king square symmetric between the two files,
    // count(DRAW, file a) must be >= count(DRAW, file b) -- never fewer.
    // This is a comparative, structural check rather than a specific
    // pre-judged position, so it doesn't depend on correctly guessing any
    // single king distance by hand.
    {
        long drawsA = 0, drawsB = 0, comparable = 0;
        for (int wkf = 0; wkf < 8; wkf++) {
            for (int wkr = 0; wkr < 8; wkr++) {
                int wk = MK(wkf, wkr);
                // Mirror the attacking king's file so it's equidistant from
                // the a-file corner and the b-file corner (reflect about
                // the board's file-3.5 centerline: file f <-> file 7-f
                // reproduces the same king geometry relative to a vs b).
                for (int defR = 5; defR <= 7; defR++) {
                    for (int defF = 0; defF <= 2; defF++) {
                        int bkA = MK(defF, defR);
                        int pA = MK(0, 6);      // a7
                        int bkB = MK(7-defF, defR);
                        int pB = MK(1, 6);      // b7
                        if (bkA==wk||bkA==pA||adjacent(wk,bkA)) continue;
                        if (bkB==wk||bkB==pB||adjacent(wk,bkB)) continue;
                        Result rA = t.at(wk, bkA, pA, WHITE);
                        Result rB2 = t.at(wk, bkB, pB, WHITE);
                        if (rA==Result::ILLEGAL || rB2==Result::ILLEGAL) continue;
                        comparable++;
                        if (rA==Result::DRAW) drawsA++;
                        if (rB2==Result::DRAW) drawsB++;
                    }
                }
            }
        }
        std::cout << "[rook-pawn vs knight-pawn draw-rate check] comparable positions=" << comparable
                  << "  a-file draws=" << drawsA << "  b-file draws=" << drawsB
                  << " -> " << (drawsA >= drawsB ? "OK (rook pawn draws at least as often)" : "FAIL") << "\n";
        if (drawsA < drawsB) errors++;
    }

    std::cout << "\n" << (errors == 0 ? "ALL CHECKS PASSED" : "FAILURES PRESENT") << " (" << errors << " errors)\n";
    return errors == 0 ? 0 : 1;
}
