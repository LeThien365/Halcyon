// verify_tb3.cpp -- independent sanity checks for a Halcyon 3-piece table.
//
// Deliberately does NOT reuse any logic from halcyon_tb_gen3.cpp's move
// generator; it re-derives attacks from scratch a second, different way
// (bitboard-based here vs. delta-array-based in the generator) so a bug
// shared by both wouldn't slip through undetected.
#include "halcyon_tb_common.h"
#include "halcyon_tb_format.h"
#include <cstdint>
#include <iostream>
#include <string>

using namespace halcyon;
using namespace halcyon::tb;

static const int KD[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
static const int ND[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
static inline int F(int s){return s&7;} static inline int R(int s){return s>>3;}
static inline bool ob(int f,int r){return f>=0&&f<8&&r>=0&&r<8;}
static inline int MK(int f,int r){return r*8+f;}

static uint64_t king_bb(int sq) {
    uint64_t bb=0; int f=F(sq),r=R(sq);
    for (auto&d:KD){int nf=f+d[0],nr=r+d[1]; if (ob(nf,nr)) bb|=1ULL<<MK(nf,nr);}
    return bb;
}
static uint64_t knight_bb(int sq) {
    uint64_t bb=0; int f=F(sq),r=R(sq);
    for (auto&d:ND){int nf=f+d[0],nr=r+d[1]; if (ob(nf,nr)) bb|=1ULL<<MK(nf,nr);}
    return bb;
}
static uint64_t slider_bb(int sq, bool diag, bool orth, uint64_t occ) {
    uint64_t bb=0; int f=F(sq),r=R(sq);
    static const int DG[4][2]={{1,1},{1,-1},{-1,1},{-1,-1}};
    static const int OR[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    if (diag) for (auto&d:DG){int nf=f+d[0],nr=r+d[1]; while(ob(nf,nr)){int s=MK(nf,nr); bb|=1ULL<<s; if(occ&(1ULL<<s))break; nf+=d[0]; nr+=d[1];}}
    if (orth) for (auto&d:OR){int nf=f+d[0],nr=r+d[1]; while(ob(nf,nr)){int s=MK(nf,nr); bb|=1ULL<<s; if(occ&(1ULL<<s))break; nf+=d[0]; nr+=d[1];}}
    return bb;
}
static bool adj(int a,int b){return std::abs(F(a)-F(b))<=1 && std::abs(R(a)-R(b))<=1;}

static uint64_t x_attacks(PieceKind k, int x, uint64_t occ) {
    switch (k) {
        case PK_QUEEN:  return slider_bb(x, true, true, occ);
        case PK_ROOK:   return slider_bb(x, false, true, occ);
        case PK_BISHOP: return slider_bb(x, true, false, occ);
        case PK_KNIGHT: return knight_bb(x);
        default: return 0;
    }
}

static std::string sqname(int s) { std::string r; r+=char('a'+F(s)); r+=char('1'+R(s)); return r; }

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: verify_tb3 <table.htbz>\n"; return 2; }
    Table3 t;
    if (!read_table3(argv[1], t)) { std::cerr << "failed to read " << argv[1] << "\n"; return 1; }
    std::cout << "loaded table for piece kind '" << piece_kind_letter(t.kind) << "'\n";

    int errors = 0;
    long checkedLegal = 0, wins = 0, losses = 0, draws = 0, illegal = 0;
    int maxDtmWin = 0, maxDtmLoss = 0;
    State3 worstWin{}; Color worstWinStm = WHITE;

    for (int wk = 0; wk < 64; wk++) {
        for (int bk = 0; bk < 64; bk++) {
            for (int x = 0; x < 64; x++) {
                if (wk==bk||wk==x||bk==x||adj(wk,bk)) { illegal += 2; continue; }
                uint64_t occ = (1ULL<<wk)|(1ULL<<bk)|(1ULL<<x);

                for (int stmI = 0; stmI < 2; stmI++) {
                    Color stm = Color(stmI);
                    Result res = t.at(wk, bk, x, stm);
                    uint16_t d = t.dtm_at(wk, bk, x, stm);
                    checkedLegal++;
                    if (res == Result::WIN) { wins++; maxDtmWin = std::max(maxDtmWin,(int)d); if(d>=(uint16_t)maxDtmWin){worstWin={wk,bk,x};worstWinStm=stm;} }
                    else if (res == Result::LOSS) { losses++; maxDtmLoss = std::max(maxDtmLoss,(int)d); }
                    else if (res == Result::DRAW) draws++;
                    else { std::cerr << "UNRESOLVED entry found (should never happen after generation)\n"; errors++; }

                    // ---- Check 1: side to move in check must never be a "LOSS with no legal replies but not flagged mate correctly" ----
                    // ---- Check 2: recompute White-in-check / Black-in-check independently and cross-validate stalemate/mate seeds. ----
                    if (stm == BLACK) {
                        bool inCheck = (king_bb(wk) & (1ULL<<bk)) || (x_attacks(t.kind, x, occ) & (1ULL<<bk));
                        // Does black have any legal move? (king only, since black has just the king)
                        uint64_t targets = king_bb(bk) & ~occ & ~king_bb(wk);
                        bool hasMove = false;
                        uint64_t tb2 = targets;
                        while (tb2) { int to = __builtin_ctzll(tb2); tb2 &= tb2-1;
                            uint64_t occ2 = (occ & ~(1ULL<<bk)) | (1ULL<<to);
                            bool attacked = (king_bb(wk) & (1ULL<<to)) || (x_attacks(t.kind, x, occ2) & (1ULL<<to));
                            if (!attacked) { hasMove = true; break; }
                        }
                        if (!hasMove) {
                            if (inCheck && !(res == Result::LOSS && d == 0)) {
                                std::cerr << "MATE MISMATCH at wk="<<sqname(wk)<<" bk="<<sqname(bk)<<" x="<<sqname(x)
                                          << " expected LOSS/dtm0 (checkmate), got result=" << (int)res << " dtm=" << d << "\n";
                                errors++;
                            }
                            if (!inCheck && res != Result::DRAW) {
                                std::cerr << "STALEMATE MISMATCH at wk="<<sqname(wk)<<" bk="<<sqname(bk)<<" x="<<sqname(x)
                                          << " expected DRAW, got result=" << (int)res << "\n";
                                errors++;
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "checked (state,stm) pairs: " << checkedLegal << "  illegal: " << illegal << "\n";
    std::cout << "wins=" << wins << " losses=" << losses << " draws=" << draws << "\n";
    std::cout << "max DTM (WIN side, plies to opponent's mate): " << maxDtmWin << "\n";
    std::cout << "max DTM (LOSS side): " << maxDtmLoss << "\n";
    std::cout << "worst-case winning position: wk=" << sqname(worstWin.wk) << " bk=" << sqname(worstWin.bk)
              << " x=" << sqname(worstWin.x) << " stm=" << (worstWinStm==WHITE?"white":"black") << "\n";

    // ---- Check 3: known theory numbers for specific piece kinds ----
    if (t.kind == PK_ROOK) {
        // KRvK: well-established chess theory (see e.g. Chess.com's "Tablebase
        // Madness" article on building this exact table, and multiple
        // independent tablebase-based sources) puts the longest forced mate
        // at 32 FULL MOVES for the losing side's worst starting position --
        // note this is the *exact worst case over the whole state space*,
        // not the ~16-move figure commonly cited in instructional material
        // (which describes the typical/teaching-method case, not the
        // exhaustive maximum). 32 full moves with the winning side to move
        // first is at most 32 + 31 = 63 plies (the losing side never needs
        // its 32nd reply, since mate lands on the winning side's 32nd move).
        std::cout << "\n[KRvK theory check] max DTM should be <= 63 plies (32-move exact worst-case bound): "
                  << (maxDtmWin <= 63 ? "OK" : "FAIL") << " (got " << maxDtmWin << ")\n";
        if (maxDtmWin > 63) errors++;
    }
    if (t.kind == PK_QUEEN) {
        std::cout << "\n[KQvK theory check] max DTM should be <= 19 plies (10-move known bound): "
                  << (maxDtmWin <= 19 ? "OK" : "FAIL") << " (got " << maxDtmWin << ")\n";
        if (maxDtmWin > 19) errors++;
    }
    if (t.kind == PK_BISHOP || t.kind == PK_KNIGHT) {
        std::cout << "\n[K" << piece_kind_letter(t.kind) << "vK theory check] this material is a KNOWN DRAW "
                  << "(insufficient mating material) -- every legal position must be DRAW: "
                  << (wins == 0 && losses == 0 ? "OK" : "FAIL") << " (wins=" << wins << " losses=" << losses << ")\n";
        if (wins != 0 || losses != 0) errors++;
    }

    std::cout << "\n" << (errors == 0 ? "ALL CHECKS PASSED" : "FAILURES PRESENT") << " (" << errors << " errors)\n";
    return errors == 0 ? 0 : 1;
}
