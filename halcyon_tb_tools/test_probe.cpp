// test_probe.cpp -- independent check of probe_tb()'s Position -> table
// mapping, especially the White/Black colour-swap path, which is the part
// most likely to have a sign/orientation error.
#include "position.h"
#include "attacks.h"
#include "zobrist.h"
#include "tablebase.h"
#include "halcyon_tb_common.h"
#include "halcyon_tb_format.h"
#include <iostream>
#include <string>

using namespace halcyon;

static int errors = 0;

static void check(const char* label, bool cond) {
    std::cout << (cond ? "OK   " : "FAIL ") << label << "\n";
    if (!cond) errors++;
}

int main() {
    init_zobrist();
    init_attacks();
    int n = init_tablebases("/tmp/tb_dir");
    std::cout << "loaded " << n << " tables\n";
    check("all 5 tables loaded", n == 5);
    check("tablebases_available() true", tablebases_available());

    // ---- Case 1: bare KvK, must always be a draw with no table needed. ----
    {
        Position pos; pos.set_fen("8/8/8/4k3/8/8/8/4K3 w - - 0 1");
        TBProbeResult r;
        bool ok = probe_tb(pos, r);
        check("KvK probes true", ok);
        check("KvK is DRAW", ok && r.result == TBResult::DRAW);
    }

    // ---- Case 2: White has the extra queen -- direct (unswapped) path.
    // Known theory: KQvK is always a win for the queen's side (except a
    // handful of immediate-stalemate tricks, not this position). ----
    {
        Position pos; pos.set_fen("8/8/8/3k4/8/8/8/KQ6 w - - 0 1");
        TBProbeResult r;
        bool ok = probe_tb(pos, r);
        check("KQvK (White queen) probes true", ok);
        check("KQvK (White queen) is WHITE_WIN", ok && r.result == TBResult::WHITE_WIN);
    }

    // ---- Case 3: Black has the extra queen -- the SWAPPED path. Same
    // SQUARES as case 2 (a1/b1/d5), only the colours are relabelled (White
    // king/queen -> Black king/queen and vice versa) and side-to-move
    // flipped, isolating the colour swap without also mixing in a board
    // mirror. (Two earlier versions of this test got the FEN wrong in two
    // different ways -- first by manually mirroring the board as well,
    // which tests something else; then by forgetting FEN's first field is
    // rank 8, not rank 1, so "kq6" on that field put the pieces on a8/b8
    // instead of a1/b1. Both mistakes were in how this test's FEN was
    // written, not in probe_tb, which was independently re-checked against
    // the raw table for whatever squares it's actually given and matched
    // exactly in every case.) Must be BLACK_WIN, DTM identical to case 2's,
    // since after undoing the colour relabelling this is the exact same
    // position. ----
    {
        Position pos; pos.set_fen("8/8/8/3K4/8/8/8/kq6 b - - 0 1");
        TBProbeResult r;
        bool ok = probe_tb(pos, r);
        check("KQvK (Black queen) probes true", ok);
        check("KQvK (Black queen) is BLACK_WIN", ok && r.result == TBResult::BLACK_WIN);

        Position posW; posW.set_fen("8/8/8/3k4/8/8/8/KQ6 w - - 0 1");
        TBProbeResult rW; probe_tb(posW, rW);
        check("swapped DTM matches colour-relabelled-only original DTM", ok && r.dtm == rW.dtm);
    }

    // ---- Case 4: insufficient material both colours (KBvK and KvKB) --
    // must be DRAW either way, exercising the swap path with a known-draw
    // material too (not just known-win). ----
    {
        Position posW; posW.set_fen("8/8/8/4k3/8/8/8/KB6 w - - 0 1");
        TBProbeResult rW; bool okW = probe_tb(posW, rW);
        check("KBvK (White bishop) probes true", okW);
        check("KBvK (White bishop) is DRAW", okW && rW.result == TBResult::DRAW);

        Position posB; posB.set_fen("8/8/8/4K3/8/8/8/kb6 b - - 0 1");
        TBProbeResult rB; bool okB = probe_tb(posB, rB);
        check("KBvK (Black bishop) probes true", okB);
        check("KBvK (Black bishop) is DRAW", okB && rB.result == TBResult::DRAW);
    }

    // ---- Case 5: pawn promotion path, both colours. White pawn one push
    // from queening with its own king controlling the queening square --
    // clean win. Same position colour-flipped (Black pawn advancing toward
    // rank 1) must also read as a win for the pawn's own side. ----
    {
        Position posW; posW.set_fen("7k/8/6K1/8/8/8/6P1/8 w - - 0 1");
        TBProbeResult rW; bool okW = probe_tb(posW, rW);
        check("KPvK (White pawn near queening) probes true", okW);
        check("KPvK (White pawn) is WHITE_WIN", okW && rW.result == TBResult::WHITE_WIN);

        // Same shape, mirrored top-to-bottom AND colour-flipped: Black pawn
        // on g7 advancing toward rank 1, Black king supporting from g3,
        // White bare king on h1.
        Position posB; posB.set_fen("8/6p1/8/8/8/6k1/8/7K b - - 0 1");
        TBProbeResult rB; bool okB = probe_tb(posB, rB);
        check("KPvK (Black pawn near queening) probes true", okB);
        check("KPvK (Black pawn) is BLACK_WIN", okB && rB.result == TBResult::BLACK_WIN);
        check("mirrored pawn DTM matches", okW && okB && rW.dtm == rB.dtm);
    }

    // ---- Case 6: more than 3 pieces on the board -- must always return
    // false (not applicable), regardless of what those pieces are. ----
    {
        Position pos; pos.set_startpos();
        TBProbeResult r;
        bool ok = probe_tb(pos, r);
        check("startpos (32 pieces) correctly NOT probed", !ok);
    }
    {
        Position pos; pos.set_fen("8/8/8/3k4/8/3K4/8/R6R w - - 0 1"); // 4 pieces (2 rooks)
        TBProbeResult r;
        bool ok = probe_tb(pos, r);
        check("4-piece position correctly NOT probed", !ok);
    }

    // ---- Case 7: black-to-move vs white-to-move at the SAME position must
    // never both be a "win for the mover" if the position is actually lost
    // for whoever is on move -- spot check internal consistency on a KRvK
    // position picked to be roughly in the middle of the game (not a
    // trivial immediate win) by comparing against the raw table file
    // directly, bypassing probe_tb entirely, as a second oracle. ----
    {
        tb::Table3 raw;
        bool loaded = tb::read_table3("/tmp/tb_dir/kr.htbz", raw);
        check("raw KRvK table reloads independently", loaded);
        if (loaded) {
            // wk=d4(=27), bk=d6(=43), rook=a1(=0)
            int wk = 27, bk = 43, x = 0;
            tb::Result rWraw = raw.at(wk, bk, x, WHITE);
            tb::Result rBraw = raw.at(wk, bk, x, BLACK);

            Position posW; posW.set_fen("8/8/3k4/8/3K4/8/8/R7 w - - 0 1");
            TBProbeResult prW; bool okW = probe_tb(posW, prW);
            Position posB; posB.set_fen("8/8/3k4/8/3K4/8/8/R7 b - - 0 1");
            TBProbeResult prB; bool okB = probe_tb(posB, prB);

            check("probe_tb white-to-move matches raw table (WIN<=>WHITE_WIN)",
                  okW && ((rWraw==tb::Result::WIN) == (prW.result==TBResult::WHITE_WIN))
                      && ((rWraw==tb::Result::LOSS) == (prW.result==TBResult::BLACK_WIN)));
            check("probe_tb black-to-move matches raw table (WIN<=>BLACK_WIN)",
                  okB && ((rBraw==tb::Result::WIN) == (prB.result==TBResult::BLACK_WIN))
                      && ((rBraw==tb::Result::LOSS) == (prB.result==TBResult::WHITE_WIN)));
        }
    }

    std::cout << "\n" << (errors == 0 ? "ALL CHECKS PASSED" : "FAILURES PRESENT") << " (" << errors << " errors)\n";
    return errors == 0 ? 0 : 1;
}
