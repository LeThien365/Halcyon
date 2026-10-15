#pragma once
#include "position.h"
#include <string>

namespace halcyon {

// ============================================================================
//  Halcyon's own 3-piece endgame tablebase (WDL + DTM), probed at runtime.
//
//  NOT Syzygy-compatible: own file format (tools/tb/halcyon_tb_format.h),
//  own retrograde-analysis generator (tools/tb/halcyon_tb_gen3*.cpp), own
//  verification tooling. Covers every K+X vs K ending with a single extra
//  piece (X in {Q,R,B,N,P}) -- i.e. any position with 3 total pieces left
//  on the board (both kings plus at most one more piece for either side).
//
//  Usage from the engine:
//    - init_tablebases(dir) once at startup (or on "setoption" like NNUE),
//      pointing at a directory containing the five table files.
//    - probe_tb(pos, result) anywhere a Position has <= 3 pieces total.
//      Returns true (and fills `result`) iff a table for this exact
//      material configuration was loaded and the position is a legal,
//      resolvable (non-illegal) tablebase entry.
// ============================================================================

enum class TBResult : uint8_t { WHITE_WIN, BLACK_WIN, DRAW };

struct TBProbeResult {
    TBResult result;
    int dtm;   // plies to mate (0 if DRAW; meaningful only for WHITE_WIN/BLACK_WIN)
};

// Score (centipawns, from White's point of view) used to report a
// tablebase-proven win/loss through the ordinary evaluate()/search score
// channel, which otherwise only carries centipawn material/positional
// scores or a true mate score. Deliberately far above any realistic
// positional evaluation (which stays within a few thousand centipawns even
// in lopsided positions) so it always dominates in comparisons, yet well
// below search.cpp's own MATE_SCORE (29000) / MATE_THRESHOLD (29000 -
// MAX_PLY = 28872) so a tablebase score is never mistaken for, or mixed up
// arithmetically with, an actual in-search mate score (which gets ply
// adjusted on the way in/out of the transposition table -- a TB score must
// NOT go through that adjustment, and staying well clear of the mate range
// is what keeps the two from colliding). Subtracting a small multiple of
// the DTM keeps faster forced wins preferred over slower ones when search
// compares multiple tablebase-resolved lines, without ever letting the
// adjustment approach the mate range even at the maximum possible DTM.
constexpr int TB_WIN_SCORE = 20000;

inline int tb_score_for(TBResult result, int dtm) {
    if (result == TBResult::DRAW) return 0;
    const int base = TB_WIN_SCORE - dtm;   // prefer the fastest forced win
    return (result == TBResult::WHITE_WIN) ? base : -base;
}

// Scans `dir` (recursively) for every *.htbz / *.htbp file and loads each
// file that matches Halcyon's HTB3 / HTBP format. Classic 3-piece stems
// (kq/kr/kb/kn/kp) are wired into probe_tb; other stems that still parse as
// HTB3/HTBP are kept and reported. Returns the number of files loaded.
int init_tablebases(const std::string& dir);

// True if at least one table is loaded and ready to probe.
bool tablebases_available();

// Attempts to resolve `pos` exactly via tablebase. Only ever meaningful when
// popcount(pos.occAll) <= 3 (both kings plus at most one more piece); returns
// false immediately otherwise. Returns false if the required table isn't
// loaded, or if the position (after colour-flipping to the table's own
// "strong side is White" convention) isn't a table entry for some other
// reason (e.g. it's actually a 2-piece KvK, which is trivially always a
// draw and handled directly without needing a table file at all).
bool probe_tb(const Position& pos, TBProbeResult& out);

// File name constants (relative to the directory passed to
// init_tablebases), exposed so external tooling/tests can refer to them by
// the same names the loader expects.
constexpr const char* TB_FILE_KQ = "kq.htbz";
constexpr const char* TB_FILE_KR = "kr.htbz";
constexpr const char* TB_FILE_KB = "kb.htbz";
constexpr const char* TB_FILE_KN = "kn.htbz";
constexpr const char* TB_FILE_KP = "kp.htbp";

} // namespace halcyon
