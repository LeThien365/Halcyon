#include "tablebase.h"
#include "halcyon_tb_common.h"
#include "halcyon_tb_format.h"
#include "bitboard.h"
#include <array>
#include <cstdio>
#include <map>
#include <string>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>

namespace halcyon {

using tb::Table3;
using tb::PieceKind;
using tb::Result;
using tb::read_table3;
using tb::read_table3_pawn;

namespace fs = std::filesystem;

namespace {

struct TBSet {
    bool loaded[PIECE_TYPE_NB] = {false};
    Table3 table[PIECE_TYPE_NB];
};

TBSet g_tb;
bool g_anyLoaded = false;
int g_loadedCount = 0;
// Extra tables keyed by lowercase stem (e.g. "kbb") for future / diagnostics.
std::map<std::string, Table3> g_extraTables;

std::string to_lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

PieceType stem_to_3piece(const std::string& stem) {
    // Recognized 3-piece KXk stems: kq, kr, kb, kn, kp
    if (stem == "kq") return QUEEN;
    if (stem == "kr") return ROOK;
    if (stem == "kb") return BISHOP;
    if (stem == "kn") return KNIGHT;
    if (stem == "kp") return PAWN;
    return NO_PIECE_TYPE;
}

bool try_load_path(const fs::path& path, int& count) {
    if (!path.has_filename()) return false;
    const std::string ext = to_lower(path.extension().string());
    const std::string stem = to_lower(path.stem().string());
    if (ext == ".txt" || ext == ".md" || ext == ".exe" || ext == ".dll" ||
        ext == ".so" || ext == ".dylib" || ext == ".json" || ext == ".pgn")
        return false;
    if (ext == ".rtbw" || ext == ".rtbz") {
        std::cout << "info string TB skip (Syzygy .rtbw/.rtbz is not Halcyon HTB): "
                  << path.filename().string() << std::endl;
        return false;
    }

    Table3 tab;
    bool ok = false;
    if (ext == ".htbp" || stem == "kp" || stem.find("kp") != std::string::npos)
        ok = read_table3_pawn(path.string(), tab);
    if (!ok)
        ok = read_table3(path.string(), tab);
    if (!ok)
        ok = read_table3_pawn(path.string(), tab);
    if (!ok) {
        if (ext == ".htbz" || ext == ".htbp" || ext == ".htb")
            std::cout << "info string TB skip (bad HTB): " << path.filename().string() << std::endl;
        return false;
    }

    PieceType pt = stem_to_3piece(stem);
    if (pt != NO_PIECE_TYPE) {
        g_tb.table[pt] = std::move(tab);
        g_tb.loaded[pt] = true;
    } else {
        g_extraTables[stem] = std::move(tab);
        std::cout << "info string TB loaded extra stem '" << stem
                  << "': " << path.filename().string() << std::endl;
    }
    count++;
    g_anyLoaded = true;
    std::cout << "info string TB loaded: " << path.filename().string() << std::endl;
    return true;
}

} // namespace

int init_tablebases(const std::string& dir) {
    for (int i = 0; i < PIECE_TYPE_NB; i++) g_tb.loaded[i] = false;
    g_extraTables.clear();
    g_anyLoaded = false;
    g_loadedCount = 0;
    int count = 0;

    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        std::cout << "info string TB path not a directory: " << dir << std::endl;
        return 0;
    }

    // Load every .htbz / .htbp in the directory (and one level of subdirs).
    auto scan = [&](const fs::path& root, bool recursive) {
        fs::directory_options opt = fs::directory_options::skip_permission_denied;
        if (recursive) {
            for (auto it = fs::recursive_directory_iterator(root, opt, ec);
                 it != fs::recursive_directory_iterator(); ++it) {
                if (ec) { ec.clear(); continue; }
                if (!it->is_regular_file(ec)) continue;
                try_load_path(it->path(), count);
            }
        } else {
            for (auto it = fs::directory_iterator(root, opt, ec);
                 it != fs::directory_iterator(); ++it) {
                if (ec) { ec.clear(); continue; }
                if (!it->is_regular_file(ec)) continue;
                try_load_path(it->path(), count);
            }
        }
    };

    scan(dir, true);

    // Fallback: also try the classic five names explicitly (in case of odd paths).
    auto tryNamed = [&](PieceType pt, const char* fname, bool pawn) {
        if (g_tb.loaded[pt]) return;
        fs::path p = fs::path(dir) / fname;
        if (!fs::exists(p)) return;
        Table3 tab;
        bool ok = pawn ? read_table3_pawn(p.string(), tab) : read_table3(p.string(), tab);
        if (!ok) return;
        g_tb.table[pt] = std::move(tab);
        g_tb.loaded[pt] = true;
        count++;
        g_anyLoaded = true;
        std::cout << "info string TB loaded: " << fname << std::endl;
    };
    tryNamed(QUEEN, TB_FILE_KQ, false);
    tryNamed(ROOK, TB_FILE_KR, false);
    tryNamed(BISHOP, TB_FILE_KB, false);
    tryNamed(KNIGHT, TB_FILE_KN, false);
    tryNamed(PAWN, TB_FILE_KP, true);

    g_loadedCount = count;
    std::cout << "info string TB total loaded files: " << count
              << " (3-piece slots q/r/b/n/p: "
              << (g_tb.loaded[QUEEN] ? "Q" : "-")
              << (g_tb.loaded[ROOK] ? "R" : "-")
              << (g_tb.loaded[BISHOP] ? "B" : "-")
              << (g_tb.loaded[KNIGHT] ? "N" : "-")
              << (g_tb.loaded[PAWN] ? "P" : "-")
              << ", extra stems: " << g_extraTables.size() << ")" << std::endl;
    return count;
}

bool tablebases_available() { return g_anyLoaded; }

bool probe_tb(const Position& pos, TBProbeResult& out) {
    const int total = popcount(pos.occAll);
    if (total > 3) return false;

    // Guard empty king bitboards (corrupt positions).
    if (!pos.pieceBB[WK] || !pos.pieceBB[BK]) return false;
    const Square wkSq = lsb(pos.pieceBB[WK]);
    const Square bkSq = lsb(pos.pieceBB[BK]);

    if (total == 2) {
        out.result = TBResult::DRAW;
        out.dtm = 0;
        return true;
    }

    PieceType extraType = NO_PIECE_TYPE;
    Color extraColor = WHITE;
    Square extraSq = NO_SQUARE;
    for (int p = 0; p < PIECE_NB; p++) {
        Piece piece = Piece(p);
        if (type_of(piece) == KING) continue;
        if (pos.pieceBB[piece]) {
            extraType = type_of(piece);
            extraColor = color_of(piece);
            extraSq = lsb(pos.pieceBB[piece]);
            break;
        }
    }
    if (extraType == NO_PIECE_TYPE) return false;
    if (!g_tb.loaded[extraType]) return false;

    const bool swapped = (extraColor == BLACK);
    const int tableWk = swapped ? bkSq : wkSq;
    const int tableBk = swapped ? wkSq : bkSq;
    const Color tableStm = swapped ? ~pos.sideToMove : pos.sideToMove;

    Result r;
    uint16_t dtm;
    if (extraType == PAWN) {
        const int tableP = swapped ? (extraSq ^ 56) : extraSq;
        const int tableWkP = swapped ? (tableWk ^ 56) : tableWk;
        const int tableBkP = swapped ? (tableBk ^ 56) : tableBk;
        r = g_tb.table[PAWN].at(tableWkP, tableBkP, tableP, tableStm);
        dtm = g_tb.table[PAWN].dtm_at(tableWkP, tableBkP, tableP, tableStm);
    } else {
        r = g_tb.table[extraType].at(tableWk, tableBk, extraSq, tableStm);
        dtm = g_tb.table[extraType].dtm_at(tableWk, tableBk, extraSq, tableStm);
    }

    if (r == Result::ILLEGAL || r == Result::UNRESOLVED) return false;

    TBResult verdict;
    if (r == Result::DRAW) {
        verdict = TBResult::DRAW;
        dtm = 0;
    } else {
        const Color winnerInTableFrame = (r == Result::WIN) ? tableStm : Color(~tableStm);
        const Color winnerReal = swapped ? Color(~winnerInTableFrame) : winnerInTableFrame;
        verdict = (winnerReal == WHITE) ? TBResult::WHITE_WIN : TBResult::BLACK_WIN;
    }

    out.result = verdict;
    out.dtm = dtm;
    return true;
}

} // namespace halcyon
