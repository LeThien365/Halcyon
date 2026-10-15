#include "position.h"
#include "attacks.h"
#include "zobrist.h"
#include "eval.h"
#include "nnue.h"
#include "movegen.h"
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
using namespace halcyon;
#ifdef OLD_EVAL
int evaluate_hce(const Position& pos);   // old eval.h has no such symbol; defined below via evaluate()
#endif

static std::string flip_fen(const std::string& fen) {
    std::istringstream is(fen);
    std::string board, stm, castle, ep, hm = "0", fm = "1";
    is >> board >> stm >> castle >> ep >> hm >> fm;
    std::vector<std::string> ranks; std::string cur;
    for (char c : board) { if (c == '/') { ranks.push_back(cur); cur.clear(); } else cur += c; }
    ranks.push_back(cur);
    std::string nb;
    for (int i = 7; i >= 0; i--) {
        for (char c : ranks[i]) nb += std::isalpha((unsigned char)c) ? (std::isupper((unsigned char)c) ? (char)std::tolower(c) : (char)std::toupper(c)) : c;
        if (i) nb += '/';
    }
    std::string ncastle;
    for (char c : castle) ncastle += std::isalpha((unsigned char)c) ? (std::isupper((unsigned char)c) ? (char)std::tolower(c) : (char)std::toupper(c)) : c;
    std::string nep = ep;
    if (ep != "-" && ep.size() == 2) nep[1] = (ep[1] == '3') ? '6' : '3';
    return nb + " " + (stm == "w" ? "b" : "w") + " " + ncastle + " " + nep + " " + hm + " " + fm;
}

int main(int argc, char** argv) {
    init_zobrist(); init_attacks(); init_eval(); g_nnue.load_default();
    const bool nnueOn = g_nnue.is_loaded();
    std::cout << "NNUE loaded: " << nnueOn << "\n";
    std::mt19937 rng(11);
    long checks = 0, bad = 0; std::vector<std::string> fens;
    for (int g = 0; g < 60; g++) {
        Position pos; pos.set_startpos();
        for (int ply = 0; ply < 130; ply++) {
            if (pos.is_draw()) break;
            MoveList ms; generate_legal_moves(pos, ms);
            if (ms.empty()) break;
            // bias toward captures so many endgames (where the new scaling rules live) show up
            Move m = ms[rng() % ms.size()];
            for (Move c : ms) if (is_capture(c) && (rng() % 2)) { m = c; break; }
            pos.make_move(m);
            std::string fen = pos.get_fen();
            Position fl; fl.set_fen(flip_fen(fen));
            int a = evaluate(pos), b = evaluate(fl);
            checks++;
            if (a != b) { if (bad < 5) std::cout << "ASYMMETRY evaluate: " << a << " vs " << b << "  " << fen << "\n"; bad++; }
#ifndef OLD_EVAL
            int ah = evaluate_hce(pos), bh = evaluate_hce(fl);
            checks++;
            if (ah != bh) { if (bad < 5) std::cout << "ASYMMETRY hce: " << ah << " vs " << bh << "  " << fen << "\n"; bad++; }
#endif
            if (fens.size() < 3000 && (ply % 3 == 0)) fens.push_back(fen);
        }
    }
    std::cout << "symmetry: " << checks << " checks, " << bad << " asymmetries\n";

    const char* spots[][2] = {
        {"KR vs KB  (no pawns, should be ~draw)", "8/8/4k3/8/3b4/8/8/R3K3 w - - 0 1"},
        {"R+B vs R  (no pawns, ~draw)",           "4k3/8/8/4r3/8/8/8/R2BK3 w - - 0 1"},
        {"opposite bishops + 1 pawn",             "8/4bk2/8/8/4P3/3B4/5K2/8 w - - 0 1"},
        {"same-colour bishops + 1 pawn",          "8/4bk2/8/8/4P3/2B5/5K2/8 w - - 0 1"},
        {"KNN vs K",                              "4k3/8/8/8/8/8/8/1NN1K3 w - - 0 1"},
        {"passer a5, enemy king far (unstoppable)","7k/7p/8/P7/8/8/8/K7 w - - 0 1"},
        {"passer a5, enemy king close",           "8/k6p/8/P7/8/8/8/K7 w - - 0 1"},
        {"queen vs rook+bishop endgame",          "4k3/8/8/4rb2/8/8/8/Q3K3 w - - 0 1"},
    };
    for (auto& s : spots) { Position p; p.set_fen(s[1]); std::cout << "  " << s[0] << " -> " << evaluate(p) << "\n"; }

    // ---- speed ----
    std::vector<Position> ps; for (size_t i = 0; i < fens.size() && i < 2000; i++) { Position p; p.set_fen(fens[i]); ps.push_back(p); }
    volatile long sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 40;
    for (int r = 0; r < reps; r++) for (auto& p : ps) sink += evaluate(p);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "speed: " << (long)(reps * ps.size() / sec) << " evals/s (" << ps.size() << " positions)\n";
    return bad ? 1 : 0;
}
