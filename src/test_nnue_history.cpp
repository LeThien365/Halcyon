// Test 1: accumulator tăng dần (sau make/unmake) phải khớp refresh từ đầu, với net HNN3.
// Test 2: lịch sử Position khi ván dài > kMaxHistory.
#include "position.h"
#include "nnue.h"
#include "movegen.h"
#include "attacks.h"
#include "zobrist.h"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <fstream>
using namespace halcyon;

static void write_hnn3(const char* path, int fset, int flags, int H) {
    const bool mirror = flags & 1;
    const int planes = (fset == NNUE_FS_HALFKP) ? 10 : 11;
    const size_t in = (size_t)(mirror ? 32 : 64) * planes * 64;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> d(-0.25f, 0.25f);
    std::ofstream f(path, std::ios::binary);
    uint32_t magic = 0x484E4E33u; f.write((char*)&magic, 4);
    int32_t v[4] = {fset, flags, H, NNUE::NUM_BUCKETS}; f.write((char*)v, 16);
    std::vector<float> w(in * H); for (auto& x : w) x = d(rng); f.write((char*)w.data(), w.size() * 4);
    std::vector<float> b1(H); for (auto& x : b1) x = d(rng); f.write((char*)b1.data(), H * 4);
    std::vector<float> w2((size_t)NNUE::NUM_BUCKETS * 2 * H); for (auto& x : w2) x = d(rng); f.write((char*)w2.data(), w2.size() * 4);
    std::vector<float> b2(NNUE::NUM_BUCKETS, 0.f); f.write((char*)b2.data(), b2.size() * 4);
}

static long long checks = 0, bad = 0;
static std::string firstBad;

static void compare(const Position& p, const std::string& what) {
    NNUEAccumulator ref;
    g_nnue.refresh(p, ref);
    checks++;
    if (std::memcmp(ref.v, p.nnueAcc.v, sizeof(ref.v)) != 0) {
        bad++;
        if (firstBad.empty()) firstBad = what + " | " + p.get_fen();
    }
}

static void dfs(Position& p, int depth) {
    if (depth == 0) return;
    MoveList ml; generate_legal_moves(p, ml);
    for (Move m : ml) {
        p.make_move(m);
        compare(p, "after make " + move_to_uci(m));
        dfs(p, depth - 1);
        p.unmake_move(m);
        compare(p, "after unmake " + move_to_uci(m));
    }
}

int main(int argc, char** argv) {
    init_zobrist(); init_attacks();
    const int H = NNUE::HIDDEN_SIZE;
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    };
    for (int fs : {NNUE_FS_HALFKP, NNUE_FS_HALFKA_V2}) for (int mir : {0, 1}) {
        write_hnn3("/tmp/t.hnn", fs, mir, H);
        if (!g_nnue.load("/tmp/t.hnn")) { printf("load failed fs=%d mir=%d\n", fs, mir); return 1; }
        checks = bad = 0; firstBad.clear();
        for (const char* fen : fens) { Position p; p.set_fen(fen); dfs(p, 3); }
        printf("[NNUE %-20s] checks=%lld mismatches=%lld %s\n", g_nnue.arch_name(), checks, bad, firstBad.c_str());
    }
    // Test 2: ván dài
    {
        Position p; p.set_startpos();
        const char* cyc[4] = {"g1f3", "g8f6", "f3g1", "f6g8"};
        for (int i = 0; i < 300; i++) {
            MoveList ml; generate_legal_moves(p, ml);
            for (Move m : ml) if (move_to_uci(m) == cyc[i % 4]) { p.make_move(m); break; }
        }
        // mở lại một nước có thể làm đổi quyền nhập thành / ep để thấy lệch
        int badKey = 0, total = 0;
        MoveList ml; generate_legal_moves(p, ml);
        for (Move m : ml) {
            std::string f0 = p.get_fen(); Bitboard k0 = p.key;
            p.make_move(m); p.unmake_move(m);
            total++;
            if (p.key != k0 || p.get_fen() != f0 || p.key != p.compute_key()) badKey++;
        }
        printf("[history] histCount=%d keyCount=%d  make/unmake roundtrip failures: %d / %d\n", p.histCount, p.keyCount, badKey, total);
    }
    return 0;
}
