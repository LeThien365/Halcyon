// nnue_convert -- expand Halcyon's king-blind 768-input net into an HNN3 HalfKAv2 net
// that evaluates IDENTICALLY (bit-for-bit) to the source net.
//
// Why this is exact: in HalfKAv2 the kings share one plane (plane 10). For king-bucket
// `ks` (the perspective's own king, in that perspective's frame) the own king is always
// ON square ks and the enemy king can never be, so the two legacy king weights never
// collide:   plane10[ks][s] = (s == ks) ? legacy_own_king[s] : legacy_enemy_king[s].
// Every other plane is copied unchanged into every king bucket.
//
// The result is a valid warm start: same strength as the source net on day one, but with
// king-conditioned capacity that training can now specialise. (HalfKP has no such exact
// conversion -- it discards kings entirely -- so it needs training from scratch.)
//
// usage:  nnue_convert --out halfkav2.hnn [--in legacy.hnn]
//         (no --in => use the network compiled into the engine)
#include "nnue_default_weights.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static constexpr int HID = 256, LEG_IN = 768, NB = 8;
static constexpr int PLANES = 11, BUCKETS = 64;

int main(int argc, char** argv) {
    std::string inPath, outPath;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--in") && i + 1 < argc) inPath = argv[++i];
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) outPath = argv[++i];
        else { std::cerr << "usage: nnue_convert --out file.hnn [--in legacy.hnn]\n"; return 2; }
    }
    if (outPath.empty()) { std::cerr << "missing --out\n"; return 2; }

    std::vector<float> W1((size_t)LEG_IN * HID), b1(HID), W2((size_t)NB * 2 * HID), b2(NB);
    if (inPath.empty()) {
        std::copy(halcyon::DEFAULT_NNUE_W1, halcyon::DEFAULT_NNUE_W1 + W1.size(), W1.begin());
        std::copy(halcyon::DEFAULT_NNUE_B1, halcyon::DEFAULT_NNUE_B1 + HID, b1.begin());
        for (int b = 0; b < NB; b++)
            std::copy(halcyon::DEFAULT_NNUE_W2, halcyon::DEFAULT_NNUE_W2 + 2 * HID, W2.begin() + (size_t)b * 2 * HID);
        std::fill(b2.begin(), b2.end(), halcyon::DEFAULT_NNUE_B2);
    } else {
        std::ifstream f(inPath, std::ios::binary);
        uint32_t magic = 0; int32_t in = 0, hid = 0, nb = 1;
        f.read((char*)&magic, 4); f.read((char*)&in, 4); f.read((char*)&hid, 4);
        if (!f || (magic != 0x484E4E31u && magic != 0x484E4E32u) || in != LEG_IN || hid != HID) {
            std::cerr << "not a legacy HNN1/HNN2 file (768 inputs, hidden 256)\n"; return 1;
        }
        if (magic == 0x484E4E32u) { f.read((char*)&nb, 4); if (nb != NB) { std::cerr << "bad bucket count\n"; return 1; } }
        std::vector<float> w2((size_t)nb * 2 * HID), bb((size_t)nb);
        f.read((char*)W1.data(), W1.size() * 4); f.read((char*)b1.data(), HID * 4);
        f.read((char*)w2.data(), w2.size() * 4); f.read((char*)bb.data(), bb.size() * 4);
        if (!f) { std::cerr << "truncated input file\n"; return 1; }
        for (int b = 0; b < NB; b++) {
            const size_t src = (nb == NB) ? (size_t)b : 0;
            std::copy(w2.begin() + src * 2 * HID, w2.begin() + (src + 1) * 2 * HID, W2.begin() + (size_t)b * 2 * HID);
            b2[b] = bb[src];
        }
    }

    // Expand. Legacy layout: (own ? 0 : 384) + pieceType*64 + s, pieceType P,N,B,R,Q,K = 0..5.
    std::vector<float> out((size_t)BUCKETS * PLANES * 64 * HID);
    auto legacyRow = [&](int idx) { return &W1[(size_t)idx * HID]; };
    for (int ks = 0; ks < BUCKETS; ks++)
        for (int plane = 0; plane < PLANES; plane++)
            for (int s = 0; s < 64; s++) {
                int legacyIdx;
                if (plane < 5)        legacyIdx = 0   + plane * 64 + s;          // own P..Q
                else if (plane < 10)  legacyIdx = 384 + (plane - 5) * 64 + s;    // enemy P..Q
                else                  legacyIdx = (s == ks ? 0 : 384) + 5 * 64 + s;  // shared king plane
                std::memcpy(&out[(((size_t)ks * PLANES + plane) * 64 + s) * HID], legacyRow(legacyIdx), HID * sizeof(float));
            }

    std::ofstream o(outPath, std::ios::binary);
    const uint32_t magic = 0x484E4E33u; const int32_t fset = 2, flags = 0, hid = HID, nb = NB;
    o.write((const char*)&magic, 4); o.write((const char*)&fset, 4); o.write((const char*)&flags, 4);
    o.write((const char*)&hid, 4);   o.write((const char*)&nb, 4);
    o.write((const char*)out.data(), out.size() * 4);
    o.write((const char*)b1.data(), b1.size() * 4);
    o.write((const char*)W2.data(), W2.size() * 4);
    o.write((const char*)b2.data(), b2.size() * 4);
    if (!o) { std::cerr << "write failed\n"; return 1; }
    std::cout << "wrote " << outPath << " (HalfKAv2, " << (out.size() * 4 / (1024 * 1024)) << " MB of feature weights)\n";
    return 0;
}
