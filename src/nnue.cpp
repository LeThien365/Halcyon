#include "nnue.h"
#include "nnue_default_weights.h"
#include "bitboard.h"
#include <fstream>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>

#if defined(__AVX2__)
#include <immintrin.h>
#define HALCYON_USE_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define HALCYON_USE_SSE2 1
#endif

namespace halcyon {

NNUE g_nnue;

static_assert(NNUEAccumulator::HIDDEN_SIZE == NNUE::HIDDEN_SIZE,
              "nnue_types.h's NNUEAccumulator::HIDDEN_SIZE must match NNUE::HIDDEN_SIZE");

// Bumped every time weights are (re)loaded so any accumulator computed against
// older weights (or a different feature set) is recognised as stale and rebuilt.
static std::atomic<uint32_t> g_weightsVersion{1};

// ---------------------------------------------------------------------------
// Feature indexing.
//
// Every perspective sees the board from its own side: squares are flipped
// vertically for Black (sq ^ 56) so the network only ever learns "my pieces
// advance up the board". For the king-dependent sets the perspective's own
// king square, seen in the same frame, selects a block of the table.
//
//   LEGACY_PS : (own?0:384) + pieceType*64 + sq                      [768]
//   HALFKP    : kingBucket*640 + plane*64 + sq, plane = (own?0:5)+pt,
//               pt in P..Q (0..4); kings produce no feature        [64*640  or 32*640]
//   HALFKA_V2 : kingBucket*704 + plane*64 + sq, planes 0..9 as above,
//               plane 10 = king of either colour                   [64*704  or 32*704]
//
// Mirroring (optional): if the perspective's king is on files e-h, flip every
// square left/right (sq ^ 7, and the king square likewise) so the king is on
// files a-d and kingBucket = rank*4 + file fits in 32 buckets.
// ---------------------------------------------------------------------------
static inline int planes_of(int fset) { return fset == NNUE_FS_HALFKP ? 10 : 11; }

static inline int input_size_of(int fset, bool mirror) {
    if (fset == NNUE_FS_LEGACY_PS) return NNUE::LEGACY_INPUT_SIZE;
    return (mirror ? 32 : 64) * planes_of(fset) * 64;
}

static inline Square king_sq_of(const Position& pos, Color c) {
    const Bitboard k = pos.pieceBB[make_piece(c, KING)];
    return k ? lsb(k) : Square(A1);   // a king-less position is not legal chess; just don't crash
}

NNUE::PCtx NNUE::make_ctx(Color persp, Square ownKingSq) const {
    PCtx c;
    c.flip = (persp == WHITE) ? 0 : 56;
    c.orient = 0;
    c.base = 0;
    if (fset == NNUE_FS_LEGACY_PS) return c;
    Square ks = ownKingSq ^ c.flip;
    if (mirror && file_of(ks) >= 4) c.orient = 7;
    ks ^= c.orient;
    const int bucket = mirror ? (rank_of(ks) * 4 + file_of(ks)) : ks;
    c.base = bucket * planes_of(fset) * 64;
    return c;
}

int NNUE::feature(const PCtx& c, PieceType pt, bool isOwn, Square sq) const {
    const int s = (sq ^ c.flip) ^ c.orient;
    if (fset == NNUE_FS_LEGACY_PS) return (isOwn ? 0 : 384) + int(pt) * 64 + s;
    int plane;
    if (pt == KING) {
        if (fset == NNUE_FS_HALFKP) return -1;
        plane = 10;
    } else {
        plane = (isOwn ? 0 : 5) + int(pt);
    }
    return c.base + plane * 64 + s;
}

int NNUE::feature_index(Color persp, Square ownKingSq, PieceType pt, bool isOwn, Square sq) const {
    return feature(make_ctx(persp, ownKingSq), pt, isOwn, sq);
}

const char* NNUE::arch_name() const {
    switch (fset) {
        case NNUE_FS_HALFKP:    return mirror ? "HalfKP (mirrored)"   : "HalfKP";
        case NNUE_FS_HALFKA_V2: return mirror ? "HalfKAv2 (mirrored)" : "HalfKAv2";
        default:                return "768 (king-blind)";
    }
}

// ---------------------------------------------------------------------------
// Quantization -- standard fixed-point NNUE scheme:
//   feature transformer weights/bias * QA -> int16; accumulator in QA units;
//   clipped-ReLU clamps each lane to [0, QA]; output weights * QB -> int16,
//   dot product in int32; descale by QA*QB back to centipawns.
// ---------------------------------------------------------------------------
static inline int16_t quant_round_i16(float x) {
    float r = std::round(x);
    if (r > 32767.0f) r = 32767.0f;
    if (r < -32768.0f) r = -32768.0f;
    return (int16_t)r;
}

void NNUE::quantize() {
    const int H = HIDDEN_SIZE;
    qW1.assign(W1.size(), 0);
    qb1.assign(H, 0);
    for (size_t i = 0; i < W1.size(); i++) qW1[i] = quant_round_i16(W1[i] * QA);
    for (int h = 0; h < H; h++) qb1[h] = quant_round_i16(b1F[h] * QA);

    // W2 / b2F are always NUM_BUCKETS heads here: single-head sources were
    // broadcast by the caller.
    qW2.assign((size_t)NUM_BUCKETS * 2 * H, 0);
    qb2.assign(NUM_BUCKETS, 0);
    for (size_t i = 0; i < W2.size(); i++) qW2[i] = quant_round_i16(W2[i] * QB);
    for (int b = 0; b < NUM_BUCKETS; b++) qb2[b] = (int32_t)std::lround(b2F[b] * QA * QB);

    W1.clear(); W1.shrink_to_fit();
    b1F.clear(); b1F.shrink_to_fit();
    W2.clear(); W2.shrink_to_fit();
    b2F.clear(); b2F.shrink_to_fit();
}


// ---------------------------------------------------------------------------
// Worst-case arithmetic bounds (see acc_bound()/out_bound() in nnue.h).
//
// The accumulator is int16 and wraps SILENTLY above 32767; the output layer's
// partial sums are int32 and wrap silently above 2^31-1. Neither wraps for the
// networks tested so far, but "tested so far" says nothing about a freshly
// trained net at a different hidden width -- so every load computes a proof
// obligation instead of hoping:
//
//   * One perspective activates at most 32 features (one piece per square, at
//     most 32 pieces), so a lane's accumulator is bounded by
//       |bias| + (sum of the 32 largest per-square weight magnitudes),
//     where "per-square magnitude" is the max |weight| over the planes a
//     single square can hold (only one piece can stand there). For the
//     king-dependent sets the king bucket is fixed per perspective, so the
//     bound is taken per bucket and the worst bucket wins.
//   * The output dot product is bounded term by term: lane h contributes at
//     most accBound[h] * |w2[h]| (ReLU only removes negatives).
//
// Deliberately conservative -- it assumes the 32 largest rows co-occur. A
// result above the limit therefore means "cannot prove safe", not "overflows".
// ---------------------------------------------------------------------------
void NNUE::compute_bounds() {
    const int H = HIDDEN_SIZE;
    const bool legacy = (fset == NNUE_FS_LEGACY_PS);
    const int planes  = legacy ? 12 : planes_of(fset);
    const int buckets = legacy ? 1 : (mirror ? 32 : 64);

    std::vector<int> neuronBound(H, 0);   // worst over king buckets, per lane
    std::vector<int> best((size_t)64 * H);
    std::vector<int> vals(64);

    for (int b = 0; b < buckets; b++) {
        std::fill(best.begin(), best.end(), 0);
        for (int plane = 0; plane < planes; plane++) {
            for (int sq = 0; sq < 64; sq++) {
                const size_t row = legacy ? (size_t)plane * 64 + sq
                                          : ((size_t)b * planes + plane) * 64 + sq;
                const int16_t* r = &qW1[row * H];
                int* dst = &best[(size_t)sq * H];
                for (int h = 0; h < H; h++) {
                    const int a = r[h] < 0 ? -(int)r[h] : (int)r[h];
                    if (a > dst[h]) dst[h] = a;
                }
            }
        }
        for (int h = 0; h < H; h++) {
            for (int sq = 0; sq < 64; sq++) vals[sq] = best[(size_t)sq * H + h];
            std::nth_element(vals.begin(), vals.begin() + 32, vals.end(), std::greater<int>());
            long long sum = qb1[h] < 0 ? -(int)qb1[h] : (int)qb1[h];
            for (int i = 0; i < 32; i++) sum += vals[i];
            if (sum > neuronBound[h]) neuronBound[h] = (int)std::min<long long>(sum, 1LL << 30);
        }
    }

    int accMax = 0;
    for (int h = 0; h < H; h++) accMax = std::max(accMax, neuronBound[h]);
    accBound = accMax;

    long long worstOut = 0;
    for (int bk = 0; bk < NUM_BUCKETS; bk++) {
        for (int half = 0; half < 2; half++) {
            const int16_t* w = &qW2[((size_t)bk * 2 + half) * H];
            long long s = 0;
            for (int h = 0; h < H; h++) s += (long long)neuronBound[h] * (w[h] < 0 ? -(int)w[h] : (int)w[h]);
            worstOut = std::max(worstOut, s);
        }
    }
    outBound = worstOut;
}

// Finish a successful load: quantize, publish the architecture, invalidate
// every accumulator computed against the old weights.
bool NNUE::commit(int newSet, bool newMirror, int newInput) {
    quantize();
    fset = newSet;
    mirror = newMirror;
    inputSize = newInput;
    loaded = true;
    compute_bounds();
    g_weightsVersion.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void NNUE::load_default() {
    // The compiled-in weights in nnue_default_weights.h were trained at a
    // FIXED hidden size (DEFAULT_NNUE_HIDDEN_SIZE, see that header) -- they
    // are a specific set of learned numbers, not a formula that generalizes
    // to a different hidden size. If NNUE::HIDDEN_SIZE (the size this build
    // was compiled with, via NNUEAccumulator::HIDDEN_SIZE in nnue_types.h)
    // doesn't match, there is no safe way to "stretch" the compiled-in
    // weights to fit -- doing so would silently read past the end of those
    // fixed-size arrays (caught at compile time as an out-of-bounds warning
    // the one time this mismatch was introduced; this check makes the
    // failure explicit and safe at runtime for any future mismatch too,
    // instead of relying on a compiler warning being noticed). A build
    // configured for a different hidden size needs its OWN trained network
    // loaded via load(), not this compiled-in one.
    static_assert(DEFAULT_NNUE_HIDDEN_SIZE > 0, "nnue_default_weights.h must define DEFAULT_NNUE_HIDDEN_SIZE");
    if (DEFAULT_NNUE_HIDDEN_SIZE != HIDDEN_SIZE) {
        loaded = false;
        return;
    }

    const int H = HIDDEN_SIZE;
    W1.assign(DEFAULT_NNUE_W1, DEFAULT_NNUE_W1 + (size_t)LEGACY_INPUT_SIZE * H);
    b1F.assign(DEFAULT_NNUE_B1, DEFAULT_NNUE_B1 + H);

    // The compiled-in default predates output buckets (single head):
    // broadcast it to every bucket => identical behaviour to "one head".
    W2.assign((size_t)NUM_BUCKETS * 2 * H, 0.0f);
    for (int b = 0; b < NUM_BUCKETS; b++)
        std::copy(DEFAULT_NNUE_W2, DEFAULT_NNUE_W2 + 2 * H, W2.begin() + (size_t)b * 2 * H);
    b2F.assign(NUM_BUCKETS, DEFAULT_NNUE_B2);

    commit(NNUE_FS_LEGACY_PS, false, LEGACY_INPUT_SIZE);
}

static constexpr uint32_t NNUE_MAGIC_V1 = 0x484E4E31u; // "HNN1"
static constexpr uint32_t NNUE_MAGIC_V2 = 0x484E4E32u; // "HNN2"
static constexpr uint32_t NNUE_MAGIC_V3 = 0x484E4E33u; // "HNN3"

bool NNUE::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { loaded = false; return false; }

    const int H = HIDDEN_SIZE;
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!f) { loaded = false; return false; }

    int newSet = NNUE_FS_LEGACY_PS;
    bool newMirror = false;
    int nb = 1;   // number of output heads stored in the file

    if (magic == NNUE_MAGIC_V1 || magic == NNUE_MAGIC_V2) {
        int32_t inSize = 0, hidSize = 0;
        f.read(reinterpret_cast<char*>(&inSize), sizeof(inSize));
        f.read(reinterpret_cast<char*>(&hidSize), sizeof(hidSize));
        if (!f || inSize != LEGACY_INPUT_SIZE || hidSize != H) { loaded = false; return false; }
        if (magic == NNUE_MAGIC_V2) {
            int32_t numBuckets = 0;
            f.read(reinterpret_cast<char*>(&numBuckets), sizeof(numBuckets));
            if (!f || numBuckets != NUM_BUCKETS) { loaded = false; return false; }
            nb = NUM_BUCKETS;
        }
    } else if (magic == NNUE_MAGIC_V3) {
        int32_t setId = 0, flags = 0, hidSize = 0, numBuckets = 0;
        f.read(reinterpret_cast<char*>(&setId), sizeof(setId));
        f.read(reinterpret_cast<char*>(&flags), sizeof(flags));
        f.read(reinterpret_cast<char*>(&hidSize), sizeof(hidSize));
        f.read(reinterpret_cast<char*>(&numBuckets), sizeof(numBuckets));
        if (!f) { loaded = false; return false; }
        if ((setId != NNUE_FS_HALFKP && setId != NNUE_FS_HALFKA_V2) || (flags & ~1) != 0 ||
            hidSize != H || numBuckets != NUM_BUCKETS) { loaded = false; return false; }
        newSet = setId;
        newMirror = (flags & 1) != 0;
        nb = NUM_BUCKETS;
    } else {
        loaded = false;
        return false;
    }

    const int newInput = input_size_of(newSet, newMirror);

    // Read into scratch first; the live tables are untouched until commit().
    std::vector<float> w1((size_t)newInput * H), bias1(H), w2((size_t)nb * 2 * H), bias2((size_t)nb);
    f.read(reinterpret_cast<char*>(w1.data()), w1.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(bias1.data()), bias1.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(w2.data()), w2.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(bias2.data()), bias2.size() * sizeof(float));
    if (!f) { loaded = false; return false; }

    // A king-dependent net must end exactly here; extra bytes mean the file
    // was produced for a different layout and would be silently mis-read.
    if (magic == NNUE_MAGIC_V3 && f.peek() != std::ifstream::traits_type::eof()) { loaded = false; return false; }

    W1.swap(w1);
    b1F.swap(bias1);
    if (nb == NUM_BUCKETS) {
        W2.swap(w2);
        b2F.swap(bias2);
    } else {   // single-head legacy net -> broadcast to every bucket
        W2.assign((size_t)NUM_BUCKETS * 2 * H, 0.0f);
        for (int b = 0; b < NUM_BUCKETS; b++)
            std::copy(w2.begin(), w2.end(), W2.begin() + (size_t)b * 2 * H);
        b2F.assign(NUM_BUCKETS, bias2[0]);
    }
    return commit(newSet, newMirror, newInput);
}

// ---------------------------------------------------------------------------
// Output bucket: a pure function of how many pieces remain, so a network can
// give the endgame its own output head without a separate feature transformer.
// ---------------------------------------------------------------------------
int NNUE::bucket_of(const Position& pos) {
    int pieceCount = popcount(pos.occAll);
    int b = (pieceCount - 1) / 4;
    if (b < 0) b = 0;
    if (b >= NUM_BUCKETS) b = NUM_BUCKETS - 1;
    return b;
}

// ---------------------------------------------------------------------------
// SIMD kernels. Every kernel has a scalar twin so behaviour (not just
// compilation) is guaranteed identical whether or not HALCYON_USE_AVX2 is
// defined -- the AVX2 path is purely a speed optimization of the exact same
// integer arithmetic, never an approximation of it.
// ---------------------------------------------------------------------------
namespace {

inline void accum_add_scalar(int16_t* __restrict acc, const int16_t* __restrict row, int n) {
    for (int i = 0; i < n; i++) acc[i] = (int16_t)(acc[i] + row[i]);
}
inline void accum_sub_scalar(int16_t* __restrict acc, const int16_t* __restrict row, int n) {
    for (int i = 0; i < n; i++) acc[i] = (int16_t)(acc[i] - row[i]);
}

#if defined(HALCYON_USE_AVX2)
inline void accum_add_avx2(int16_t* acc, const int16_t* row, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i*)(acc + i));
        __m256i r = _mm256_loadu_si256((const __m256i*)(row + i));
        _mm256_store_si256((__m256i*)(acc + i), _mm256_add_epi16(a, r));
    }
    for (; i < n; i++) acc[i] = (int16_t)(acc[i] + row[i]);
}
inline void accum_sub_avx2(int16_t* acc, const int16_t* row, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i*)(acc + i));
        __m256i r = _mm256_loadu_si256((const __m256i*)(row + i));
        _mm256_store_si256((__m256i*)(acc + i), _mm256_sub_epi16(a, r));
    }
    for (; i < n; i++) acc[i] = (int16_t)(acc[i] - row[i]);
}
#endif

#if defined(HALCYON_USE_SSE2) && !defined(HALCYON_USE_AVX2)
// Portable SSE2 path: works on virtually all x86-64 Windows/Linux CPUs without AVX2/512.
inline void accum_add_sse2(int16_t* acc, const int16_t* row, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a = _mm_loadu_si128((const __m128i*)(acc + i));
        __m128i r = _mm_loadu_si128((const __m128i*)(row + i));
        _mm_storeu_si128((__m128i*)(acc + i), _mm_add_epi16(a, r));
    }
    for (; i < n; i++) acc[i] = (int16_t)(acc[i] + row[i]);
}
inline void accum_sub_sse2(int16_t* acc, const int16_t* row, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a = _mm_loadu_si128((const __m128i*)(acc + i));
        __m128i r = _mm_loadu_si128((const __m128i*)(row + i));
        _mm_storeu_si128((__m128i*)(acc + i), _mm_sub_epi16(a, r));
    }
    for (; i < n; i++) acc[i] = (int16_t)(acc[i] - row[i]);
}
#endif

inline void accum_add(int16_t* __restrict acc, const int16_t* __restrict row, int n) {
#if defined(HALCYON_USE_AVX2)
    accum_add_avx2(acc, row, n);
#elif defined(HALCYON_USE_SSE2)
    accum_add_sse2(acc, row, n);
#else
    accum_add_scalar(acc, row, n);
#endif
}
inline void accum_sub(int16_t* __restrict acc, const int16_t* __restrict row, int n) {
#if defined(HALCYON_USE_AVX2)
    accum_sub_avx2(acc, row, n);
#elif defined(HALCYON_USE_SSE2)
    accum_sub_sse2(acc, row, n);
#else
    accum_sub_scalar(acc, row, n);
#endif
}

// ReLU dot product: clamp each accumulator lane to >= 0 only (NO upper
// bound). The compiled-in trained network (and any legacy-format net
// derived from it) was trained with plain ReLU, not clipped-ReLU -- its
// pre-activation values legitimately exceed the "1.0" (== QA in quantized
// units) a clipped-ReLU network would be trained to stay under (measured
// up to ~4.7x over 300 random games; a well-trained clipped-ReLU network by
// contrast rarely if ever approaches its own ceiling, since training
// actively discourages saturating it). Clamping here would silently
// throw away real signal and desync inference from what the net learned.
// `int32_t` per-lane accumulation stays safe: even a generously-bounded
// v ~ 8*QA (way beyond anything observed) times a saturated int16 weight
// is still four orders of magnitude under the int32 range, per lane, well
// before the horizontal add.
inline int32_t crelu_dot_scalar(const int16_t* __restrict acc, const int16_t* __restrict w, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t v = acc[i];
        if (v < 0) v = 0;
        sum += v * (int32_t)w[i];
    }
    return sum;
}

#if defined(HALCYON_USE_AVX2)
inline int32_t crelu_dot_avx2(const int16_t* acc, const int16_t* w, int n) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i accSum = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(acc + i));
        a = _mm256_max_epi16(a, zero);   // ReLU only -- no upper clamp, see above
        __m256i wv = _mm256_loadu_si256((const __m256i*)(w + i));
        // madd: pairs of int16*int16 -> int32, horizontally added per pair.
        __m256i prod = _mm256_madd_epi16(a, wv);
        accSum = _mm256_add_epi32(accSum, prod);
    }
    alignas(32) int32_t lanes[8];
    _mm256_store_si256((__m256i*)lanes, accSum);
    int32_t sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < n; i++) {
        int32_t v = acc[i];
        if (v < 0) v = 0;
        sum += v * (int32_t)w[i];
    }
    return sum;
}
#endif

#if defined(HALCYON_USE_SSE2) && !defined(HALCYON_USE_AVX2)
inline int32_t crelu_dot_sse2(const int16_t* acc, const int16_t* w, int n) {
    const __m128i zero = _mm_setzero_si128();
    __m128i accSum = _mm_setzero_si128();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a = _mm_loadu_si128((const __m128i*)(acc + i));
        a = _mm_max_epi16(a, zero);
        __m128i wv = _mm_loadu_si128((const __m128i*)(w + i));
        __m128i prod = _mm_madd_epi16(a, wv);
        accSum = _mm_add_epi32(accSum, prod);
    }
    alignas(16) int32_t lanes[4];
    _mm_store_si128((__m128i*)lanes, accSum);
    int32_t sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; i++) {
        int32_t v = acc[i];
        if (v < 0) v = 0;
        sum += v * (int32_t)w[i];
    }
    return sum;
}
#endif

inline int32_t crelu_dot(const int16_t* __restrict acc, const int16_t* __restrict w, int n) {
#if defined(HALCYON_USE_AVX2)
    return crelu_dot_avx2(acc, w, n);
#elif defined(HALCYON_USE_SSE2)
    return crelu_dot_sse2(acc, w, n);
#else
    return crelu_dot_scalar(acc, w, n);
#endif
}

} // namespace

// ---------------------------------------------------------------------------
// Accumulator maintenance.
// ---------------------------------------------------------------------------
void NNUE::refresh_perspective(const Position& pos, Accumulator& acc, Color persp) const {
    if (qW1.empty()) return;
    const int H = HIDDEN_SIZE;
    int16_t* a = acc.v[int(persp)];
    std::memcpy(a, qb1.data(), sizeof(int16_t) * H);

    const PCtx ctx = make_ctx(persp, king_sq_of(pos, persp));
    for (int p = 0; p < PIECE_NB; p++) {
        const Piece piece = Piece(p);
        const bool isOwn = (color_of(piece) == persp);
        const PieceType pt = type_of(piece);
        Bitboard bb = pos.pieceBB[piece];
        while (bb) {
            const Square sq = pop_lsb(bb);
            const int idx = feature(ctx, pt, isOwn, sq);
            if (idx >= 0) accum_add(a, &qW1[(size_t)idx * H], H);
        }
    }
}

void NNUE::refresh(const Position& pos, Accumulator& acc) const {
    if (qW1.empty()) { acc.computed = false; return; }   // nothing loaded yet: evaluate() retries later
    refresh_perspective(pos, acc, WHITE);
    refresh_perspective(pos, acc, BLACK);
    acc.computed = true;
    acc.weightsVersion = g_weightsVersion.load(std::memory_order_relaxed);
}

void NNUE::update_add(Accumulator& acc, Piece pc, Square sq, Square wksq, Square bksq, int skipPersp) const {
    const int H = HIDDEN_SIZE;
    const PieceType pt = type_of(pc);
    const Color pieceColor = color_of(pc);
    for (int persp = 0; persp < 2; persp++) {
        if (persp == skipPersp) continue;
        const Color c = Color(persp);
        const int idx = feature(make_ctx(c, persp == WHITE ? wksq : bksq), pt, pieceColor == c, sq);
        if (idx >= 0) accum_add(acc.v[persp], &qW1[(size_t)idx * H], H);
    }
}

void NNUE::update_sub(Accumulator& acc, Piece pc, Square sq, Square wksq, Square bksq, int skipPersp) const {
    const int H = HIDDEN_SIZE;
    const PieceType pt = type_of(pc);
    const Color pieceColor = color_of(pc);
    for (int persp = 0; persp < 2; persp++) {
        if (persp == skipPersp) continue;
        const Color c = Color(persp);
        const int idx = feature(make_ctx(c, persp == WHITE ? wksq : bksq), pt, pieceColor == c, sq);
        if (idx >= 0) accum_sub(acc.v[persp], &qW1[(size_t)idx * H], H);
    }
}

int NNUE::output_from(const Accumulator& acc, Color sideToMove, int bucket) const {
    const int H = HIDDEN_SIZE;
    const int16_t* accUs   = acc.v[int(sideToMove)];
    const int16_t* accThem = acc.v[int(~sideToMove)];
    const int16_t* wUs   = &qW2[(size_t)bucket * 2 * H];
    const int16_t* wThem = wUs + H;

    int64_t sum = qb2[bucket];
    sum += crelu_dot(accUs, wUs, H);
    sum += crelu_dot(accThem, wThem, H);

    // Descale out of both quantization scales, THEN apply the same *100
    // "raw network output -> centipawns" convention the original (pre-
    // quantization) float implementation used -- i.e. the true target here
    // is (sum / (QA*QB)) * 100, not sum / (QA*QB). Multiplying by 100 before
    // dividing (rather than after) keeps the intermediate exact in integer
    // math instead of rounding twice; denom absorbs the /100 the other way.
    // Headroom check: |sum| stays well under 2^40 for any realistic
    // accumulator/weight range, so sum*100 cannot overflow int64.
    const int64_t denom = (int64_t)QA * QB;
    sum *= 100;
    if (sum >= 0) return (int)((sum + denom / 2) / denom);
    else          return (int)(-((-sum + denom / 2) / denom));
}

// ---------------------------------------------------------------------------
// evaluate(): Position keeps pos.nnueAcc current through make/unmake, so this
// is normally just a bucket lookup and one dot product. It rebuilds only on a
// cold start, or when the loaded network changed since the accumulator was
// computed (new weights and/or a different feature set).
// ---------------------------------------------------------------------------
int NNUE::evaluate(const Position& pos) const {
    Accumulator& acc = pos.nnueAcc;
    const uint32_t ver = g_weightsVersion.load(std::memory_order_relaxed);

    if (!acc.computed || acc.weightsVersion != ver)
        refresh(pos, acc);
    if (!acc.computed) return 0;   // no weights loaded at all: nothing to evaluate with

    return output_from(acc, pos.sideToMove, bucket_of(pos));
}

} // namespace halcyon
