#pragma once
#include "position.h"
#include "nnue_types.h"
#include <string>
#include <vector>
#include <cstdint>

namespace halcyon {

// ============================================================================
//  NNUE -- Halcyon's own network format (NOT Stockfish's architecture or file
//  format; independently designed and implemented).
//
//  Every network shares the same body:
//
//     features -> [shared feature transformer, HIDDEN_SIZE wide]
//              -> clipped-ReLU (once per perspective: side to move, opponent)
//              -> concat (2*HIDDEN_SIZE)
//              -> one of NUM_BUCKETS linear output heads (chosen by piece count)
//              -> centipawns
//
//  and differs only in how a (piece, square) pair becomes a feature index --
//  the "feature set". Three are supported, selected by the weight file:
//
//   LEGACY_PS   768 inputs. (own/enemy) x (6 piece types) x 64 squares.
//               Blind to where the kings are. This is the format of the
//               compiled-in default network and of HNN1/HNN2 files.
//
//   HALFKP      Each perspective's features are conditioned on the square of
//               THAT perspective's own king: (king square) x (10 planes:
//               own/enemy x P,N,B,R,Q) x 64 squares. Kings are not features
//               (their square is already the conditioning context).
//
//   HALFKA_V2   Like HALFKP but the kings ARE features, merged into a single
//               11th plane shared by both colours. This is lossless: the
//               perspective's own king always sits on the bucket's king
//               square and the enemy king never can, so the two cases never
//               collide inside the shared plane.
//
//  Either king-dependent set can be "mirrored" (flag bit 0 in HNN3): the
//  board is flipped left/right whenever the perspective's king stands on
//  files e-h, so the king always lives on files a-d. That halves the king
//  buckets (32 instead of 64) and therefore the table size, and lets every
//  weight see twice as much training data.
//
//  The names HalfKP / HalfKAv2 describe the general *idea* of king-relative
//  input features; the exact index layout and file format here are
//  Halcyon's own and are not interchangeable with any other engine's nets.
//
//  What king-dependence costs at runtime: when a king moves, every feature of
//  that side changes, so that perspective's accumulator is rebuilt from
//  scratch (Position::make_move does this). All other moves stay O(1)
//  incremental. The tables are also much larger (see input_size()) so they
//  live further from the CPU than the 768-input net -- expect lower NPS in
//  exchange for the extra eval accuracy a trained net provides.
//
//  Weight files (all little-endian float32, quantized to int16 at load time):
//    HNN1  uint32 magic "HNN1", int32 768, int32 hidden, W1, b1, W2[2*hidden], b2
//    HNN2  as HNN1 but int32 numBuckets after hidden, W2[numBuckets][2*hidden], b2[numBuckets]
//    HNN3  uint32 magic "HNN3", int32 featureSet (1=HalfKP, 2=HalfKAv2),
//          int32 flags (bit0 = mirrored), int32 hidden, int32 numBuckets,
//          W1[inputSize][hidden], b1[hidden],
//          W2[numBuckets][2*hidden], b2[numBuckets]
//    HNN3 files must end exactly after b2 (trailing bytes => rejected), so a
//    net trained for a different layout can't be silently mis-read.
// ============================================================================
enum NNUEFeatureSet : int {
    NNUE_FS_LEGACY_PS = 0,
    NNUE_FS_HALFKP    = 1,
    NNUE_FS_HALFKA_V2 = 2
};

class NNUE {
public:
    static constexpr int LEGACY_INPUT_SIZE = 768;
    static constexpr int INPUT_SIZE  = LEGACY_INPUT_SIZE;   // kept for source compatibility
    static constexpr int HIDDEN_SIZE = NNUEAccumulator::HIDDEN_SIZE;
    static constexpr int NUM_BUCKETS = 8;      // indexed by (popcount(all pieces)-1)/4
    static constexpr int QA          = 255;    // feature-transformer quantization scale
    static constexpr int QB          = 64;     // output-layer quantization scale
    // No activation ceiling: the compiled-in net (and legacy-format nets
    // derived from it) were trained with plain ReLU, whose pre-activation
    // values legitimately exceed 1.0 (== QA in quantized units) -- see the
    // comment above crelu_dot_scalar() in nnue.cpp. A future net trained
    // FOR clipped-ReLU from the start could reintroduce a ceiling, but that
    // is a training-time decision, not something to bolt on at inference.

    bool load(const std::string& path);   // HNN1 / HNN2 / HNN3
    void load_default();                  // compiled-in trained legacy net (always succeeds)
    void set_enabled(bool e) { enabled = e; }
    bool is_loaded() const { return loaded && enabled; }

    // centipawns, from side-to-move perspective.
    int evaluate(const Position& pos) const;

    // ---- Architecture of the currently loaded network ---------------------
    int         feature_set() const    { return fset; }
    bool        king_dependent() const { return fset != NNUE_FS_LEGACY_PS; }
    bool        mirrored() const       { return mirror; }
    int         input_size() const     { return inputSize; }

    // Worst-case arithmetic bounds of the currently loaded network, computed
    // once at load time (see compute_bounds() in nnue.cpp). Conservative: they
    // assume the 32 largest-magnitude feature rows of every neuron are active
    // at once, which no real position can do -- so a bound above the limit
    // means "cannot PROVE safe", not "will overflow".
    //   acc_bound() : max |accumulator lane| any position could produce; the
    //                 int16 accumulator wraps silently above 32767.
    //   out_bound() : max |partial dot product| in the int32 output layer;
    //                 wraps above 2147483647.
    int         acc_bound() const      { return accBound; }
    long long   out_bound() const      { return outBound; }
    bool        arithmetic_safe() const { return accBound <= 32767 && outBound < 2147483647LL; }
    const char* arch_name() const;

    // ---- Incremental accumulator API (driven by Position) -----------------
    using Accumulator = NNUEAccumulator;

    // Full rebuild of both perspectives; stamps the accumulator as computed
    // against the current weights. If no weights are loaded yet it leaves the
    // accumulator marked "not computed" (evaluate() will retry later).
    void refresh(const Position& pos, Accumulator& acc) const;

    // Rebuild ONE perspective (used when that side's king moves). Does not
    // touch the other perspective and does not stamp the weights version.
    void refresh_perspective(const Position& pos, Accumulator& acc, Color persp) const;

    // Add/remove one piece. wksq/bksq are the two kings' squares in the
    // position the caller is updating from; the feature index of each
    // perspective depends on that perspective's own king. `skipPersp`
    // (0/1, or -1 for none) omits a perspective the caller is about to
    // rebuild wholesale anyway.
    void update_add(Accumulator& acc, Piece pc, Square sq, Square wksq, Square bksq, int skipPersp = -1) const;
    void update_sub(Accumulator& acc, Piece pc, Square sq, Square wksq, Square bksq, int skipPersp = -1) const;

    int  output_from(const Accumulator& acc, Color sideToMove, int bucket) const;
    static int bucket_of(const Position& pos);

    // Feature index for one (piece, square) as seen from `persp` given that
    // perspective's own king square (absolute board coordinates). Returns -1
    // if the piece is not a feature under the current feature set (kings in
    // HalfKP). Exposed for tools and tests.
    int feature_index(Color persp, Square ownKingSq, PieceType pt, bool isOwn, Square sq) const;

private:
    bool loaded = false;
    bool enabled = true;

    // Architecture of the loaded net. Only ever changed at the very end of a
    // successful load, so a failed load can't leave these half-updated.
    int  fset = NNUE_FS_LEGACY_PS;
    bool mirror = false;
    int  inputSize = LEGACY_INPUT_SIZE;
    int       accBound = 0;
    long long outBound = 0;

    // Float weights as loaded (kept only long enough to quantize, then freed).
    std::vector<float> W1, b1F;
    std::vector<float> W2, b2F;

    // Quantized weights actually used at inference time.
    std::vector<int16_t> qW1;              // [inputSize][HIDDEN_SIZE]
    std::vector<int16_t> qb1;              // [HIDDEN_SIZE]
    std::vector<int16_t> qW2;              // [NUM_BUCKETS][2*HIDDEN_SIZE]
    std::vector<int32_t> qb2;              // [NUM_BUCKETS]

    struct PCtx { int base; int orient; int flip; };
    PCtx make_ctx(Color persp, Square ownKingSq) const;
    int  feature(const PCtx& c, PieceType pt, bool isOwn, Square sq) const;

    void quantize();
    void compute_bounds();
    bool commit(int newSet, bool newMirror, int newInput);
};

extern NNUE g_nnue;

} // namespace halcyon
