#pragma once
#include "types.h"
#include <cstdint>

namespace halcyon {

// Standalone accumulator type, deliberately split out of nnue.h so that
// position.h can embed one directly (inside StateInfo, for the O(1)
// push/pop accumulator stack) without position.h having to include the
// full nnue.h -- which itself includes position.h. Keeping this struct
// dependency-free avoids that circular include entirely.
//
// Layout must stay in sync with NNUE::HIDDEN_SIZE in nnue.h (a static_assert
// in nnue.cpp enforces this at compile time).
struct NNUEAccumulator {
    // Hidden width of the feature transformer. Override at build time with
    // -DHALCYON_HIDDEN_SIZE=N (a networks file must be trained at exactly this
    // width -- the .hnn header carries it and load() rejects a mismatch).
    // Must be a multiple of 16 so every AVX2 row/accumulator access stays
    // 32-byte aligned and the SIMD loops have no scalar tail.
#ifndef HALCYON_HIDDEN_SIZE
#define HALCYON_HIDDEN_SIZE 1024
#endif
    static constexpr int HIDDEN_SIZE = HALCYON_HIDDEN_SIZE;
    static_assert(HIDDEN_SIZE % 16 == 0 && HIDDEN_SIZE > 0, "HIDDEN_SIZE must be a positive multiple of 16");
    alignas(32) int16_t v[COLOR_NB][HIDDEN_SIZE];
    bool computed = false;      // lazily refreshed the first time it's actually needed
    uint32_t weightsVersion = 0; // which set of weights v[] was computed against
};

} // namespace halcyon
