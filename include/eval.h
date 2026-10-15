#pragma once
#include "position.h"

namespace halcyon {

// Must be called once at startup (after init_attacks) before evaluate() is used.
void init_eval();

// Centipawn score from the point of view of the side to move.
//
// Pipeline: exact tablebase result (<= 3 pieces, if tables are loaded) ->
// NNUE if one is loaded, otherwise the hand-crafted evaluation -> drawish-
// ending scaling (pawnless "can't convert" endings, opposite-coloured bishops,
// KNN vs K) -> fifty-move damping. The last two steps apply to either
// evaluator, so a network cannot overrate a known fortress/draw.
int evaluate(const Position& pos);

// The hand-crafted evaluation alone: no tablebase, no NNUE, no scaling.
// Used by tools that need labels independent of any network (self-play data
// generation, tuning) and by tests that compare evaluators.
int evaluate_hce(const Position& pos);

} // namespace halcyon
