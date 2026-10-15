#pragma once
#include "position.h"

namespace halcyon {

// Static Exchange Evaluation: simulates the full capture sequence on the
// destination square of `m` and returns the net material result (centipawns)
// for the side making the move. Used to prune/order captures.
int see(const Position& pos, Move m);

} // namespace halcyon
