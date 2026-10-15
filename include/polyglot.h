#pragma once
#include "position.h"
#include "movegen.h"
#include <string>
#include <vector>
#include <cstdint>
#include <random>

namespace halcyon {

// Computes the Polyglot-format Zobrist key for a position. This is a
// *different* hash from the engine's own internal Zobrist key (halcyon/zobrist.h) --
// it exists purely to be compatible with the public Polyglot book format
// (Fabien Letouzey's format, GPL, reimplemented independently by countless
// engines and GUIs), so that any standard .bin opening book can be read.
uint64_t polyglot_key(const Position& pos);

struct PolyglotEntry {
    uint64_t key;
    uint16_t move;
    uint16_t weight;
    uint32_t learn;
};

class PolyglotBook {
public:
    bool load(const std::string& path);
    bool is_loaded() const { return loaded; }
    size_t size() const { return entries.size(); }

    // Returns MOVE_NONE if the book has no move for this position.
    // Selection is weighted-random over all matching entries (ties broken
    // by trying the next-heaviest entry if a candidate doesn't decode to a
    // legal move in this position).
    Move probe(Position& pos, std::mt19937_64& rng) const;

private:
    bool loaded = false;
    std::vector<PolyglotEntry> entries; // kept sorted by key for binary search
};

extern PolyglotBook g_book;

} // namespace halcyon
