#pragma once
// Halcyon's own endgame tablebase file format ("HTBZ"), independently
// designed -- not Syzygy's .rtbw/.rtbz format, not derived from it.
//
// v0 layout (this file): intentionally UNCOMPRESSED. The priority for this
// first version is verifying the retrograde-analysis algorithm and the
// probing code end-to-end are correct; compression is a pure disk-space
// optimization that can be added later (v1) once correctness is nailed
// down, without changing anything about how the generator computes results.
// A 3-piece table is small enough (1.5 MB raw) that shipping it uncompressed
// is completely fine for now.
//
//   uint32 magic       = 0x48544233 ("HTB3", 3-piece table)
//   uint32 version      = 0
//   uint8  pieceKind     (0=Q,1=R,2=B,3=N -- see PieceKind in halcyon_tb_common.h)
//   uint8  reserved[3]   = 0
//   uint8  result[64*64*64*2]   -- Result enum value per (wk,bk,x,stm) index
//   uint16 dtm[64*64*64*2]      -- little-endian; meaningful only where
//                                  result == WIN or LOSS, 0 otherwise
//
// Index function (must match the generator and the prober exactly):
//   index(wk, bk, x, stm) = ((wk*64 + bk)*64 + x)*2 + stm
//
// KPvK ("HTBP") is a second, separate format using the same index space and
// result/dtm layout (minus the pieceKind byte, since there is no PieceKind
// for a pawn -- its move rules and promotion handling are different enough
// that it isn't just another value of that enum). See the generator
// (halcyon_tb_gen3_pawn.cpp) for how promotion is resolved by cross-
// referencing an already-generated KQvK table.
#include "halcyon_tb_common.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace halcyon { namespace tb {

constexpr uint32_t HTB3_MAGIC = 0x48544233u; // "HTB3"
constexpr uint32_t HTBP_MAGIC = 0x48544250u; // "HTBP"
constexpr uint32_t HTB3_VERSION = 0;
constexpr size_t HTB3_ENTRIES = 64u * 64u * 64u * 2u;

struct Table3 {
    // For a piece table, the PieceKind that was loaded. For a pawn table,
    // set to the sentinel PK_NB (there is no PieceKind for "pawn") --
    // callers that might receive either kind of table should check this
    // before trusting `kind` as a real PieceKind.
    PieceKind kind;
    std::vector<Result> result;
    std::vector<uint16_t> dtm;

    Result at(int wk, int bk, int x, Color stm) const {
        return result[(((size_t)wk * 64 + bk) * 64 + x) * 2 + (size_t)stm];
    }
    uint16_t dtm_at(int wk, int bk, int x, Color stm) const {
        return dtm[(((size_t)wk * 64 + bk) * 64 + x) * 2 + (size_t)stm];
    }
};

inline void write_table3(const std::string& path, PieceKind kind,
                          const std::vector<Result>& result, const std::vector<uint16_t>& dtm) {
    if (result.size() != HTB3_ENTRIES || dtm.size() != HTB3_ENTRIES)
        throw std::runtime_error("write_table3: size mismatch");
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_table3: cannot open " + path);
    uint32_t magic = HTB3_MAGIC, ver = HTB3_VERSION;
    uint8_t pk = (uint8_t)kind, resv[3] = {0, 0, 0};
    f.write((const char*)&magic, 4);
    f.write((const char*)&ver, 4);
    f.write((const char*)&pk, 1);
    f.write((const char*)resv, 3);
    std::vector<uint8_t> resultBytes(HTB3_ENTRIES);
    for (size_t i = 0; i < HTB3_ENTRIES; i++) resultBytes[i] = (uint8_t)result[i];
    f.write((const char*)resultBytes.data(), (std::streamsize)resultBytes.size());
    f.write((const char*)dtm.data(), (std::streamsize)(dtm.size() * sizeof(uint16_t)));
    if (!f) throw std::runtime_error("write_table3: write failed");
}

inline bool read_table3(const std::string& path, Table3& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = 0, ver = 0;
    f.read((char*)&magic, 4);
    f.read((char*)&ver, 4);
    if (!f || magic != HTB3_MAGIC || ver != HTB3_VERSION) return false;
    uint8_t pk = 0, resv[3];
    f.read((char*)&pk, 1);
    f.read((char*)resv, 3);
    if (!f || pk >= PK_NB) return false;
    out.kind = (PieceKind)pk;
    std::vector<uint8_t> resultBytes(HTB3_ENTRIES);
    f.read((char*)resultBytes.data(), (std::streamsize)resultBytes.size());
    if (!f) return false;
    out.result.assign(HTB3_ENTRIES, Result::ILLEGAL);
    for (size_t i = 0; i < HTB3_ENTRIES; i++) out.result[i] = (Result)resultBytes[i];
    out.dtm.assign(HTB3_ENTRIES, 0);
    f.read((char*)out.dtm.data(), (std::streamsize)(out.dtm.size() * sizeof(uint16_t)));
    if (!f) return false;
    // Must end exactly here.
    f.peek();
    if (!f.eof()) return false;
    return true;
}

inline void write_table3_pawn(const std::string& path,
                               const std::vector<Result>& result, const std::vector<uint16_t>& dtm) {
    if (result.size() != HTB3_ENTRIES || dtm.size() != HTB3_ENTRIES)
        throw std::runtime_error("write_table3_pawn: size mismatch");
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_table3_pawn: cannot open " + path);
    uint32_t magic = HTBP_MAGIC, ver = HTB3_VERSION;
    f.write((const char*)&magic, 4);
    f.write((const char*)&ver, 4);
    std::vector<uint8_t> resultBytes(HTB3_ENTRIES);
    for (size_t i = 0; i < HTB3_ENTRIES; i++) resultBytes[i] = (uint8_t)result[i];
    f.write((const char*)resultBytes.data(), (std::streamsize)resultBytes.size());
    f.write((const char*)dtm.data(), (std::streamsize)(dtm.size() * sizeof(uint16_t)));
    if (!f) throw std::runtime_error("write_table3_pawn: write failed");
}

inline bool read_table3_pawn(const std::string& path, Table3& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = 0, ver = 0;
    f.read((char*)&magic, 4);
    f.read((char*)&ver, 4);
    if (!f || magic != HTBP_MAGIC || ver != HTB3_VERSION) return false;
    out.kind = PK_NB;   // sentinel: "this is a pawn table, not a PieceKind table"
    std::vector<uint8_t> resultBytes(HTB3_ENTRIES);
    f.read((char*)resultBytes.data(), (std::streamsize)resultBytes.size());
    if (!f) return false;
    out.result.assign(HTB3_ENTRIES, Result::ILLEGAL);
    for (size_t i = 0; i < HTB3_ENTRIES; i++) out.result[i] = (Result)resultBytes[i];
    out.dtm.assign(HTB3_ENTRIES, 0);
    f.read((char*)out.dtm.data(), (std::streamsize)(out.dtm.size() * sizeof(uint16_t)));
    if (!f) return false;
    f.peek();
    if (!f.eof()) return false;
    return true;
}

} } // namespace halcyon::tb
