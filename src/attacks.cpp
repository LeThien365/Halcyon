#include "attacks.h"
#include <cstring>
#include <cstdio>
#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace halcyon {

Bitboard PAWN_ATTACKS[COLOR_NB][SQUARE_NB];
Bitboard KNIGHT_ATTACKS[SQUARE_NB];
Bitboard KING_ATTACKS[SQUARE_NB];

static Bitboard BISHOP_MASKS[SQUARE_NB];
static Bitboard ROOK_MASKS[SQUARE_NB];
static int      BISHOP_BITS[SQUARE_NB];
static int      ROOK_BITS[SQUARE_NB];
static Bitboard BISHOP_MAGICS[SQUARE_NB];
static Bitboard ROOK_MAGICS[SQUARE_NB];
static Bitboard BISHOP_TABLE[SQUARE_NB][512];
static Bitboard ROOK_TABLE[SQUARE_NB][4096];

// ---------- leaper attacks, computed via file/rank deltas ----------

static Bitboard pawn_deltas(Color c, Square sq) {
    Bitboard attacks = 0;
    int f = file_of(sq), r = rank_of(sq);
    int dr = (c == WHITE) ? 1 : -1;
    for (int df : {-1, 1}) {
        int nf = f + df, nr = r + dr;
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) set_bit(attacks, make_square(nf, nr));
    }
    return attacks;
}

static Bitboard knight_deltas(Square sq) {
    static const int df[8] = {1,2,2,1,-1,-2,-2,-1};
    static const int dr[8] = {2,1,-1,-2,-2,-1,1,2};
    Bitboard attacks = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (int i = 0; i < 8; i++) {
        int nf = f + df[i], nr = r + dr[i];
        if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) set_bit(attacks, make_square(nf, nr));
    }
    return attacks;
}

static Bitboard king_deltas(Square sq) {
    Bitboard attacks = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (int df = -1; df <= 1; df++)
        for (int dr = -1; dr <= 1; dr++) {
            if (df == 0 && dr == 0) continue;
            int nf = f + df, nr = r + dr;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) set_bit(attacks, make_square(nf, nr));
        }
    return attacks;
}

// ---------- sliding piece masks / on-the-fly attacks (for magic generation) ----------

static Bitboard bishop_mask(Square sq) {
    Bitboard mask = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (int df=1,dr=1; f+df<=6 && r+dr<=6; df++,dr++) set_bit(mask, make_square(f+df,r+dr));
    for (int df=1,dr=1; f-df>=1 && r+dr<=6; df++,dr++) set_bit(mask, make_square(f-df,r+dr));
    for (int df=1,dr=1; f+df<=6 && r-dr>=1; df++,dr++) set_bit(mask, make_square(f+df,r-dr));
    for (int df=1,dr=1; f-df>=1 && r-dr>=1; df++,dr++) set_bit(mask, make_square(f-df,r-dr));
    return mask;
}

static Bitboard rook_mask(Square sq) {
    Bitboard mask = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (int nf=f+1; nf<=6; nf++) set_bit(mask, make_square(nf, r));
    for (int nf=f-1; nf>=1; nf--) set_bit(mask, make_square(nf, r));
    for (int nr=r+1; nr<=6; nr++) set_bit(mask, make_square(f, nr));
    for (int nr=r-1; nr>=1; nr--) set_bit(mask, make_square(f, nr));
    return mask;
}

static Bitboard bishop_attacks_otf(Square sq, Bitboard block) {
    Bitboard attacks = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (int df=1,dr=1; f+df<=7 && r+dr<=7; df++,dr++) { Square s=make_square(f+df,r+dr); set_bit(attacks,s); if (get_bit(block,s)) break; }
    for (int df=1,dr=1; f-df>=0 && r+dr<=7; df++,dr++) { Square s=make_square(f-df,r+dr); set_bit(attacks,s); if (get_bit(block,s)) break; }
    for (int df=1,dr=1; f+df<=7 && r-dr>=0; df++,dr++) { Square s=make_square(f+df,r-dr); set_bit(attacks,s); if (get_bit(block,s)) break; }
    for (int df=1,dr=1; f-df>=0 && r-dr>=0; df++,dr++) { Square s=make_square(f-df,r-dr); set_bit(attacks,s); if (get_bit(block,s)) break; }
    return attacks;
}

static Bitboard rook_attacks_otf(Square sq, Bitboard block) {
    Bitboard attacks = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (int nf=f+1; nf<=7; nf++) { Square s=make_square(nf,r); set_bit(attacks,s); if (get_bit(block,s)) break; }
    for (int nf=f-1; nf>=0; nf--) { Square s=make_square(nf,r); set_bit(attacks,s); if (get_bit(block,s)) break; }
    for (int nr=r+1; nr<=7; nr++) { Square s=make_square(f,nr); set_bit(attacks,s); if (get_bit(block,s)) break; }
    for (int nr=r-1; nr>=0; nr--) { Square s=make_square(f,nr); set_bit(attacks,s); if (get_bit(block,s)) break; }
    return attacks;
}

static Bitboard set_occupancy(int index, int bits_in_mask, Bitboard mask) {
    Bitboard occupancy = 0;
    for (int i = 0; i < bits_in_mask; i++) {
        Square sq = pop_lsb(mask);
        if (index & (1 << i)) set_bit(occupancy, sq);
    }
    return occupancy;
}



#if defined(__BMI2__)
static inline int attack_index(Bitboard occ, Bitboard mask, Bitboard, int) {
    return (int)_pext_u64(occ, mask);
}
#else
static inline int attack_index(Bitboard occ, Bitboard, Bitboard magic, int bits) {
    return (int)((occ * magic) >> (64 - bits));
}
#endif

static const Bitboard BISHOP_MAGIC_CONST[64] = {
0x8102000c4044043,
0x308880124102000,
0x8030028204d20820,
0x24041090003000,
0x2210882000320800,
0x1010841008000,
0x4800491011100804,
0x820802020218,
0x40601806480065,
0x1024104122008600,
0x82084180081080e,
0x8882050400000,
0x440145040085040,
0x18008804404880,
0x9008088084000,
0x4153008401011000,
0x260044018850b29,
0x10c40881090400,
0x2010100800404008,
0x44c4000804161080,
0x8804001880a00022,
0x20300080c020a01,
0x804052086008,
0x2001024822802,
0x30281024083084,
0x4002108260042088,
0x900502002202,
0x1120048128008010,
0x204840000822000,
0x800802000c648406,
0x115040402308400,
0x1010200404804,
0x2090410405082,
0x40ca100204112200,
0x100802400408406,
0x100900820840400,
0x2004004010140100,
0x96095300421000,
0x20180084a0010801,
0x1050602010440,
0x808441044300801,
0x400821000084e,
0x8006020202041100,
0x22a00a018008100,
0x2303400109003200,
0x5048012808200601,
0x10a0840400803041,
0x1010109180200,
0x80820802400000,
0x300440208421900,
0x82204404048202,
0x1381051840c2044,
0x200080e004240920,
0x80a00530008000,
0x20200142048820,
0x10042400c4110002,
0xa004410808212400,
0x2004208010800,
0x804e01208c3000,
0x200410080420200,
0x1050248112400,
0x801c1060015100,
0x41c100302440400,
0x404010921a014110
};
static const Bitboard ROOK_MAGIC_CONST[64] = {
0x6080008020104000,
0x40200010004002,
0x10020004011000a,
0x500050020100008,
0x1500100500420800,
0x300110002080400,
0x480220000800100,
0x2000240820d0824,
0x800080204004,
0x200400040201002,
0x8808010002000,
0x200801000800804,
0x10808008008400,
0x102000200041108,
0x801000401000200,
0x2200008104284a,
0x8580208008904000,
0x200464000201008,
0x828010002000,
0x9004420008120022,
0x300808004000800,
0x902008002800400,
0x850040008108201,
0x220014488104,
0x40800040c0012010,
0x2a40008480200042,
0x6001001100402000,
0x10008080080012,
0x4000080080800400,
0x102020080040080,
0x2821400900138,
0x300f204200008104,
0x180004000402008,
0x8800200080804004,
0x80801000802004,
0x4080100a02004020,
0x80080800402,
0x10800400800200,
0x20804001001,
0x880010a902000044,
0x1a00400080208000,
0x400830040030020,
0x2058100020008080,
0x2300090010010020,
0x14008040080800,
0x22000408020010,
0x121000200010004,
0xa400804400820001,
0xa24080150100,
0x4200804000200080,
0x1002001020804200,
0xc204100302a00900,
0x2108048001c0180,
0x86800400020080,
0x5038011002080400,
0x51058841200,
0x201028000106941,
0x1004020801202,
0x4001100c1200039,
0x204040900201001,
0x800600200834508a,
0x1051000400080203,
0x2000804016082,
0x80840080410022
};

void init_attacks() {
    for (Square sq = 0; sq < 64; sq++) {
        PAWN_ATTACKS[WHITE][sq] = pawn_deltas(WHITE, sq);
        PAWN_ATTACKS[BLACK][sq] = pawn_deltas(BLACK, sq);
        KNIGHT_ATTACKS[sq] = knight_deltas(sq);
        KING_ATTACKS[sq]   = king_deltas(sq);
        BISHOP_MASKS[sq] = bishop_mask(sq);
        ROOK_MASKS[sq]   = rook_mask(sq);
        BISHOP_BITS[sq]  = popcount(BISHOP_MASKS[sq]);
        ROOK_BITS[sq]    = popcount(ROOK_MASKS[sq]);
        BISHOP_MAGICS[sq] = BISHOP_MAGIC_CONST[sq];
        ROOK_MAGICS[sq]   = ROOK_MAGIC_CONST[sq];
    }
    for (Square sq = 0; sq < 64; sq++) {
        int n = 1 << BISHOP_BITS[sq];
        for (int i = 0; i < n; i++) {
            Bitboard occ = set_occupancy(i, BISHOP_BITS[sq], BISHOP_MASKS[sq]);
            int idx = attack_index(occ, BISHOP_MASKS[sq], BISHOP_MAGICS[sq], BISHOP_BITS[sq]);
            BISHOP_TABLE[sq][idx] = bishop_attacks_otf(sq, occ);
        }
    }
    for (Square sq = 0; sq < 64; sq++) {
        int n = 1 << ROOK_BITS[sq];
        for (int i = 0; i < n; i++) {
            Bitboard occ = set_occupancy(i, ROOK_BITS[sq], ROOK_MASKS[sq]);
            int idx = attack_index(occ, ROOK_MASKS[sq], ROOK_MAGICS[sq], ROOK_BITS[sq]);
            ROOK_TABLE[sq][idx] = rook_attacks_otf(sq, occ);
        }
    }
}

Bitboard get_bishop_attacks(Square sq, Bitboard occupancy) {
    occupancy &= BISHOP_MASKS[sq];
    int idx = attack_index(occupancy, BISHOP_MASKS[sq], BISHOP_MAGICS[sq], BISHOP_BITS[sq]);
    return BISHOP_TABLE[sq][idx];
}

Bitboard get_rook_attacks(Square sq, Bitboard occupancy) {
    occupancy &= ROOK_MASKS[sq];
    int idx = attack_index(occupancy, ROOK_MASKS[sq], ROOK_MAGICS[sq], ROOK_BITS[sq]);
    return ROOK_TABLE[sq][idx];
}

} // namespace halcyon
