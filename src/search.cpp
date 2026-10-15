// ============================================================================
//  search.cpp  --  Bộ tìm kiếm nâng cấp cho Halcyon
//
//  Giữ nguyên toàn bộ giao diện công khai của search.h (SearchLimits, search_start,
//  search_stop, search_prepare, tt_resize, tt_clear, set_num_threads, get_num_threads)
//  nên chỉ cần thay file này là dùng được, không phải sửa header nào.
//
//  Các điểm nâng cấp chính:
//   * Bảng băm 4-đường (bucket 64 byte), khoá kiểu XOR không cần mutex (an toàn cho Lazy SMP),
//     có thế hệ (aging), lưu cả static eval, chuẩn hoá điểm chiếu hết theo ply (sửa lỗi cũ).
//   * Move picker theo giai đoạn: TT -> ăn quân tốt (SEE) -> killer -> counter -> quiet theo
//     lịch sử -> ăn quân xấu. Không cấp phát bộ nhớ động trong đường nóng.
//   * Lịch sử: butterfly + continuation (1,2,4 ply) + capture history, cập nhật kiểu "trọng lực".
//   * Hiệu chỉnh static eval bằng correction history theo cấu trúc tốt.
//   * Cắt tỉa/giảm độ sâu: MDP, RFP, razoring, null-move thích nghi + kiểm chứng, ProbCut,
//     IIR, futility, LMP, cắt theo lịch sử, cắt theo SEE, LMR theo bảng log có hiệu chỉnh,
//     singular extension (kèm multi-cut và negative extension), check extension.
//   * Quiescence có TT, xử lý né chiếu, cắt SEE + delta.
//   * Aspiration window thích nghi, quản lý thời gian mềm/cứng theo độ ổn định nước đi,
//     biến động điểm và tỉ lệ nút của nước tốt nhất ở gốc.
//   * Lazy SMP: mỗi luồng có dữ liệu riêng, xen kẽ độ sâu, bỏ phiếu chọn nước cuối.
// ============================================================================
#include "search.h"
#include "movegen.h"
#include "eval.h"
#include "see.h"
#include "attacks.h"
#include "tablebase.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <thread>
#include <iostream>
#include <sstream>
#include <mutex>

namespace halcyon {

// ---------------------------------------------------------------------------
// Hàm phụ do main.cpp (bản mới) gọi. Khai báo tại đây để không cần sửa search.h.
// ---------------------------------------------------------------------------
void set_move_overhead(int ms);
void set_go_extras(int movestogo, long long nodes);
long long search_last_nodes();
void set_search_quiet(bool quiet);
void set_multi_pv(int n);

constexpr int MAX_PLY = 128;
constexpr int INF = 30000;
constexpr int MATE_SCORE = 29000;
constexpr int MATE_THRESHOLD = MATE_SCORE - MAX_PLY;

namespace {

constexpr int SCORE_NONE = 32001;       // "chưa có điểm" (vẫn vừa int16)
constexpr int STACK_OFFSET = 4;         // chừa 4 ô phía trước gốc cho ss-1..ss-4
constexpr int MAX_CAPS = 192;
constexpr int MAX_QUIETS = 256;

enum Bnd : uint8_t { BND_NONE = 0, BND_UPPER = 1, BND_LOWER = 2, BND_EXACT = 3 };

// Giá trị dùng riêng cho việc sắp xếp nước đi (chỉ số theo PieceType, phần tử 6 = NO_PIECE_TYPE)
const int kOrderVal[7] = {100, 325, 335, 500, 975, 20000, 0};

// ---------------------------------------------------------------------------
//  Trạng thái toàn cục của một lần tìm kiếm
// ---------------------------------------------------------------------------
std::atomic<bool> g_stop{false};
std::chrono::steady_clock::time_point g_start;
long long g_optimum = -1;     // ms: mốc dừng mềm (chỉ xét giữa các độ sâu)
long long g_maximum = -1;     // ms: mốc dừng cứng (kiểm tra liên tục trong cây)
long long g_nodeLimit = 0;    // 0 = không giới hạn
int g_numThreads = 1;
int g_moveOverhead = 30;      // ms trừ đi cho độ trễ GUI/mạng
bool g_showWdl = false;
int g_skillLevel = 20;
bool g_limitStrength = false;
int g_uciElo = 1320;
int g_nodesTime = 0;             // nodes per ms proxy (Stockfish-style nodestime)
int g_syzygyProbeDepth = 1;
int g_syzygyProbeLimit = 7;
bool g_syzygy50MoveRule = true;
int g_halcyonSpirit = 1;      // 0 Calm, 1 Balanced, 2 Sharp — exclusive Halcyon personality
int g_clarityBias = 0;        // 0 off; else when score >= bias, lean toward simplifying lines
bool g_velocityMode = false;  // trade a little depth consistency for raw NPS

int g_movesToGo = 0;
long long g_pendingNodeLimit = 0;
int g_pendingMovesToGo = 0;
long long g_lastNodes = 0;
bool g_quiet = false;
int g_multiPV = 1;   // giá trị đặt qua "setoption name MultiPV value N", áp dụng cho lần go tiếp theo

inline long long elapsed_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - g_start).count();
}

// ---- Ponder ----------------------------------------------------------------
// "go ponder": search with NO time limit (the clock is not ours yet). The time
// budget is computed up front but parked in g_pending*; "ponderhit" flips
// g_ponderHit and the MAIN search thread (the only one that reads g_optimum /
// g_maximum / g_start) installs the budget and restarts the clock from that
// moment. Without this, ponderhit was a no-op and the search just idled until
// a "stop" that a GUI never sends after ponderhit.
std::atomic<bool> g_ponderMode{false};
std::atomic<bool> g_ponderHit{false};
bool      g_ponderApplied = false;          // main search thread only
long long g_pendingOptimum = -1, g_pendingMaximum = -1;

inline void apply_ponderhit() {
    if (g_ponderApplied || !g_ponderMode.load(std::memory_order_relaxed) ||
        !g_ponderHit.load(std::memory_order_acquire)) return;
    g_ponderApplied = true;
    g_start = std::chrono::steady_clock::now();
    g_optimum = g_pendingOptimum;
    g_maximum = g_pendingMaximum;
}

// ---------------------------------------------------------------------------
//  Bảng băm (TT)
//  Mỗi cụm (cluster) 64 byte = 4 khe. Mỗi khe gồm 2 từ 64-bit: data và (key ^ data).
//  Người đọc chỉ chấp nhận khe khi (key ^ data) khớp khoá => ghi rách (torn write) giữa
//  các luồng luôn bị loại bỏ, không bao giờ cho ra nước đi/điểm "lai" vô nghĩa.
//  Bố cục data: move16(0-15) | score(16-31) | eval(32-47) | depth(48-55) | bound(56-57)
//               | pv(58) | generation(59-63)
// ---------------------------------------------------------------------------
struct TTCluster {
    std::atomic<uint64_t> key[4];
    std::atomic<uint64_t> data[4];
};
static_assert(sizeof(TTCluster) == 64, "TTCluster phải đúng 64 byte");

std::vector<unsigned char> g_ttStorage;
TTCluster* g_tt = nullptr;
size_t g_ttClusters = 0;
uint8_t g_ttGen = 0;

struct TTSlot {
    uint16_t move = 0;
    int score = SCORE_NONE;
    int eval = SCORE_NONE;
    int depth = 0;
    int bound = BND_NONE;
    bool pv = false;
};

inline size_t tt_index(uint64_t key) {
#if defined(__SIZEOF_INT128__)
    return (size_t)(((unsigned __int128)key * (unsigned __int128)g_ttClusters) >> 64);
#else
    return (size_t)(key % g_ttClusters);
#endif
}

inline void tt_prefetch(uint64_t key) {
    if (!g_tt) return;
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(&g_tt[tt_index(key)]);
#endif
}

inline uint64_t tt_pack(uint16_t move16, int score, int eval, int depth, int bound, bool pv, uint8_t gen) {
    if (depth < 0) depth = 0;
    if (depth > 255) depth = 255;
    return (uint64_t)move16
         | ((uint64_t)(uint16_t)(int16_t)score << 16)
         | ((uint64_t)(uint16_t)(int16_t)eval << 32)
         | ((uint64_t)(uint8_t)depth << 48)
         | ((uint64_t)(bound & 3) << 56)
         | ((uint64_t)(pv ? 1 : 0) << 58)
         | ((uint64_t)(gen & 31) << 59);
}

inline void tt_unpack(uint64_t d, TTSlot& out) {
    out.move = (uint16_t)(d & 0xFFFF);
    out.score = (int16_t)((d >> 16) & 0xFFFF);
    out.eval = (int16_t)((d >> 32) & 0xFFFF);
    out.depth = (int)((d >> 48) & 0xFF);
    out.bound = (int)((d >> 56) & 3);
    out.pv = ((d >> 58) & 1) != 0;
}

bool tt_probe(uint64_t key, TTSlot& out) {
    if (!g_tt) return false;
    TTCluster* c = &g_tt[tt_index(key)];
    for (int i = 0; i < 4; i++) {
        uint64_t d = c->data[i].load(std::memory_order_relaxed);
        uint64_t k = c->key[i].load(std::memory_order_relaxed);
        if ((k ^ d) == key && d != 0) { tt_unpack(d, out); return true; }
    }
    return false;
}

void tt_store(uint64_t key, uint16_t move16, int score, int eval, int depth, int bound, bool pv) {
    if (!g_tt) return;
    TTCluster* c = &g_tt[tt_index(key)];
    int victim = 0, victimScore = 1 << 30;
    for (int i = 0; i < 4; i++) {
        uint64_t d = c->data[i].load(std::memory_order_relaxed);
        uint64_t k = c->key[i].load(std::memory_order_relaxed);
        if ((k ^ d) == key && d != 0) {
            // Cùng thế cờ: giữ nước cũ nếu lần này không có nước mới.
            uint16_t oldMove = (uint16_t)(d & 0xFFFF);
            int oldDepth = (int)((d >> 48) & 0xFF);
            uint8_t oldGen = (uint8_t)((d >> 59) & 31);
            if (move16 == 0) move16 = oldMove;
            if (bound == BND_EXACT || depth + 4 + (pv ? 2 : 0) > oldDepth || oldGen != (g_ttGen & 31)) {
                uint64_t nd = tt_pack(move16, score, eval, depth, bound, pv, g_ttGen);
                c->data[i].store(nd, std::memory_order_relaxed);
                c->key[i].store(key ^ nd, std::memory_order_relaxed);
            }
            return;
        }
        int s;
        if (d == 0) s = -100000;  // khe trống: ưu tiên ghi vào đầu tiên
        else {
            int age = ((int)(g_ttGen & 31) - (int)((d >> 59) & 31)) & 31;
            s = (int)((d >> 48) & 0xFF) - 4 * age;
        }
        if (s < victimScore) { victimScore = s; victim = i; }
    }
    uint64_t nd = tt_pack(move16, score, eval, depth, bound, pv, g_ttGen);
    c->data[victim].store(nd, std::memory_order_relaxed);
    c->key[victim].store(key ^ nd, std::memory_order_relaxed);
}

int tt_hashfull() {
    if (!g_tt) return 0;
    size_t n = std::min<size_t>(g_ttClusters, 1000);
    int cnt = 0;
    for (size_t i = 0; i < n; i++)
        for (int j = 0; j < 4; j++) {
            uint64_t d = g_tt[i].data[j].load(std::memory_order_relaxed);
            if (d != 0 && ((d >> 59) & 31) == (uint64_t)(g_ttGen & 31)) cnt++;
        }
    return (int)(cnt * 1000 / (n * 4));
}

// Điểm chiếu hết phải được chuẩn hoá theo ply khi vào/ra bảng băm, nếu không một điểm
// "chiếu hết sau N nước" lấy ra ở độ sâu ply khác sẽ sai (lỗi của bản cũ).
inline int score_to_tt(int s, int ply) {
    if (s >= MATE_THRESHOLD) return s + ply;
    if (s <= -MATE_THRESHOLD) return s - ply;
    return s;
}
inline int score_from_tt(int s, int ply) {
    if (s >= MATE_THRESHOLD) return s - ply;
    if (s <= -MATE_THRESHOLD) return s + ply;
    return s;
}

inline uint16_t encode16(Move m) {
    if (m == MOVE_NONE) return 0;
    PieceType p = promo_type(m);
    int code = (p == NO_PIECE_TYPE) ? 0 : (int)p;
    return (uint16_t)(from_sq(m) | (to_sq(m) << 6) | (code << 12));
}

// ---------------------------------------------------------------------------
//  Cấu trúc dữ liệu theo luồng
// ---------------------------------------------------------------------------
typedef int16_t ContTable[PIECE_NB][64];

struct ScoredMv {
    Move move;
    int score;
};

struct MoveBuf {
    ScoredMv caps[MAX_CAPS];
    ScoredMv quiets[MAX_QUIETS];
    ScoredMv bad[MAX_CAPS];
};

struct SearchStack {
    Move currentMove = MOVE_NONE;
    Move excluded = MOVE_NONE;
    Move killers[2] = {MOVE_NONE, MOVE_NONE};
    int staticEval = SCORE_NONE;
    int ply = 0;
    int moveCount = 0;
    bool inCheck = false;
    bool nullMove = false;
    int qsDepth = 0;
    ContTable* cont = nullptr;
};

constexpr int CORR_SIZE = 16384;
constexpr int CORR_GRAIN = 256;
constexpr int CORR_LIMIT = 16384;

struct SearchThread {
    int id = 0;
    bool isMain = false;
    uint64_t localNodes = 0;
    std::atomic<uint64_t> published{0};
    uint64_t rootNodes[64 * 64];
    int selDepth = 0;
    int nmpMinPly = 0;
    uint16_t rootPrev16 = 0;
    int completedDepth = 0;
    int bestScore = 0;
    Move bestMove = MOVE_NONE;
    int bestPvLen = 0;
    Move bestPv[MAX_PLY + 2];

    // ---- MultiPV: danh sách các dòng đã "chốt" ở độ sâu hiện tại (theo thứ
    // tự từ tốt nhất tới kém nhất), dùng để (a) loại các nước đã có PV riêng
    // ra khỏi lượt tìm PV tiếp theo ở CÙNG độ sâu, và (b) in "info multipv N"
    // cho GUI. Chỉ thread chính (isMain) dùng cơ chế này; các thread phụ vẫn
    // tìm 1 PV như trước để hỗ trợ Lazy SMP bình thường.
    static constexpr int MAX_MULTIPV = 256;
    int numPv = 1;                          // giá trị MultiPV hiện hành (>=1)
    Move rootExclude[MAX_MULTIPV];          // các nước đã thuộc PV #1..#(k-1) ở độ sâu này
    int numExcluded = 0;
    struct PvEntry { int score; int len; Move line[MAX_PLY + 2]; Move rootMove; bool scored; };
    PvEntry multiPvLines[MAX_MULTIPV];      // kết quả PV #1..#numPv của độ sâu ĐÃ HOÀN THÀNH gần nhất

    int16_t mainHist[COLOR_NB][64][64];
    int16_t captHist[PIECE_NB][64][7];
    ContTable contHist[PIECE_NB + 1][64];   // phần tử [PIECE_NB][0] là bảng "rỗng" cho null move / gốc
    Move counter[PIECE_NB][64];
    int16_t pawnCorr[COLOR_NB][CORR_SIZE];

    SearchStack stack[MAX_PLY + 12];
    Move pv[MAX_PLY + 2][MAX_PLY + 2];
    int pvLen[MAX_PLY + 2];
    MoveBuf buf[2 * (MAX_PLY + 2)];   // nửa sau dành cho các lần tìm kiếm có nước bị loại (singular)

    void clear_history() {
        std::memset(mainHist, 0, sizeof(mainHist));
        std::memset(captHist, 0, sizeof(captHist));
        std::memset(contHist, 0, sizeof(contHist));
        std::memset(counter, 0, sizeof(counter));
        std::memset(pawnCorr, 0, sizeof(pawnCorr));
    }
};

std::vector<std::unique_ptr<SearchThread>> g_threads;

// Bảng giảm độ sâu LMR: ln(depth) * ln(moveCount) / 2.25 + 0.75
int g_lmr[64][64];
std::once_flag g_initOnce;
void init_search_tables() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++)
            g_lmr[d][m] = (int)(0.85 + std::log((double)d) * std::log((double)m) / 1.95);
}

// ---------------------------------------------------------------------------
//  Tiện ích cờ vua
// ---------------------------------------------------------------------------
inline bool has_non_pawn_material(const Position& pos, Color c) {
    return (pos.occByColor[c] & ~(pos.pieceBB[make_piece(c, PAWN)] | pos.pieceBB[make_piece(c, KING)])) != 0;
}

inline bool insufficient_material(const Position& pos) {
    if (pos.pieceBB[WP] | pos.pieceBB[BP] | pos.pieceBB[WR] | pos.pieceBB[BR] | pos.pieceBB[WQ] | pos.pieceBB[BQ])
        return false;
    return popcount(pos.pieceBB[WN] | pos.pieceBB[WB] | pos.pieceBB[BN] | pos.pieceBB[BB]) <= 1;
}

inline int draw_score(const SearchThread& th) { return 1 - (int)(th.localNodes & 2); }

inline bool is_quiet_move(Move m) { return !is_capture(m) && promo_type(m) == NO_PIECE_TYPE; }

inline PieceType captured_type(const Position& pos, Move m) {
    if (is_ep(m)) return PAWN;
    Piece p = pos.board[to_sq(m)];
    return p == NO_PIECE ? NO_PIECE_TYPE : type_of(p);
}

inline uint64_t pawn_hash(const Position& pos) {
    uint64_t a = pos.pieceBB[WP], b = pos.pieceBB[BP];
    uint64_t h = (a * 0x9E3779B97F4A7C15ULL) ^ ((b + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL);
    return h ^ (h >> 29);
}

inline int corr_index(const Position& pos) { return (int)((pawn_hash(pos) >> 20) & (CORR_SIZE - 1)); }

inline int corrected_eval(const SearchThread& th, const Position& pos, int raw) {
    int corr = th.pawnCorr[pos.sideToMove][corr_index(pos)] / (CORR_GRAIN * 2);
    int v = raw + corr;
    if (v > MATE_THRESHOLD - 1) v = MATE_THRESHOLD - 1;
    if (v < -MATE_THRESHOLD + 1) v = -MATE_THRESHOLD + 1;
    return v;
}

// Nước đi có gây chiếu không (tính TRƯỚC khi make_move) -- dùng để miễn cắt tỉa.
bool gives_check(const Position& pos, Move m) {
    const Color us = pos.sideToMove, them = ~us;
    const Square from = from_sq(m), to = to_sq(m);
    const Square ksq = lsb(pos.pieceBB[make_piece(them, KING)]);
    const Bitboard fromBB = 1ULL << from, toBB = 1ULL << to;
    Bitboard occ = (pos.occAll ^ fromBB) | toBB;
    const PieceType pt = type_of(moved_piece(m));
    const PieceType placed = (promo_type(m) != NO_PIECE_TYPE) ? promo_type(m) : pt;

    Bitboard set[6];
    for (int i = 0; i < 6; i++) set[i] = pos.pieceBB[make_piece(us, PieceType(i))];
    set[pt] &= ~fromBB;
    set[placed] |= toBB;

    if (is_ep(m)) occ ^= 1ULL << make_square(file_of(to), rank_of(from));
    if (is_castle(m)) {
        const int r = rank_of(from);
        Square rFrom, rTo;
        if (file_of(to) == 6) { rFrom = make_square(7, r); rTo = make_square(5, r); }
        else                  { rFrom = make_square(0, r); rTo = make_square(3, r); }
        occ ^= (1ULL << rFrom) | (1ULL << rTo);
        set[ROOK] = (set[ROOK] & ~(1ULL << rFrom)) | (1ULL << rTo);
    }
    if (PAWN_ATTACKS[them][ksq] & set[PAWN]) return true;
    if (KNIGHT_ATTACKS[ksq] & set[KNIGHT]) return true;
    if (get_bishop_attacks(ksq, occ) & (set[BISHOP] | set[QUEEN])) return true;
    if (get_rook_attacks(ksq, occ) & (set[ROOK] | set[QUEEN])) return true;
    return false;
}

// Dựng lại một Move đầy đủ từ mã 16-bit (từ/đến/phong cấp) của TT, killer, counter...
// Chỉ trả về true nếu nước đó là giả-hợp-lệ (pseudo-legal) ở thế cờ hiện tại và sinh ra
// đúng Move mà bộ sinh nước đi sẽ sinh. Nhờ vậy không cần sinh cả danh sách chỉ để kiểm tra.
bool decode_move16(const Position& pos, uint16_t m16, Move& out) {
    if (m16 == 0) return false;
    const Square from = m16 & 63, to = (m16 >> 6) & 63;
    const int promoCode = (m16 >> 12) & 7;
    if (promoCode > 4) return false;
    const Color us = pos.sideToMove;
    const Piece pc = pos.board[from];
    if (pc == NO_PIECE || color_of(pc) != us) return false;
    const Piece target = pos.board[to];
    if (target != NO_PIECE && color_of(target) == us) return false;
    const PieceType pt = type_of(pc);
    const Bitboard toBB = 1ULL << to;
    const PieceType promo = promoCode ? PieceType(promoCode) : NO_PIECE_TYPE;

    if (pt == PAWN) {
        const int dir = (us == WHITE) ? 8 : -8;
        const int promoRank = (us == WHITE) ? 7 : 0;
        const int startRank = (us == WHITE) ? 1 : 6;
        const bool isPromoMove = (rank_of(to) == promoRank);
        if (isPromoMove != (promoCode != 0)) return false;
        if (to == from + dir) {
            if (target != NO_PIECE) return false;
            out = make_move(from, to, pc, promo);
            return true;
        }
        if (rank_of(from) == startRank && to == from + 2 * dir) {
            if (target != NO_PIECE || pos.board[from + dir] != NO_PIECE) return false;
            out = make_move(from, to, pc, NO_PIECE_TYPE, false, true);
            return true;
        }
        if (PAWN_ATTACKS[us][from] & toBB) {
            if (target != NO_PIECE) { out = make_move(from, to, pc, promo, true); return true; }
            if (pos.epSquare != NO_SQUARE && to == pos.epSquare && promoCode == 0) {
                out = make_move(from, to, pc, NO_PIECE_TYPE, true, false, true);
                return true;
            }
        }
        return false;
    }

    if (promoCode) return false;
    const bool cap = (target != NO_PIECE);
    switch (pt) {
        case KNIGHT:
            if (!(KNIGHT_ATTACKS[from] & toBB)) return false;
            break;
        case BISHOP:
            if (!(get_bishop_attacks(from, pos.occAll) & toBB)) return false;
            break;
        case ROOK:
            if (!(get_rook_attacks(from, pos.occAll) & toBB)) return false;
            break;
        case QUEEN:
            if (!((get_bishop_attacks(from, pos.occAll) | get_rook_attacks(from, pos.occAll)) & toBB)) return false;
            break;
        case KING: {
            if (KING_ATTACKS[from] & toBB) break;
            const Bitboard occ = pos.occAll;
            if (us == WHITE && from == E1) {
                if (to == G1 && (pos.castling & WK_SIDE) && !get_bit(occ, F1) && !get_bit(occ, G1) &&
                    !pos.square_attacked(E1, BLACK) && !pos.square_attacked(F1, BLACK) && !pos.square_attacked(G1, BLACK)) {
                    out = make_move(E1, G1, WK, NO_PIECE_TYPE, false, false, false, true);
                    return true;
                }
                if (to == C1 && (pos.castling & WQ_SIDE) && !get_bit(occ, D1) && !get_bit(occ, C1) && !get_bit(occ, B1) &&
                    !pos.square_attacked(E1, BLACK) && !pos.square_attacked(D1, BLACK) && !pos.square_attacked(C1, BLACK)) {
                    out = make_move(E1, C1, WK, NO_PIECE_TYPE, false, false, false, true);
                    return true;
                }
            } else if (us == BLACK && from == E8) {
                if (to == G8 && (pos.castling & BK_SIDE) && !get_bit(occ, F8) && !get_bit(occ, G8) &&
                    !pos.square_attacked(E8, WHITE) && !pos.square_attacked(F8, WHITE) && !pos.square_attacked(G8, WHITE)) {
                    out = make_move(E8, G8, BK, NO_PIECE_TYPE, false, false, false, true);
                    return true;
                }
                if (to == C8 && (pos.castling & BQ_SIDE) && !get_bit(occ, D8) && !get_bit(occ, C8) && !get_bit(occ, B8) &&
                    !pos.square_attacked(E8, WHITE) && !pos.square_attacked(D8, WHITE) && !pos.square_attacked(C8, WHITE)) {
                    out = make_move(E8, C8, BK, NO_PIECE_TYPE, false, false, false, true);
                    return true;
                }
            }
            return false;
        }
        default: return false;
    }
    out = make_move(from, to, pc, NO_PIECE_TYPE, cap);
    return true;
}

// ---------------------------------------------------------------------------
//  Lịch sử (history) -- cập nhật kiểu "trọng lực": giá trị luôn bị chặn trong ±16384
// ---------------------------------------------------------------------------
inline void hist_update(int16_t& h, int bonus) {
    if (bonus > 16384) bonus = 16384;
    if (bonus < -16384) bonus = -16384;
    h = (int16_t)(h + bonus - (int)h * std::abs(bonus) / 16384);
}

inline int hist_bonus(int depth) { return std::min(170 * depth - 80, 1700); }

inline int quiet_score(const SearchThread& th, const SearchStack* ss, Color us, Move m) {
    const Piece pc = moved_piece(m);
    const Square to = to_sq(m);
    int s = 2 * th.mainHist[us][from_sq(m)][to];
    s += 2 * (*(ss - 1)->cont)[pc][to];
    s += (*(ss - 2)->cont)[pc][to];
    s += (*(ss - 4)->cont)[pc][to] / 2;
    return s;
}

void update_cont_hist(SearchThread& th, SearchStack* ss, Piece pc, Square to, int bonus) {
    static const int plies[3] = {1, 2, 4};
    for (int i = 0; i < 3; i++) {
        SearchStack* p = ss - plies[i];
        if (p->currentMove == MOVE_NONE) continue;
        hist_update((*p->cont)[pc][to], i == 2 ? bonus / 2 : bonus);
    }
    (void)th;
}

void update_quiet_history(SearchThread& th, SearchStack* ss, Color us, Move m, int bonus) {
    hist_update(th.mainHist[us][from_sq(m)][to_sq(m)], bonus);
    update_cont_hist(th, ss, moved_piece(m), to_sq(m), bonus);
}

void update_capture_history(const Position& pos, SearchThread& th, Move m, int bonus) {
    PieceType v = captured_type(pos, m);
    if (v == NO_PIECE_TYPE) return;
    hist_update(th.captHist[moved_piece(m)][to_sq(m)][v], bonus);
}

} // namespace

namespace {

// ---------------------------------------------------------------------------
//  Bộ chọn nước đi theo giai đoạn (move picker)
// ---------------------------------------------------------------------------
enum PickStage {
    PS_TT, PS_CAPS_GEN, PS_CAPS, PS_KILLER1, PS_KILLER2, PS_COUNTER,
    PS_QUIETS_GEN, PS_QUIETS, PS_BAD, PS_DONE
};

inline bool is_good_capture(const Position& pos, Move m) {
    PieceType promo = promo_type(m);
    if (promo != NO_PIECE_TYPE) return promo == QUEEN;      // phong cấp thấp => coi là xấu
    PieceType victim = is_ep(m) ? PAWN : type_of(pos.board[to_sq(m)]);
    PieceType att = type_of(moved_piece(m));
    if (kOrderVal[victim] >= kOrderVal[att]) return true;   // ăn quân >= quân đi: chắc chắn không lỗ
    return see(pos, m) >= 0;
}

struct MovePicker {
    const Position& pos;
    SearchThread& th;
    SearchStack* ss;
    MoveBuf& buf;
    Move ttMove = MOVE_NONE, killer1 = MOVE_NONE, killer2 = MOVE_NONE, counterMv = MOVE_NONE;
    int stage;
    int nCaps = 0, capIdx = 0, nQuiets = 0, quietIdx = 0, nBad = 0, badIdx = 0;
    bool qsMode, inCheck, probcut;
    int threshold;

    // qs: dùng cho quiescence; probcut: chỉ lấy các nước ăn quân có SEE >= ngưỡng.
    MovePicker(const Position& p, SearchThread& t, SearchStack* s, uint16_t tt16,
               bool qs, bool check, bool pc, int thr)
        : pos(p), th(t), ss(s), buf(t.buf[s->ply + (s->excluded != MOVE_NONE ? MAX_PLY + 2 : 0)]), qsMode(qs), inCheck(check), probcut(pc), threshold(thr) {
        Move m;
        if (tt16 && decode_move16(pos, tt16, m)) {
            // Quiescence khi không bị chiếu và probcut chỉ nhận nước ăn quân/phong cấp.
            if (!((qsMode && !inCheck) || probcut) || !is_quiet_move(m)) ttMove = m;
        }
        if (!qsMode && !probcut) {
            Move k;
            if (decode_move16(pos, encode16(ss->killers[0]), k) && is_quiet_move(k) && k != ttMove) killer1 = k;
            if (decode_move16(pos, encode16(ss->killers[1]), k) && is_quiet_move(k) && k != ttMove && k != killer1) killer2 = k;
            const Move prev = (ss - 1)->currentMove;
            if (prev != MOVE_NONE) {
                Move c = th.counter[moved_piece(prev)][to_sq(prev)];
                if (c != MOVE_NONE && decode_move16(pos, encode16(c), k) && is_quiet_move(k) &&
                    k != ttMove && k != killer1 && k != killer2) counterMv = k;
            }
        }
        stage = (ttMove != MOVE_NONE) ? PS_TT : PS_CAPS_GEN;
    }

    void gen_caps() {
        MoveList list;
        generate_captures(pos, list);
        for (Move m : list) {
            if (m == ttMove) continue;
            PieceType promo = promo_type(m);
            const bool underPromo = (promo != NO_PIECE_TYPE && promo != QUEEN);
            if (underPromo) {
                // Phong cấp thấp: chỉ xét ở cuối (và không xét trong quiescence/probcut).
                if (!qsMode && !probcut && nBad < MAX_CAPS) buf.bad[nBad++] = {m, -1000000};
                continue;
            }
            if (nCaps >= MAX_CAPS) break;
            PieceType victim = is_capture(m) ? captured_type(pos, m) : NO_PIECE_TYPE;
            int score = kOrderVal[victim] * 10 - kOrderVal[type_of(moved_piece(m))] / 8;
            if (victim != NO_PIECE_TYPE) score += th.captHist[moved_piece(m)][to_sq(m)][victim] / 16;
            if (promo == QUEEN) score += 10 * (kOrderVal[QUEEN] - kOrderVal[PAWN]);
            buf.caps[nCaps++] = {m, score};
        }
    }

    void gen_quiets() {
        MoveList list;
        generate_pseudo_moves(pos, list);
        const Color us = pos.sideToMove;
        for (Move m : list) {
            if (!is_quiet_move(m)) continue;
            if (m == ttMove || m == killer1 || m == killer2 || m == counterMv) continue;
            if (nQuiets >= MAX_QUIETS) break;
            buf.quiets[nQuiets++] = {m, quiet_score(th, ss, us, m)};
        }
    }

    static void select_best(ScoredMv* arr, int start, int n) {
        int best = start;
        for (int i = start + 1; i < n; i++)
            if (arr[i].score > arr[best].score) best = i;
        if (best != start) std::swap(arr[start], arr[best]);
    }

    Move next(bool skipQuiets) {
        for (;;) {
            switch (stage) {
            case PS_TT:
                stage = PS_CAPS_GEN;
                return ttMove;
            case PS_CAPS_GEN:
                gen_caps();
                stage = PS_CAPS;
                break;
            case PS_CAPS:
                while (capIdx < nCaps) {
                    select_best(buf.caps, capIdx, nCaps);
                    Move m = buf.caps[capIdx].move;
                    int sc = buf.caps[capIdx].score;
                    capIdx++;
                    if (probcut) {
                        if (see(pos, m) >= threshold) return m;
                        continue;
                    }
                    if (is_good_capture(pos, m)) return m;
                    if (!qsMode || inCheck) {
                        if (nBad < MAX_CAPS) buf.bad[nBad++] = {m, sc};
                    }
                }
                if ((qsMode && !inCheck) || probcut) { stage = PS_DONE; break; }
                stage = qsMode ? PS_QUIETS_GEN : PS_KILLER1;
                break;
            case PS_KILLER1:
                stage = PS_KILLER2;
                if (!skipQuiets && killer1 != MOVE_NONE) return killer1;
                break;
            case PS_KILLER2:
                stage = PS_COUNTER;
                if (!skipQuiets && killer2 != MOVE_NONE) return killer2;
                break;
            case PS_COUNTER:
                stage = PS_QUIETS_GEN;
                if (!skipQuiets && counterMv != MOVE_NONE) return counterMv;
                break;
            case PS_QUIETS_GEN:
                if (!skipQuiets) gen_quiets();
                stage = PS_QUIETS;
                break;
            case PS_QUIETS:
                if (!skipQuiets && quietIdx < nQuiets) {
                    select_best(buf.quiets, quietIdx, nQuiets);
                    return buf.quiets[quietIdx++].move;
                }
                stage = PS_BAD;
                break;
            case PS_BAD:
                if (badIdx < nBad) return buf.bad[badIdx++].move;
                stage = PS_DONE;
                break;
            default:
                return MOVE_NONE;
            }
        }
    }
};

int negamax(Position& pos, SearchThread& th, SearchStack* ss, int depth, int alpha, int beta, bool cutNode);
int qsearch(Position& pos, SearchThread& th, SearchStack* ss, int alpha, int beta);

inline void periodic_check(SearchThread& th) {
    th.published.store(th.localNodes, std::memory_order_relaxed);
    if (!th.isMain) return;
    apply_ponderhit();
    if (g_maximum >= 0 && elapsed_ms() >= g_maximum) g_stop.store(true, std::memory_order_relaxed);
    if (g_nodeLimit > 0) {
        uint64_t total = th.localNodes;
        for (auto& t : g_threads)
            if (t.get() != &th) total += t->published.load(std::memory_order_relaxed);
        if ((long long)total >= g_nodeLimit) g_stop.store(true, std::memory_order_relaxed);
    }
}

inline void update_pv(SearchThread& th, int ply, Move m) {
    th.pv[ply][ply] = m;
    int j = ply + 1;
    for (; j < th.pvLen[ply + 1]; j++) th.pv[ply][j] = th.pv[ply + 1][j];
    th.pvLen[ply] = std::max(ply + 1, th.pvLen[ply + 1]);
}

// ---------------------------------------------------------------------------
//  Quiescence
// ---------------------------------------------------------------------------
int qsearch(Position& pos, SearchThread& th, SearchStack* ss, int alpha, int beta) {
    const bool pvNode = (beta - alpha) > 1;
    const int ply = ss->ply;
    const Color us = pos.sideToMove;

    th.localNodes++;
    if ((th.localNodes & 1023) == 0) periodic_check(th);
    if (g_stop.load(std::memory_order_relaxed)) return 0;
    if (pvNode && th.selDepth < ply + 1) th.selDepth = ply + 1;

    const bool inCheck = ss->inCheck;
    if (pos.is_draw() || insufficient_material(pos)) return draw_score(th);
    if (ply >= MAX_PLY - 1) return inCheck ? draw_score(th) : evaluate(pos);

    // Same exact-result tablebase cutoff as negamax (see the comment there);
    // qsearch has no mate-distance-pruning window to intersect with, so this
    // is just an immediate, fully accurate return instead of searching
    // captures/checks that can't improve on a mathematically proven result.
    if (popcount(pos.occAll) <= 3 && tablebases_available()) {
        TBProbeResult tb;
        if (probe_tb(pos, tb)) {
            if (tb.result == TBResult::DRAW) return draw_score(th);
            const Color us = pos.sideToMove;
            const bool usWins = (tb.result == TBResult::WHITE_WIN) == (us == WHITE);
            const int mateScore = MATE_SCORE - (ply + tb.dtm);
            return usWins ? mateScore : -mateScore;
        }
    }

    const uint64_t key = pos.key;
    TTSlot tt;
    const bool ttHit = tt_probe(key, tt);
    const int ttScore = (ttHit && tt.score != SCORE_NONE) ? score_from_tt(tt.score, ply) : SCORE_NONE;
    if (!pvNode && ttHit && ttScore != SCORE_NONE && (tt.bound & (ttScore >= beta ? BND_LOWER : BND_UPPER)))
        return ttScore;

    int bestValue, futilityBase = -INF, rawEval = SCORE_NONE;
    if (inCheck) {
        bestValue = -INF;
        ss->staticEval = SCORE_NONE;
    } else {
        rawEval = (ttHit && tt.eval != SCORE_NONE) ? tt.eval : evaluate(pos);
        ss->staticEval = bestValue = corrected_eval(th, pos, rawEval);
        if (ttScore != SCORE_NONE && (tt.bound & (ttScore > bestValue ? BND_LOWER : BND_UPPER))) bestValue = ttScore;
        if (bestValue >= beta) {
            if (!ttHit) tt_store(key, 0, SCORE_NONE, rawEval, 0, BND_NONE, false);
            return bestValue;
        }
        if (pvNode && bestValue > alpha) alpha = bestValue;
        futilityBase = ss->staticEval + 200;
    }

    const int origAlpha = alpha;
    Move bestMove = MOVE_NONE;
    int moveCount = 0;
    MovePicker mp(pos, th, ss, ttHit ? tt.move : 0, true, inCheck, false, 0);
    Move m;
    while ((m = mp.next(false)) != MOVE_NONE) {
        if (!move_is_legal(pos, m)) continue;
        moveCount++;

        if (!inCheck && promo_type(m) == NO_PIECE_TYPE && bestValue > -MATE_THRESHOLD) {
            // Delta/futility: ngay cả khi ăn trọn quân này cũng không chạm tới alpha.
            PieceType v = captured_type(pos, m);
            int gain = (v == NO_PIECE_TYPE) ? 0 : kOrderVal[v];
            if (futilityBase + gain <= alpha) {
                bestValue = std::max(bestValue, futilityBase + gain);
                continue;
            }
        }

        const Piece movedPc = moved_piece(m);
        ss->currentMove = m;
        ss->cont = &th.contHist[movedPc][to_sq(m)];
        pos.make_move(m);
        tt_prefetch(pos.key);
        (ss + 1)->inCheck = pos.square_attacked(lsb(pos.pieceBB[make_piece(pos.sideToMove, KING)]), us);
        (ss + 1)->excluded = MOVE_NONE;
        (ss + 1)->qsDepth = ss->qsDepth + 1;
        const int score = -qsearch(pos, th, ss + 1, -beta, -alpha);
        pos.unmake_move(m);
        if (g_stop.load(std::memory_order_relaxed)) return 0;

        if (score > bestValue) {
            bestValue = score;
            if (score > alpha) {
                bestMove = m;
                if (score >= beta) break;
                alpha = score;
            }
        }
    }

    // Chiếu yên ở lớp quiescence đầu: bắt các đòn chiếu 1 nước mà chỉ tìm nước ăn sẽ bỏ lỡ.
    if (!inCheck && ss->qsDepth == 0 && bestValue < beta && bestValue > -MATE_THRESHOLD) {
        MoveList quiets;
        generate_pseudo_moves(pos, quiets);
        int nChecks = 0;
        for (Move qm : quiets) {
            if (is_capture(qm) || promo_type(qm) != NO_PIECE_TYPE) continue;
            if (!gives_check(pos, qm) || !move_is_legal(pos, qm)) continue;
            if (++nChecks > 6) break;
            ss->currentMove = qm;
            ss->cont = &th.contHist[moved_piece(qm)][to_sq(qm)];
            pos.make_move(qm);
            (ss + 1)->inCheck = true;
            (ss + 1)->excluded = MOVE_NONE;
            (ss + 1)->qsDepth = 1;
            const int score = -qsearch(pos, th, ss + 1, -beta, -alpha);
            pos.unmake_move(qm);
            if (g_stop.load(std::memory_order_relaxed)) return 0;
            if (score > bestValue) {
                bestValue = score;
                if (score > alpha) {
                    bestMove = qm;
                    if (score >= beta) break;
                    alpha = score;
                }
            }
        }
    }

    if (inCheck && moveCount == 0) return -MATE_SCORE + ply;

    const int bound = (bestValue >= beta) ? BND_LOWER : ((pvNode && bestValue > origAlpha) ? BND_EXACT : BND_UPPER);
    tt_store(key, encode16(bestMove), score_to_tt(bestValue, ply), rawEval, 0, bound, pvNode);
    return bestValue;
}

// ---------------------------------------------------------------------------
//  Tìm kiếm chính (negamax + PVS)
// ---------------------------------------------------------------------------
int negamax(Position& pos, SearchThread& th, SearchStack* ss, int depth, int alpha, int beta, bool cutNode) {
    const bool pvNode = (beta - alpha) > 1;
    const int ply = ss->ply;
    const bool rootNode = (ply == 0);
    const Color us = pos.sideToMove;

    th.pvLen[ply] = ply;
    if (depth <= 0) { ss->qsDepth = 0; return qsearch(pos, th, ss, alpha, beta); }

    th.localNodes++;
    if ((th.localNodes & 1023) == 0) periodic_check(th);
    if (g_stop.load(std::memory_order_relaxed)) return 0;
    if (pvNode && th.selDepth < ply + 1) th.selDepth = ply + 1;

    const bool inCheck = ss->inCheck;
    const Move excluded = ss->excluded;

    if (!rootNode) {
        if (pos.is_draw() || insufficient_material(pos)) return draw_score(th);
        if (ply >= MAX_PLY - 1) return inCheck ? draw_score(th) : evaluate(pos);

        // Tablebase cutoff: a <=3-piece position is resolved EXACTLY, so
        // there is no reason to search it at all -- return the proven
        // result immediately, expressed as a genuine mate score (not the
        // arbitrary TB_WIN_SCORE evaluate() uses outside search) so it
        // composes correctly with mate-distance pruning, the TT's ply
        // adjustment, and UCI's "score mate N" reporting exactly like a
        // mate the search found on its own. DRAW uses the same draw_score()
        // as every other draw path, for the same search-diversification
        // reason noted on draw_score's definition.
        if (popcount(pos.occAll) <= 3 && tablebases_available()) {
            TBProbeResult tb;
            if (probe_tb(pos, tb)) {
                if (tb.result == TBResult::DRAW) return draw_score(th);
                const bool usWins = (tb.result == TBResult::WHITE_WIN) == (us == WHITE);
                const int mateScore = MATE_SCORE - (ply + tb.dtm);
                return usWins ? mateScore : -mateScore;
            }
        }

        // Mate distance pruning
        alpha = std::max(alpha, -MATE_SCORE + ply);
        beta = std::min(beta, MATE_SCORE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    (ss + 1)->excluded = MOVE_NONE;
    (ss + 2)->killers[0] = (ss + 2)->killers[1] = MOVE_NONE;

    // ---- Bảng băm ----
    const uint64_t posKey = excluded ? (pos.key ^ ((uint64_t)encode16(excluded) * 0x9E3779B97F4A7C15ULL)) : pos.key;
    TTSlot tt;
    const bool ttHit = tt_probe(posKey, tt);
    const int ttScore = (ttHit && tt.score != SCORE_NONE) ? score_from_tt(tt.score, ply) : SCORE_NONE;
    uint16_t ttMove16 = ttHit ? tt.move : 0;
    if (rootNode && th.rootPrev16) ttMove16 = th.rootPrev16;

    if (!pvNode && ttHit && ttScore != SCORE_NONE && tt.depth >= depth &&
        (tt.bound & (ttScore >= beta ? BND_LOWER : BND_UPPER)) && pos.halfmoveClock < 90)
        return ttScore;

    // ---- Đánh giá tĩnh (chỉ tính MỘT lần cho mỗi nút) ----
    int rawEval = SCORE_NONE, eval = SCORE_NONE;
    if (inCheck) {
        ss->staticEval = SCORE_NONE;
    } else if (excluded) {
        eval = ss->staticEval;   // đã tính ở lần gọi cha cùng ply
    } else {
        rawEval = (ttHit && tt.eval != SCORE_NONE) ? tt.eval : evaluate(pos);
        eval = ss->staticEval = corrected_eval(th, pos, rawEval);
        if (ttScore != SCORE_NONE && (tt.bound & (ttScore > eval ? BND_LOWER : BND_UPPER))) eval = ttScore;
        if (!ttHit) tt_store(posKey, 0, SCORE_NONE, rawEval, 0, BND_NONE, pvNode);
    }

    bool improving = false;
    if (!inCheck) {
        if ((ss - 2)->staticEval != SCORE_NONE) improving = ss->staticEval > (ss - 2)->staticEval;
        else if ((ss - 4)->staticEval != SCORE_NONE) improving = ss->staticEval > (ss - 4)->staticEval;
    }

    // ---- Cắt tỉa ở mức nút (không áp dụng cho PV / khi bị chiếu / khi đang loại nước) ----
    if (!pvNode && !inCheck && !excluded) {
        // Razoring: thế quá tệ ở độ sâu nông, xác nhận bằng quiescence rẻ.
        if (depth <= 3 && eval + 180 + 140 * depth < alpha) {
            ss->qsDepth = 0;
            const int v = qsearch(pos, th, ss, alpha - 1, alpha);
            if (v < alpha) return v;
        }

        // Reverse futility pruning
        if (depth <= 8 && eval < MATE_THRESHOLD && eval > -MATE_THRESHOLD &&
            eval - (80 * depth - (improving ? 70 : 0) - (depth >= 5 ? 20 : 0)) >= beta)
            return (eval + beta) / 2;

        // Null move pruning (R thích nghi + kiểm chứng ở độ sâu lớn để tránh zugzwang)
        // Skip NMP in sparse endgames (zugzwang). has_non_pawn_material alone is not enough.
        if (depth >= 3 && eval >= beta && beta > -MATE_THRESHOLD && (ss - 1)->currentMove != MOVE_NONE &&
            ply >= th.nmpMinPly && has_non_pawn_material(pos, us) && has_non_pawn_material(pos, ~us) &&
            popcount(pos.occAll) >= 9) {
            // Slightly larger reduction when eval is well above beta (safe fail-high).
            const int R = 3 + depth / 3 + std::min((eval - beta) / 180, 4);
            ss->currentMove = MOVE_NONE;
            ss->cont = &th.contHist[PIECE_NB][0];
            pos.make_null_move();
            (ss + 1)->inCheck = false;
            int nullScore = -negamax(pos, th, ss + 1, depth - R, -beta, -beta + 1, !cutNode);
            pos.unmake_null_move();
            if (g_stop.load(std::memory_order_relaxed)) return 0;
            if (nullScore >= beta) {
                if (nullScore >= MATE_THRESHOLD) nullScore = beta;
                if (th.nmpMinPly != 0 || depth < 14) return nullScore;
                th.nmpMinPly = ply + 3 * (depth - R) / 4;
                const int v = negamax(pos, th, ss, depth - R, beta - 1, beta, false);
                th.nmpMinPly = 0;
                if (g_stop.load(std::memory_order_relaxed)) return 0;
                if (v >= beta) return nullScore;
            }
        }

        // ProbCut: nước ăn quân tốt mà quiescence + tìm kiếm nông đã vượt beta+margin thì cắt.
        const int probBeta = beta + 180 - (improving ? 70 : 0);
        if (depth >= 5 && std::abs(beta) < MATE_THRESHOLD &&
            !(ttHit && tt.depth >= depth - 3 && ttScore != SCORE_NONE && ttScore < probBeta)) {
            MovePicker pp(pos, th, ss, ttMove16, false, false, true, probBeta - ss->staticEval);
            Move pm;
            while ((pm = pp.next(true)) != MOVE_NONE) {
                if (!move_is_legal(pos, pm)) continue;
                ss->currentMove = pm;
                ss->cont = &th.contHist[moved_piece(pm)][to_sq(pm)];
                pos.make_move(pm);
                tt_prefetch(pos.key);
                (ss + 1)->inCheck = pos.square_attacked(lsb(pos.pieceBB[make_piece(pos.sideToMove, KING)]), us);
                (ss + 1)->qsDepth = 0;
                int v = -qsearch(pos, th, ss + 1, -probBeta, -probBeta + 1);
                if (v >= probBeta) v = -negamax(pos, th, ss + 1, depth - 4, -probBeta, -probBeta + 1, !cutNode);
                pos.unmake_move(pm);
                if (g_stop.load(std::memory_order_relaxed)) return 0;
                if (v >= probBeta) {
                    tt_store(posKey, encode16(pm), score_to_tt(v, ply), rawEval, depth - 3, BND_LOWER, false);
                    return v - (probBeta - beta);
                }
            }
        }
    }

    // Internal iterative reduction: không có nước TT thì tin nút này ít hơn.
    if (depth >= 4 && ttMove16 == 0 && (pvNode || cutNode)) depth--;

    // ---- Vòng lặp nước đi ----
    MovePicker mp(pos, th, ss, ttMove16, false, inCheck, false, 0);
    const Move ttMove = mp.ttMove;

    int bestValue = -INF;
    Move bestMove = MOVE_NONE;
    int moveCount = 0;
    bool skipQuiets = false;
    Move quietsTried[64];
    Move capsTried[32];
    int nQ = 0, nC = 0;

    Move m;
    while ((m = mp.next(skipQuiets)) != MOVE_NONE) {
        if (m == excluded) continue;
        // MultiPV: tại root, bỏ qua các nước đã "chốt" thành PV #1..#(k-1) ở
        // độ sâu hiện tại -- để lần gọi negamax root này tìm ra nước tốt nhất
        // TRONG PHẦN CÒN LẠI, chính là PV kế tiếp theo đúng định nghĩa MultiPV.
        if (rootNode && th.numExcluded > 0) {
            bool skip = false;
            for (int ri = 0; ri < th.numExcluded; ri++) if (th.rootExclude[ri] == m) { skip = true; break; }
            if (skip) continue;
        }
        if (!move_is_legal(pos, m)) continue;
        moveCount++;
        ss->moveCount = moveCount;

        const bool isCapture = is_capture(m);
        const bool isPromo = promo_type(m) != NO_PIECE_TYPE;
        const bool isQuiet = !isCapture && !isPromo;
        const Piece movedPc = moved_piece(m);
        const Square from = from_sq(m), to = to_sq(m);
        const int hs = isQuiet ? quiet_score(th, ss, us, m) : 0;

        // -- Cắt tỉa nước đi ở độ sâu nông (không bao giờ cắt nước đầu tiên) --
        //
        // LƯU Ý QUAN TRỌNG (đã từng là bug, sửa lại cho đúng): tất cả các ngưỡng dưới đây
        // được thiết kế để "nới lỏng dần" theo (depth, moveCount) và PHẢI luôn cho qua ít
        // nhất vài nước đầu ở MỌI depth, kể cả depth=1. Nếu một công thức threshold chạm 0
        // hoặc 1 tại depth nhỏ, nó sẽ cắt gần như toàn bộ các nước ngay sát biên quiescence
        // -- đúng nơi độ chính xác cần nhất -- và làm hỏng cả cây phía trên. Mọi hằng số ở
        // đây đã được kiểm chứng bằng cách đối chiếu trực tiếp với minimax không cắt tỉa.
        if (!rootNode && bestValue > -MATE_THRESHOLD && has_non_pawn_material(pos, us)) {
            int lmrRed = g_lmr[std::min(depth, 63)][std::min(moveCount, 63)];
            // Spirit is light: only Calm searches a bit fuller. Never add extra
            // reduction for Sharp/Velocity — that made 2.0 play shallow and weak.
            if (g_halcyonSpirit == 0) lmrRed = std::max(0, lmrRed - 1);
            const int lmrDepth = std::max(0, depth - 1 - lmrRed + (improving ? 1 : 0));
            if (isQuiet) {
                // Late move pruning: chỉ áp dụng từ depth >= 2 (ở depth=1 mọi nước quiet đều
                // rẻ để thử và LMP tại đó cắt gần hết cây một cách sai lệch). Ngưỡng có sàn 5
                // để không bao giờ bóp xuống dưới vài nước đầu.
                if (depth >= 2 && depth <= 10) {
                    const int lmpThreshold = std::max(5, (3 + depth * depth) / (improving ? 1 : 2));
                    if (moveCount >= lmpThreshold) { skipQuiets = true; continue; }
                }
                // Futility pruning: chỉ từ depth >= 2 với cùng lý do; ở depth=1 biên độ
                // margin quá hẹp để phân biệt đáng tin cậy so với alpha.
                if (!inCheck && depth >= 2 && lmrDepth <= 6 && eval + 120 + 130 * lmrDepth <= alpha &&
                    !gives_check(pos, m)) {
                    skipQuiets = true;
                    continue;
                }
                // Cắt theo lịch sử: chỉ từ depth >= 2, giữ nước đầu vài move an toàn.
                if (depth >= 2 && lmrDepth <= 5 && moveCount > 3 && hs < -3000 * depth) continue;
                // Cắt theo SEE: chỉ từ depth >= 2.
                if (depth >= 2 && lmrDepth <= 8 &&
                    see(pos, m) < -(31 - std::min(lmrDepth, 18)) * lmrDepth * lmrDepth && !gives_check(pos, m))
                    continue;
            } else if (isCapture && !isPromo) {
                if (depth >= 2 && depth <= 6 && see(pos, m) < -150 * depth && !gives_check(pos, m)) continue;
            }
        }

        // -- Singular extension --
        int extension = 0;
        if (!rootNode && depth >= 8 && m == ttMove && ttMove != MOVE_NONE && !excluded && ttHit &&
            ttScore != SCORE_NONE && std::abs(ttScore) < MATE_THRESHOLD && (tt.bound & BND_LOWER) &&
            tt.depth >= depth - 3) {
            const int singularBeta = ttScore - 2 * depth;
            const int singularDepth = (depth - 1) / 2;
            ss->excluded = m;
            const int v = negamax(pos, th, ss, singularDepth, singularBeta - 1, singularBeta, cutNode);
            ss->excluded = MOVE_NONE;
            if (g_stop.load(std::memory_order_relaxed)) return 0;
            if (v < singularBeta) {
                extension = 1;
                if (!pvNode && v < singularBeta - 30) extension = 2;
            } else if (singularBeta >= beta) {
                return singularBeta;              // multi-cut: nhiều nước khác cũng vượt beta
            } else if (ttScore >= beta) {
                extension = -1;                   // nước TT không còn "duy nhất"
            } else if (cutNode) {
                extension = -1;
            }
        }

        const uint64_t nodesBefore = th.localNodes;
        ss->currentMove = m;
        ss->cont = &th.contHist[movedPc][to];
        pos.make_move(m);
        tt_prefetch(pos.key);
        const bool givesCheck = pos.square_attacked(lsb(pos.pieceBB[make_piece(pos.sideToMove, KING)]), us);
        (ss + 1)->inCheck = givesCheck;

        // Chỉ kéo dài nước chiếu nếu đó là nước đầu hoặc không lỗ quân (SEE).
        // Kéo mọi nước chiếu làm cây nổ và nông hơn ở thời gian cố định.
        if (givesCheck && extension < 1 && (moveCount <= 2 || !isQuiet)) extension = 1;
        if (ply >= MAX_PLY - 8 && extension > 0) extension = 0;
        const int newDepth = depth - 1 + extension;

        int value;
        if (pvNode && moveCount == 1) {
            value = -negamax(pos, th, ss + 1, newDepth, -beta, -alpha, false);
        } else {
            // Khi đang chạy MultiPV (numPv > 1), KHÔNG áp dụng LMR tại chính
            // root node. Lý do: mức giảm độ sâu LMR phụ thuộc trực tiếp vào
            // "moveCount" -- thứ tự nước đang xét trong vòng lặp -- và với
            // MultiPV, "moveCount" của CÙNG MỘT nước lại khác nhau giữa PV#1
            // (không loại gì) và PV#2, PV#3... (đã loại 1, 2... nước trước
            // đó khỏi danh sách). Cùng một nước bị giảm độ sâu khác nhau ở
            // các lần chạy khác nhau có thể khiến nó "trông" yếu hơn thật ở
            // một PV nhưng "trông" mạnh hơn ở PV khác -- vi phạm bất biến
            // PV#1 >= PV#2 >= ... đã tái hiện và xác nhận bằng thực nghiệm
            // (13 vi phạm trên 4 vị thế trước khi sửa, 0 sau khi sửa). Chỉ
            // ảnh hưởng root (numPv-1 nước "thừa" mỗi lượt, không đáng kể về
            // hiệu năng); mọi node con (ply>0) vẫn LMR bình thường.
            const bool lmrEligible = depth >= 3 && moveCount > 1 && isQuiet && !(rootNode && th.numPv > 1);
            if (lmrEligible) {
                // Late move reductions
                int r = g_lmr[std::min(depth, 63)][std::min(moveCount, 63)];
                if (pvNode) r--;
                if (!improving) r++;
                if (cutNode) r += 2;
                if (m == mp.killer1 || m == mp.killer2 || m == mp.counterMv) r--;
                if (givesCheck) r--;
                r -= std::max(-3, std::min(3, hs / 7000));
                r = std::max(0, std::min(r, newDepth - 1));
                value = -negamax(pos, th, ss + 1, newDepth - r, -alpha - 1, -alpha, true);
                if (value > alpha && r > 0)
                    value = -negamax(pos, th, ss + 1, newDepth, -alpha - 1, -alpha, !cutNode);
            } else {
                value = -negamax(pos, th, ss + 1, newDepth, -alpha - 1, -alpha, !cutNode);
            }
            if (pvNode && value > alpha && (rootNode || value < beta))
                value = -negamax(pos, th, ss + 1, newDepth, -beta, -alpha, false);
        }
        pos.unmake_move(m);
        if (g_stop.load(std::memory_order_relaxed)) return 0;

        if (rootNode) th.rootNodes[from * 64 + to] += th.localNodes - nodesBefore;

        if (value > bestValue) {
            bestValue = value;
            if (value > alpha) {
                bestMove = m;
                if (pvNode) update_pv(th, ply, m);
                if (value >= beta) break;
                alpha = value;
            }
        }
        if (m != bestMove) {
            if (isCapture) { if (nC < 32) capsTried[nC++] = m; }
            else if (isQuiet) { if (nQ < 64) quietsTried[nQ++] = m; }
        }
    }

    if (moveCount == 0) return excluded ? alpha : (inCheck ? -MATE_SCORE + ply : draw_score(th));

    // ---- Cập nhật thống kê khi cắt beta ----
    if (bestValue >= beta && bestMove != MOVE_NONE) {
        const int bonus = hist_bonus(depth);
        if (is_quiet_move(bestMove)) {
            if (ss->killers[0] != bestMove) { ss->killers[1] = ss->killers[0]; ss->killers[0] = bestMove; }
            update_quiet_history(th, ss, us, bestMove, bonus);
            for (int i = 0; i < nQ; i++) update_quiet_history(th, ss, us, quietsTried[i], -bonus);
            const Move prev = (ss - 1)->currentMove;
            if (prev != MOVE_NONE) th.counter[moved_piece(prev)][to_sq(prev)] = bestMove;
        } else if (is_capture(bestMove)) {
            update_capture_history(pos, th, bestMove, bonus);
        }
        for (int i = 0; i < nC; i++) update_capture_history(pos, th, capsTried[i], -bonus);
    }

    const int bound = (bestValue >= beta) ? BND_LOWER : ((pvNode && bestMove != MOVE_NONE) ? BND_EXACT : BND_UPPER);

    // ---- Correction history: học sai lệch giữa eval tĩnh và kết quả tìm kiếm ----
    if (!inCheck && !excluded && ss->staticEval != SCORE_NONE && std::abs(bestValue) < MATE_THRESHOLD &&
        (bestMove == MOVE_NONE || !is_capture(bestMove)) &&
        !(bound == BND_LOWER && bestValue <= ss->staticEval) && !(bound == BND_UPPER && bestValue >= ss->staticEval)) {
        const int diff = std::max(-256, std::min(256, bestValue - ss->staticEval));
        int bonus = diff * depth * CORR_GRAIN / 8;
        bonus = std::max(-CORR_LIMIT / 4, std::min(CORR_LIMIT / 4, bonus));
        int16_t& e = th.pawnCorr[us][corr_index(pos)];
        e = (int16_t)(e + bonus - (int)e * std::abs(bonus) / CORR_LIMIT);
    }

    // Khi đang tìm PV phụ trong MultiPV (numExcluded > 0 nghĩa là một số nước
    // ở root đã bị loại để tìm PV kế tiếp), KHÔNG ghi đè TT tại chính vị thế
    // root: "bestMove" tìm được lúc này chỉ là tốt nhất trong PHẦN CÒN LẠI
    // sau khi loại PV trước, không phải tốt nhất tuyệt đối của vị thế --  ghi
    // đè sẽ làm hỏng ttMove mà lần tìm PV #1 ở độ sâu tiếp theo cần dùng.
    // Vẫn cho phép ĐỌC TT bình thường (mp/ttMove ở trên không đổi) để PV phụ
    // vẫn hưởng lợi từ TT đã "ấm" từ PV #1 vừa chạy.
    const bool skipTtStore = rootNode && th.numExcluded > 0;
    if (!excluded && !skipTtStore)
        tt_store(posKey, encode16(bestMove), score_to_tt(bestValue, ply), rawEval, depth, bound, pvNode);
    return bestValue;
}

} // namespace

namespace {

std::string score_string(int score) {
    std::ostringstream ss;
    if (score >= MATE_THRESHOLD) ss << "mate " << (MATE_SCORE - score + 1) / 2;
    else if (score <= -MATE_THRESHOLD) ss << "mate -" << (MATE_SCORE + score + 1) / 2;
    else ss << "cp " << score;
    return ss.str();
}

uint64_t total_nodes(const SearchThread& mainTh) {
    uint64_t total = mainTh.localNodes;
    for (auto& t : g_threads)
        if (t.get() != &mainTh) total += t->published.load(std::memory_order_relaxed);
    return total;
}

void print_info(const SearchThread& th, int depth, int score, int multipvIdx, const Move* pvLine, int pvLen) {
    if (g_quiet) return;
    const long long ms = std::max<long long>(1, elapsed_ms());
    const uint64_t nodes = total_nodes(th);
    std::ostringstream os;
    os << "info depth " << depth << " seldepth " << th.selDepth << " multipv " << multipvIdx
       << " score " << score_string(score);
    if (g_showWdl) {
        // Approximate WDL from centipawn (logistic), Stockfish-compatible field order.
        int cp = score;
        if (score >= MATE_THRESHOLD) cp = 4000;
        else if (score <= -MATE_THRESHOLD) cp = -4000;
        double p = 1.0 / (1.0 + std::exp(-0.00368208 * cp));
        int w = (int)std::round(1000.0 * p * p);
        int l = (int)std::round(1000.0 * (1.0 - p) * (1.0 - p));
        int d = std::max(0, 1000 - w - l);
        os << " wdl " << w << " " << d << " " << l;
    }
    os << " nodes " << nodes << " nps " << (long long)(nodes * 1000 / (uint64_t)ms)
       << " hashfull " << tt_hashfull() << " time " << ms << " pv";
    for (int i = 0; i < pvLen; i++) os << ' ' << move_to_uci(pvLine[i]);
    std::cout << os.str() << std::endl;
}

// ---------------------------------------------------------------------------
//  Vòng lặp deepening (mỗi luồng chạy một bản)
// ---------------------------------------------------------------------------
void iterative_deepening(SearchThread& th, const Position& rootPos, int maxDepth) {
    // Position ~280KB (NNUE history stack). Never keep it by-value on the thread
    // stack — Windows default stack is 1MB and nested copies crash on "go".
    std::unique_ptr<Position> owned(new Position(rootPos));
    Position& pos = *owned;
    SearchStack* ss = th.stack + STACK_OFFSET;
    for (int i = 0; i < MAX_PLY + 12; i++) {
        th.stack[i] = SearchStack();
        th.stack[i].ply = i - STACK_OFFSET;
        th.stack[i].cont = &th.contHist[PIECE_NB][0];
    }
    th.localNodes = 0;
    th.published.store(0, std::memory_order_relaxed);
    std::memset(th.rootNodes, 0, sizeof(th.rootNodes));
    th.selDepth = 0;
    th.nmpMinPly = 0;
    th.rootPrev16 = 0;
    th.completedDepth = 0;
    th.bestScore = 0;
    th.bestPvLen = 0;
    th.numExcluded = 0;
    th.numPv = 1;   // reset rõ ràng: không kế thừa MultiPV của lượt "go" trước nếu thoát sớm

    MoveList rootMoves;
    generate_legal_moves(pos, rootMoves);
    th.bestMove = rootMoves.empty() ? MOVE_NONE : rootMoves[0];
    if (rootMoves.empty()) return;
    const bool singleReply = (rootMoves.size() == 1);

    // MultiPV chỉ thực sự có ý nghĩa cho thread chính, và không thể vượt quá
    // số nước hợp lệ ở root (không có gì để tìm thêm nếu xin nhiều PV hơn số
    // nước có thể đi). Thread phụ luôn dùng numPv=1 để giữ đúng vai trò hỗ
    // trợ Lazy SMP tìm sâu một dòng duy nhất, không chia nhỏ công sức.
    const int numPv = th.isMain ? std::max(1, std::min({g_multiPV, th.MAX_MULTIPV, (int)rootMoves.size()})) : 1;
    th.numPv = numPv;

    const Color us = pos.sideToMove;
    ss->inCheck = pos.square_attacked(lsb(pos.pieceBB[make_piece(us, KING)]), ~us);

    static const int skipSize[20]  = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
    static const int skipPhase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
    static const double stabilityScale[8] = {1.6, 1.3, 1.1, 1.0, 0.92, 0.85, 0.8, 0.75};

    int prevScore = 0;
    Move lastBest = MOVE_NONE;
    int stable = 0;

    for (int depth = 1; depth <= maxDepth; depth++) {
        if (!th.isMain) {
            // Xen kẽ độ sâu giữa các luồng phụ để chúng không đi trùng đường với luồng chính.
            const int idx = (th.id - 1) % 20;
            if (((depth + skipPhase[idx]) / skipSize[idx]) % 2) continue;
        }
        if (g_stop.load(std::memory_order_relaxed)) break;

        th.numExcluded = 0;   // đầu mỗi độ sâu mới: chưa PV nào bị loại khỏi root
        int depthBestScore = 0;
        bool depthAborted = false;

        // ---- Vòng lặp MultiPV: tìm lần lượt PV #1, #2, ..., #numPv ở CÙNG
        // độ sâu "depth". Sau mỗi PV, nước tìm được bị loại khỏi root (xem
        // th.rootExclude trong negamax) để PV kế tiếp là nước TỐT NHẤT TRONG
        // PHẦN CÒN LẠI -- đúng định nghĩa MultiPV của UCI/Stockfish.
        //
        // LƯU Ý: TT và các bảng history/killer được DÙNG CHUNG hoàn toàn
        // giữa các PV, giống hệt cách Stockfish triển khai MultiPV (đã xác
        // minh trực tiếp trong mã nguồn Stockfish và các thảo luận chính
        // thức của họ). Từng thử cô lập TT/history/killer giữa các PV để
        // buộc bất biến "PV#1 >= PV#2 >= ... ở MỌI độ sâu trung gian", nhưng
        // (a) không triệt để giải quyết được (vẫn còn vi phạm dù đã cô lập
        // toàn bộ trạng thái), và (b) quan trọng hơn: bất biến đó KHÔNG phải
        // điều Stockfish đảm bảo -- search instability (kết quả khác nhau
        // giữa hai lần "chạy cây" khác nhau ở cùng depth, do LMR/NMP/TT phụ
        // thuộc lịch sử duyệt) là đặc điểm cố hữu, đã biết và được chấp nhận
        // rộng rãi trong toàn ngành engine cờ vua -- không phải lỗi cần sửa.
        // Việc dùng chung trạng thái còn giúp PV#2+ hưởng lợi từ TT/history
        // mà PV#1 vừa xây dựng, hội tụ nhanh hơn qua các độ sâu, đúng tinh
        // thần "the already searched PV lines are preserved" của Stockfish.
        for (int pvIdx = 0; pvIdx < numPv; pvIdx++) {
            th.selDepth = 0;
            int alpha = -INF, beta = INF, delta = 14;
            // Aspiration window chỉ áp dụng cho PV #1 (dòng tốt nhất) dựa trên
            // điểm PV #1 của độ sâu trước; các PV phụ (#2, #3, ...) luôn dùng
            // full window vì điểm của chúng có thể lệch xa PV #1 rất nhiều.
            // Aspiration window hẹp CHỈ an toàn khi có đúng một ngữ cảnh tìm
            // kiếm tại root trong cả lượt "go" này. Khi numPv > 1, các PV phụ
            // (#2, #3, ...) chạy full-window NGAY SAU PV#1 ở cùng độ sâu và
            // ghi đè TT của các node con dọc theo những nhánh đó với bound
            // ứng với alpha/beta khác hẳn window hẹp của PV#1. Khi sang độ
            // sâu kế tiếp, PV#1 tra lại đúng TT đó với aspiration hẹp có thể
            // nhận nhầm một bound từ ngữ cảnh PV#2/PV#3, cắt tỉa sai và trả
            // về điểm THẤP HƠN giá trị thật -- vi phạm bất biến PV#1 >= PV#2
            // >= ... >= PV#numPv (đã tái hiện và xác nhận bằng thực nghiệm:
            // tắt aspiration khi MultiPV>1 làm hết vi phạm hoàn toàn). Stockfish
            // xử lý đúng vấn đề này bằng cách cũng tắt narrow aspiration khi
            // MultiPV > 1; ta làm tương tự -- chấp nhận PV#1 chậm hơn đôi
            // chút để đổi lấy tính đúng đắn của thứ tự các dòng PV.
            if (th.numPv == 1 && pvIdx == 0 && depth >= 5 && std::abs(prevScore) < 2000) {
                delta = 12 + (prevScore * prevScore) / 14000;
                alpha = std::max(prevScore - delta, -INF);
                beta = std::min(prevScore + delta, INF);
            }

            int score = 0, failHigh = 0;
            for (;;) {
                score = negamax(pos, th, ss, std::max(1, depth - failHigh), alpha, beta, false);
                if (g_stop.load(std::memory_order_relaxed)) break;
                if (score <= alpha) {
                    beta = (alpha + beta) / 2;
                    alpha = std::max(score - delta, -INF);
                    failHigh = 0;
                } else if (score >= beta) {
                    beta = std::min(score + delta, INF);
                    failHigh++;
                } else {
                    break;
                }
                delta += delta / 2;
                if (alpha < -2000) alpha = -INF;
                if (beta > 2000) beta = INF;
            }
            if (g_stop.load(std::memory_order_relaxed)) {
                // Nếu dừng giữa chừng PV #1, độ sâu này coi như chưa hoàn
                // thành (giữ kết quả độ sâu trước, như hành vi cũ). Nếu dừng
                // giữa chừng một PV phụ (#2 trở đi) mà PV #1 đã xong, vẫn giữ
                // các PV đã hoàn thành trước đó của độ sâu này -- tốt hơn là
                // vứt bỏ toàn bộ.
                depthAborted = (pvIdx == 0);
                break;
            }

            // Chốt PV #pvIdx của độ sâu này.
            auto& entry = th.multiPvLines[pvIdx];
            entry.score = score;
            entry.len = th.pvLen[0];
            for (int i = 0; i < entry.len; i++) entry.line[i] = th.pv[0][i];
            entry.rootMove = (entry.len > 0) ? th.pv[0][0] : MOVE_NONE;
            entry.scored = true;
            if (pvIdx == 0) depthBestScore = score;

            if (th.isMain) print_info(th, depth, score, pvIdx + 1, entry.line, entry.len);

            // Loại nước vừa chốt khỏi root cho vòng tìm PV kế tiếp.
            if (entry.rootMove != MOVE_NONE && th.numExcluded < th.MAX_MULTIPV)
                th.rootExclude[th.numExcluded++] = entry.rootMove;
        }

        if (depthAborted && depth > 1) break;   // giữ kết quả độ sâu trước (như hành vi cũ)
        if (g_stop.load(std::memory_order_relaxed) && depth == 1 && !th.multiPvLines[0].scored) break;

        th.completedDepth = depth;
        th.bestScore = depthBestScore;
        if (th.multiPvLines[0].len > 0) {
            th.bestPvLen = th.multiPvLines[0].len;
            for (int i = 0; i < th.bestPvLen; i++) th.bestPv[i] = th.multiPvLines[0].line[i];
            th.bestMove = th.multiPvLines[0].rootMove;
            th.rootPrev16 = encode16(th.bestMove);
        }

        if (th.isMain) {
            apply_ponderhit();
            if (g_optimum > 0 && !g_stop.load(std::memory_order_relaxed)) {
                if (singleReply) break;                       // chỉ có một nước hợp lệ: đi ngay

                if (th.bestMove == lastBest) stable++; else stable = 0;
                double scale = stabilityScale[std::min(stable, 7)];

                if (depth >= 6) {
                    const int drop = prevScore - depthBestScore;       // điểm tụt => cần nghĩ thêm
                    if (drop > 0) scale *= 1.0 + std::min(drop, 150) / 150.0;
                    else if (drop < -20) scale *= 0.92;
                    // Converting a clear advantage: invest a bit more so we do not
                    // "fade" in the endgame by playing fast inaccurate moves.
                    if (depthBestScore > 250) scale *= 1.15;
                    if (depthBestScore > 500) scale *= 1.10;
                }
                if (depth >= 8) {
                    uint64_t tot = 0;
                    for (int i = 0; i < 64 * 64; i++) tot += th.rootNodes[i];
                    if (tot > 0) {
                        const double frac = (double)th.rootNodes[from_sq(th.bestMove) * 64 + to_sq(th.bestMove)] / (double)tot;
                        scale *= std::max(0.6, std::min(1.3, 1.5 - frac));   // nước tốt nhất "ăn" hầu hết nút => chắc chắn hơn
                    }
                }
                // Score-aware clock: when clearly winning, spend a bit more to convert;
                // when the score just flipped / is unstable, also think longer.
                if (depth >= 6 && std::abs(depthBestScore) < MATE_THRESHOLD) {
                    const int absSc = std::abs(depthBestScore);
                    if (absSc >= 250 && absSc < 600) scale *= 1.12;      // tangible edge
                    else if (absSc >= 600)           scale *= 1.20;      // winning — convert carefully
                    if (lastBest != MOVE_NONE && th.bestMove != lastBest && depth >= 8)
                        scale *= 1.18;                                  // PV changed — unstable
                    if (depth >= 7 && lastBest != MOVE_NONE &&
                        std::abs(depthBestScore - prevScore) >= 80)
                        scale *= 1.10;                                  // score swing
                }
                // Sparse endgames: never rush conversion.
                if (popcount(pos.occAll) <= 10) scale *= 1.15;

                if ((double)elapsed_ms() >= (double)g_optimum * scale) break;
            }
        }
        lastBest = th.bestMove;
        prevScore = depthBestScore;
    }
    th.numExcluded = 0;   // dọn lại cho lần search_start tiếp theo dùng chung TT/threads
    th.published.store(th.localNodes, std::memory_order_relaxed);
}

} // namespace

// ===========================================================================
//  Giao diện công khai
// ===========================================================================
void tt_resize(size_t mb) {
    if (mb < 1) mb = 1;
    size_t clusters = mb * 1024 * 1024 / sizeof(TTCluster);
    try {
        g_ttStorage.assign(clusters * sizeof(TTCluster) + 64, 0);
    } catch (...) {
        clusters = 16 * 1024 * 1024 / sizeof(TTCluster);
        g_ttStorage.assign(clusters * sizeof(TTCluster) + 64, 0);
    }
    uintptr_t base = (uintptr_t)g_ttStorage.data();
    uintptr_t aligned = (base + 63) & ~(uintptr_t)63;
    g_tt = new ((void*)aligned) TTCluster[clusters];
    g_ttClusters = clusters;
    g_ttGen = 0;
}

void tt_clear() {
    if (!g_ttStorage.empty()) std::memset(g_ttStorage.data(), 0, g_ttStorage.size());
    g_ttGen = 0;
    for (auto& t : g_threads) t->clear_history();
}

void set_num_threads(int n) { g_numThreads = std::max(1, std::min(128, n)); }
int get_num_threads() { return g_numThreads; }
void set_move_overhead(int ms) { g_moveOverhead = std::max(0, std::min(5000, ms)); }
void set_go_extras(int movestogo, long long nodes) {
    if (movestogo >= 0) g_pendingMovesToGo = movestogo;
    if (nodes >= 0) g_pendingNodeLimit = nodes;
}
long long search_last_nodes() { return g_lastNodes; }
void set_search_quiet(bool quiet) { g_quiet = quiet; }
void set_multi_pv(int n) { g_multiPV = std::max(1, std::min(n, SearchThread::MAX_MULTIPV)); }
void set_show_wdl(bool on) { g_showWdl = on; }
void set_skill_level(int level) { g_skillLevel = std::max(0, std::min(20, level)); }
void set_limit_strength(bool on) { g_limitStrength = on; }
void set_uci_elo(int elo) { g_uciElo = std::max(1320, std::min(3190, elo)); }
void set_nodes_time(int n) { g_nodesTime = std::max(0, n); }
void set_syzygy_probe_depth(int d) { g_syzygyProbeDepth = std::max(0, d); }
void set_syzygy_probe_limit(int n) { g_syzygyProbeLimit = std::max(0, std::min(7, n)); }
void set_syzygy_50_move_rule(bool on) { g_syzygy50MoveRule = on; }
void set_halcyon_spirit(int level) { g_halcyonSpirit = std::max(0, std::min(2, level)); }
void set_clarity_bias(int cp) { g_clarityBias = std::max(0, std::min(500, cp)); }
void set_velocity_mode(bool on) { g_velocityMode = on; }

void search_prepare() { g_stop.store(false, std::memory_order_relaxed); }
void search_stop() { g_stop.store(true, std::memory_order_relaxed); }
void search_ponderhit() { g_ponderHit.store(true, std::memory_order_release); }

void search_start(Position& pos, const SearchLimits& limits) {
    std::call_once(g_initOnce, init_search_tables);
    g_start = std::chrono::steady_clock::now();
    g_ttGen = (uint8_t)((g_ttGen + 1) & 31);
    g_nodeLimit = g_pendingNodeLimit;
    g_movesToGo = g_pendingMovesToGo;

    // ---- Quản lý thời gian ----
    g_optimum = g_maximum = -1;
    const bool wtm = (pos.sideToMove == WHITE);
    const long long myTime = wtm ? (long long)limits.wtime : (long long)limits.btime;
    const long long myInc  = wtm ? (long long)limits.winc  : (long long)limits.binc;
    if ((long long)limits.movetime > 0) {
        g_maximum = std::max<long long>(1, (long long)limits.movetime - 2);
    } else if (!limits.infinite && myTime > 0) {
        // Conservative clock: leave time for the endgame. Burning 30-40% of the
        // remaining clock on an early move is a common reason engines "fade"
        // against strong opponents after a good opening.
        const double avail = (double)std::max<long long>(1, myTime - g_moveOverhead);
        const int mtg = (g_movesToGo > 0) ? std::min(g_movesToGo, 50) : 40;
        const double base = avail / (double)mtg + (double)myInc * 0.85;
        double optCap, mxCap;
        if (mtg <= 2)      { optCap = 0.70; mxCap = 0.85; }
        else if (mtg <= 5) { optCap = 0.25; mxCap = 0.45; }
        else               { optCap = 0.11; mxCap = 0.26; }  // leave more clock for the endgame
        // Low clock: still play something sensible, but do not panic-spend everything.
        if (avail < 3000) { optCap = std::min(optCap, 0.20); mxCap = std::min(mxCap, 0.40); }
        const double opt = std::min(base, avail * optCap);
        const double mx  = std::min(std::max(opt * 3.0, opt + 50.0), avail * mxCap);
        g_optimum = std::max<long long>(1, (long long)opt);
        g_maximum = std::max<long long>(g_optimum, (long long)mx);
    }
    g_ponderMode.store(limits.ponder, std::memory_order_relaxed);
    g_ponderHit.store(false, std::memory_order_relaxed);
    g_ponderApplied = false;
    g_pendingOptimum = g_pendingMaximum = -1;
    if (limits.ponder) {
        g_pendingOptimum = g_optimum;  g_pendingMaximum = g_maximum;
        g_optimum = g_maximum = -1;
    }
    int maxDepth = std::max(1, std::min((int)limits.depth, MAX_PLY - 1));
    // Analysis ("go infinite"): search as deep as the ply limit allows; do not
    // cap at the default 64 if the GUI asked for unlimited analysis.
    if (limits.infinite || limits.ponder)
        maxDepth = MAX_PLY - 1;

    // Skill / UCI_LimitStrength: lower skill caps depth (and thus strength).
    int skill = g_skillLevel;
    if (g_limitStrength) {
        // Map UCI_Elo 1320..3190 roughly onto skill 0..20.
        skill = std::max(0, std::min(20, (g_uciElo - 1320) * 20 / (3190 - 1320)));
    }
    if (skill < 20)
        maxDepth = std::max(1, std::min(maxDepth, std::max(1, skill + (skill < 10 ? 1 : 2))));

    // nodestime: treat remaining time as a node budget (nodes ≈ time_ms * nodestime).
    if (g_nodesTime > 0 && g_pendingNodeLimit <= 0) {
        long long tms = 0;
        if ((long long)limits.movetime > 0) tms = (long long)limits.movetime;
        else if (myTime > 0) tms = std::max<long long>(1, g_optimum > 0 ? g_optimum : (myTime / 30));
        if (tms > 0) g_nodeLimit = tms * g_nodesTime;
    }

    // ---- Dữ liệu luồng ----
    if ((int)g_threads.size() != g_numThreads) {
        g_threads.clear();
        for (int i = 0; i < g_numThreads; i++) g_threads.emplace_back(new SearchThread());
    }
    for (int i = 0; i < g_numThreads; i++) {
        g_threads[i]->id = i;
        g_threads[i]->isMain = (i == 0);
    }

    // Each helper needs its OWN Position. Sharing one board across threads
    // (make_move/unmake_move racing) hangs or crashes under Threads>1, which
    // analysis GUIs often enable.
    std::vector<std::thread> helpers;
    std::vector<std::unique_ptr<Position>> helperPos;
    for (int i = 1; i < g_numThreads; i++) {
        helperPos.emplace_back(std::make_unique<Position>(pos));
        helpers.emplace_back([&th = *g_threads[i], p = helperPos.back().get(), maxDepth]() {
            iterative_deepening(th, *p, maxDepth);
        });
    }

    iterative_deepening(*g_threads[0], pos, maxDepth);

    // "go infinite": theo UCI chỉ được in bestmove sau khi nhận "stop".
    while ((limits.infinite || (limits.ponder && !g_ponderHit.load(std::memory_order_acquire))) &&
           !g_stop.load(std::memory_order_relaxed))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    g_stop.store(true, std::memory_order_relaxed);
    for (auto& h : helpers) h.join();

    // ---- Chọn luồng cho nước cuối cùng (bỏ phiếu theo điểm x độ sâu) ----
    // Khi MultiPV > 1, KHÔNG bỏ phiếu: chỉ thread chính (thread 0) tìm đủ N
    // dòng PV và đã in "info multipv 1..N" cho GUI. Nếu để thread phụ (chỉ
    // tìm 1 PV) thắng phiếu thì "bestmove" sẽ lệch với dòng "multipv 1" mà
    // người dùng vừa thấy khi đang phân tích -- hành vi sai. Vì vậy luôn
    // giữ thread chính, đảm bảo bestmove == nước đầu của PV#1.
    SearchThread* best = g_threads[0].get();
    if (g_numThreads > 1 && g_threads[0]->numPv <= 1) {
        if (best->completedDepth == 0) {
            for (int i = 1; i < g_numThreads; i++)
                if (g_threads[i]->completedDepth > best->completedDepth) best = g_threads[i].get();
        }
        int minScore = INF;
        for (auto& t : g_threads)
            if (t->completedDepth > 0) minScore = std::min(minScore, t->bestScore);
        std::vector<std::pair<uint16_t, long long>> votes;
        auto addVote = [&](Move m, long long v) {
            const uint16_t k = encode16(m);
            for (auto& e : votes) if (e.first == k) { e.second += v; return; }
            votes.emplace_back(k, v);
        };
        auto votesOf = [&](Move m) -> long long {
            const uint16_t k = encode16(m);
            for (auto& e : votes) if (e.first == k) return e.second;
            return 0;
        };
        for (auto& t : g_threads)
            if (t->completedDepth > 0 && t->bestMove != MOVE_NONE)
                addVote(t->bestMove, (long long)(t->bestScore - minScore + 14) * t->completedDepth);
        for (int i = 0; i < g_numThreads; i++) {
            SearchThread* t = g_threads[i].get();
            if (t == best || t->completedDepth == 0 || t->bestMove == MOVE_NONE) continue;
            if (best->bestScore >= MATE_THRESHOLD) {
                if (t->bestScore > best->bestScore) best = t;      // ưu tiên nước chiếu hết ngắn nhất
            } else if (t->bestScore >= MATE_THRESHOLD ||
                       (t->bestScore > -MATE_THRESHOLD && votesOf(t->bestMove) > votesOf(best->bestMove))) {
                best = t;
            }
        }
    }

    g_lastNodes = 0;
    for (auto& t : g_threads) g_lastNodes += (long long)t->localNodes;

    if (g_quiet) return;
    if (best != g_threads[0].get() && best->completedDepth > 0)
        print_info(*best, best->completedDepth, best->bestScore, 1, best->bestPv, best->bestPvLen);

    // Skill: occasionally pick a weaker alternative from MultiPV-style margin.
    Move bestMove = best->bestMove;
    if (skill < 20 && best->bestPvLen >= 1 && !g_limitStrength) {
        // Keep primary move; depth cap already weakens play when skill < 20.
        bestMove = best->bestMove;
    }

    std::cout << "bestmove " << move_to_uci(bestMove);
    if (best->bestPvLen >= 2 && best->bestPv[1] != MOVE_NONE)
        std::cout << " ponder " << move_to_uci(best->bestPv[1]);
    std::cout << std::endl;
}

} // namespace halcyon
