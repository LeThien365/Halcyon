#include "position.h"
#include "movegen.h"
#include "search.h"
#include "attacks.h"
#include "zobrist.h"
#include "eval.h"
#include "nnue.h"
#include "polyglot.h"
#include "tablebase.h"
#include <random>
#include <thread>
#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

using namespace halcyon;

static Position pos;
static bool ownBook = true;
static bool ponderEnabled = false;
static bool chess960 = false;
static int contempt = 0;
static bool silenceUCI = false;
static bool normalizeScores = true;
static bool useSoftNodes = false;
static int hardSoftNodeFactor = 100;
static int nodeFudging = 0;
static int botMoveSeed = 0;
static std::string uciOpponent;
static std::string contemptOverrides;
static bool syzygyDTZFiltering = true;
static bool minimalMode = false;
static std::string numaPolicy = "auto";
static std::mt19937_64 bookRng(std::random_device{}());
static std::thread searchThread;
static std::ofstream debugLog;

static void dbg(const std::string& prefix, const std::string& s) {
    if (debugLog.is_open()) {
        debugLog << prefix << s << std::endl;
        debugLog.flush();
    }
}

// Any in-flight search runs on its own thread so the UCI loop keeps reading
// stdin (and can act on "stop") while a search is in progress. Call this
// before touching `pos` or starting a new search.
static void stop_and_join_search() {
    search_stop();
    if (searchThread.joinable()) searchThread.join();
}

static Move parse_move(Position& p, const std::string& moveStr) {
    MoveList moves;
    generate_legal_moves(p, moves);
    for (Move m : moves)
        if (move_to_uci(m) == moveStr) return m;
    return MOVE_NONE;
}

static void position_command(std::istringstream& iss) {
    std::string token;
    iss >> token;
    if (token == "startpos") {
        pos.set_startpos();
        iss >> token;
    } else if (token == "fen") {
        std::string fen, part;
        while (iss >> part) {
            if (part == "moves") { token = part; break; }
            fen += part + " ";
        }
        pos.set_fen(fen);
        if (part != "moves") token = "";
    }
    if (token == "moves") {
        std::string mv;
        while (iss >> mv) {
            Move m = parse_move(pos, mv);
            if (m != MOVE_NONE) pos.make_move(m);
        }
        pos.rebase();
    }
}

// Runs on the search thread: try the book first, otherwise search. Takes its
// own snapshot of the position so later "position"/"go" commands from the
// UCI loop can't race with it.
static void run_search(std::shared_ptr<Position> snapshot, SearchLimits limits) {
    if (ownBook && g_book.is_loaded()) {
        Move bookMove = g_book.probe(*snapshot, bookRng);
        if (bookMove != MOVE_NONE) {
            std::cout << "info string book move" << std::endl;
            std::cout << "bestmove " << move_to_uci(bookMove) << std::endl;
            return;
        }
    }
    search_start(*snapshot, limits);
}

static void go_command(std::istringstream& iss) {
    SearchLimits limits;
    std::string token;
    int mtg = 0;
    long long nodeLimit = 0;
    while (iss >> token) {
        if (token == "depth") iss >> limits.depth;
        else if (token == "movetime") iss >> limits.movetime;
        else if (token == "wtime") iss >> limits.wtime;
        else if (token == "btime") iss >> limits.btime;
        else if (token == "winc") iss >> limits.winc;
        else if (token == "binc") iss >> limits.binc;
        else if (token == "infinite") limits.infinite = true;
        else if (token == "ponder") limits.ponder = true;
        else if (token == "movestogo") iss >> mtg;
        else if (token == "nodes") iss >> nodeLimit;
    }
    set_go_extras(mtg, nodeLimit);
    stop_and_join_search();
    search_prepare();
    searchThread = std::thread(run_search, std::make_shared<Position>(pos), limits);
}

static long long perft(Position& p, int depth) {
    if (depth == 0) return 1;
    MoveList moves;
    generate_pseudo_moves(p, moves);
    long long nodes = 0;
    for (Move m : moves) {
        if (!move_is_legal(p, m)) continue;
        p.make_move(m);
        nodes += perft(p, depth - 1);
        p.unmake_move(m);
    }
    return nodes;
}

static void perft_command(int depth) {
    long long total = 0;
    MoveList moves;
    generate_pseudo_moves(pos, moves);
    for (Move m : moves) {
        if (!move_is_legal(pos, m)) continue;
        pos.make_move(m);
        long long n = perft(pos, depth - 1);
        pos.unmake_move(m);
        std::cout << move_to_uci(m) << ": " << n << "\n";
        total += n;
    }
    std::cout << "\nNodes searched: " << total << std::endl;
}

int main() {
    // UCI engines must flush every line immediately when stdout is a pipe (GUI).
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    std::cout << "Halcyon 2.0.1 by Thien from Vietnam" << std::endl;
    init_zobrist();
    init_attacks();
    init_eval();
    g_nnue.load_default(); // NNUE is the default evaluator from the moment the engine starts
    pos.set_startpos();
    tt_resize(64);

    std::string line;
    while (std::getline(std::cin, line)) {
        // Windows GUIs (en-croissant, Arena, ...) often send CRLF; leave \r on
        // the last token and "go infinite" becomes "infinite\r" → never matches.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        dbg(">> ", line);
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "uci") {
            std::cout << "id name Halcyon 2.0.1\n";
            std::cout << "id author Thien from Vietnam\n";
            std::cout << "option name Debug Log File type string default\n";
            std::cout << "option name NumaPolicy type combo default auto var auto var system var none\n";
            std::cout << "option name Threads type spin default 1 min 1 max 128\n";
            std::cout << "option name Hash type spin default 64 min 1 max 4096\n";
            std::cout << "option name MultiPV type spin default 1 min 1 max 256\n";
            std::cout << "option name Skill Level type spin default 20 min 0 max 20\n";
            std::cout << "option name Move Overhead type spin default 30 min 0 max 5000\n";
            std::cout << "option name nodestime type spin default 0 min 0 max 10000\n";
            std::cout << "option name UCI_Elo type spin default 1320 min 1320 max 3190\n";
            std::cout << "option name SyzygyPath type string default <empty>\n";
            std::cout << "option name SyzygyProbeDepth type spin default 1 min 0 max 100\n";
            std::cout << "option name SyzygyProbeLimit type spin default 7 min 0 max 7\n";
            std::cout << "option name NNUEFile type string default <empty>\n";
            std::cout << "option name BookFile type string default <empty>\n";
            std::cout << "option name OwnBook type check default true\n";
            std::cout << "option name Ponder type check default false\n";
            std::cout << "option name UCI_Chess960 type check default false\n";
            std::cout << "option name UCI_LimitStrength type check default false\n";
            std::cout << "option name UCI_ShowWDL type check default false\n";
            std::cout << "option name Syzygy50MoveRule type check default true\n";
            std::cout << "option name Contempt type spin default 0 min -100 max 100\n";
            std::cout << "option name BotMoveSeed type spin default 0 min 0 max 2147483647\n";
            std::cout << "option name HardSoftNodeFactor type spin default 100 min 1 max 1000\n";
            std::cout << "option name NodeFudging type spin default 0 min 0 max 100\n";
            std::cout << "option name UCI_Opponent type string default\n";
            std::cout << "option name ContemptOverrides type string default\n";
            std::cout << "option name Minimal type check default false\n";
            std::cout << "option name Normalize type check default true\n";
            std::cout << "option name SilenceUCI type check default false\n";
            std::cout << "option name SyzygyDTZFiltering type check default true\n";
            std::cout << "option name UseSoftNodes type check default false\n";
            std::cout << "option name MoveOverhead type spin default 30 min 0 max 5000\n";
            // Exclusive Halcyon personality controls (not Stockfish clones).
            std::cout << "option name HalcyonSpirit type combo default Balanced var Calm var Balanced var Sharp\n";
            std::cout << "option name ClarityBias type spin default 0 min 0 max 500\n";
            std::cout << "option name VelocityMode type check default false\n";
            std::cout << "option name HalcyonTBPath type string default <empty>\n";
            std::cout << "uciok" << std::endl;
        } else if (cmd == "isready") {
            // Never join the search here: analysis GUIs poll isready while
            // "go infinite" is running. Joining would freeze the GUI.
            std::cout << "readyok" << std::endl;
        } else if (cmd == "ucinewgame") {
            stop_and_join_search();
            pos.set_startpos();
            tt_clear();
        } else if (cmd == "position") {
            stop_and_join_search();
            position_command(iss);
        } else if (cmd == "go") {
            std::string first;
            iss >> first;
            if (first == "perft") {
                int d = 1; iss >> d;
                stop_and_join_search();
                perft_command(d);
            } else {
                std::istringstream rest(line.substr(cmd.length()));
                go_command(rest);
            }
        } else if (cmd == "setoption") {
            std::string t0;
            iss >> t0; // "name"
            // Multi-word option names: tokens until "value"
            std::string name, tok;
            while (iss >> tok) {
                if (tok == "value") break;
                if (!name.empty()) name.push_back(' ');
                name += tok;
            }
            std::string value;
            std::getline(iss, value);
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(value.begin());
            while (!value.empty() && (value.back() == ' ' || value.back() == char(13))) value.pop_back();
            auto is_true = [&](const std::string& v) {
                return v == "true" || v == "True" || v == "1";
            };

            if (name == "Hash" && !value.empty()) {
                tt_resize((size_t)std::stoul(value));
            } else if (name == "Threads" && !value.empty()) {
                set_num_threads(std::stoi(value));
            } else if (name == "NNUEFile" && !value.empty() && value != "<empty>") {
                bool ok = g_nnue.load(value);
                std::cout << (ok ? "info string NNUE loaded: " : "info string NNUE load failed: ") << value << std::endl;
            } else if (name == "BookFile" && !value.empty() && value != "<empty>") {
                bool ok = g_book.load(value);
                std::cout << (ok ? "info string book loaded (" + std::to_string(g_book.size()) + " entries): "
                                 : "info string book load failed: ") << value << std::endl;
            } else if (name == "OwnBook") {
                ownBook = is_true(value);
            } else if (name == "MultiPV" && !value.empty()) {
                set_multi_pv(std::stoi(value));
            } else if ((name == "MoveOverhead" || name == "Move Overhead") && !value.empty()) {
                set_move_overhead(std::stoi(value));
            } else if ((name == "SyzygyPath" || name == "HalcyonTBPath") && !value.empty() && value != "<empty>") {
                int n = init_tablebases(value);
                std::cout << "info string Halcyon TB loaded " << n << " file(s) from " << value << std::endl;
            } else if (name == "HalcyonSpirit") {
                if (value == "Calm") set_halcyon_spirit(0);
                else if (value == "Sharp") set_halcyon_spirit(2);
                else set_halcyon_spirit(1);
                std::cout << "info string HalcyonSpirit=" << value << std::endl;
            } else if (name == "ClarityBias" && !value.empty()) {
                set_clarity_bias(std::stoi(value));
            } else if (name == "VelocityMode") {
                set_velocity_mode(is_true(value));
            } else if (name == "Debug Log File") {
                if (debugLog.is_open()) debugLog.close();
                if (!value.empty()) {
                    debugLog.open(value, std::ios::out | std::ios::app);
                    std::cout << (debugLog.is_open() ? "info string debug log: " : "info string debug log failed: ")
                              << value << std::endl;
                }
            } else if (name == "NumaPolicy") {
                numaPolicy = value.empty() ? "auto" : value;
                std::cout << "info string NumaPolicy=" << numaPolicy << " (stored; no binding on this build)" << std::endl;
            } else if (name == "Skill Level" && !value.empty()) {
                set_skill_level(std::stoi(value));
            } else if (name == "nodestime" && !value.empty()) {
                set_nodes_time(std::stoi(value));
            } else if (name == "UCI_Elo" && !value.empty()) {
                set_uci_elo(std::stoi(value));
            } else if (name == "SyzygyProbeDepth" && !value.empty()) {
                set_syzygy_probe_depth(std::stoi(value));
            } else if (name == "SyzygyProbeLimit" && !value.empty()) {
                set_syzygy_probe_limit(std::stoi(value));
            } else if (name == "Ponder") {
                ponderEnabled = is_true(value);
            } else if (name == "UCI_Chess960") {
                chess960 = is_true(value);
                if (chess960)
                    std::cout << "info string UCI_Chess960 accepted but FRC castling is not fully supported yet" << std::endl;
            } else if (name == "UCI_LimitStrength") {
                set_limit_strength(is_true(value));
            } else if (name == "UCI_ShowWDL") {
                set_show_wdl(is_true(value));
            } else if (name == "Syzygy50MoveRule") {
                set_syzygy_50_move_rule(is_true(value));
            } else if (name == "Contempt" && !value.empty()) {
                contempt = std::stoi(value);
            } else if (name == "BotMoveSeed" && !value.empty()) {
                botMoveSeed = std::stoi(value);
                if (botMoveSeed) bookRng.seed((uint64_t)botMoveSeed);
            } else if (name == "HardSoftNodeFactor" && !value.empty()) {
                hardSoftNodeFactor = std::stoi(value);
            } else if (name == "NodeFudging" && !value.empty()) {
                nodeFudging = std::stoi(value);
            } else if (name == "UCI_Opponent") {
                uciOpponent = value;
            } else if (name == "ContemptOverrides") {
                contemptOverrides = value;
            } else if (name == "Minimal") {
                minimalMode = is_true(value);
            } else if (name == "Normalize") {
                normalizeScores = is_true(value);
            } else if (name == "SilenceUCI") {
                silenceUCI = is_true(value);
                set_search_quiet(silenceUCI);
            } else if (name == "SyzygyDTZFiltering") {
                syzygyDTZFiltering = is_true(value);
            } else if (name == "UseSoftNodes") {
                useSoftNodes = is_true(value);
            }
        } else if (cmd == "stop") {
            search_stop(); // do NOT join: search thread prints bestmove itself
        } else if (cmd == "ponderhit") {
            search_ponderhit();
        } else if (cmd == "quit") {
            stop_and_join_search();
            break;
        } else if (cmd == "d" || cmd == "print") {
            pos.print();
        } else if (cmd == "eval") {
            std::cout << "static eval (side to move perspective, cp): " << evaluate(pos) << std::endl;
        } else if (cmd == "polykey") {
            std::cout << std::hex << "0x" << polyglot_key(pos) << std::dec << std::endl;
        }
    }
    return 0;
}
