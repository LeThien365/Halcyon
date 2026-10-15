#pragma once
#include "position.h"
#include <cstddef>

namespace halcyon {

struct SearchLimits {
    int depth = 64;
    long long movetime = -1;
    long long wtime = -1, btime = -1, winc = 0, binc = 0;
    bool infinite = false;
    bool ponder = false;
};

void search_start(Position& pos, const SearchLimits& limits);
void search_prepare();
void search_stop();
void search_ponderhit();   // GUI played the predicted move: switch ponder search to a timed one
void tt_resize(size_t mb);
void tt_clear();
void set_num_threads(int n);
int  get_num_threads();

void set_multi_pv(int n);
void set_move_overhead(int ms);
void set_go_extras(int movestogo, long long nodes);
void set_search_quiet(bool quiet);
void set_show_wdl(bool on);
void set_skill_level(int level);
void set_limit_strength(bool on);
void set_uci_elo(int elo);
void set_nodes_time(int n);
void set_syzygy_probe_depth(int d);
void set_syzygy_probe_limit(int n);
void set_syzygy_50_move_rule(bool on);
void set_halcyon_spirit(int level);   // 0 Calm, 1 Balanced, 2 Sharp
void set_clarity_bias(int cp);        // prefer simplifying when ahead by >= cp
void set_velocity_mode(bool on);      // leaner LMR / higher NPS

} // namespace halcyon
