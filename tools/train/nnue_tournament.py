"""
Round-robin tournament between several Halcyon .hnn networks, to objectively
determine which one actually plays strongest -- static eval alone (what
nnue_compare printed) cannot answer that, since two networks can disagree by
hundreds of centipawns on a position and there's no way to tell which one is
"righter" without seeing how that translates into actual game results.

Each pairing plays a fixed number of games alternating colours, from a small
set of varied opening positions (not always the start position, so the
match isn't just one deterministic game repeated). Uses the SAME engine
binary for both sides (process A has network X loaded, process B has
network Y loaded) -- i.e. this isolates the network's contribution, not an
engine-vs-different-engine comparison.

Usage:
    python3 nnue_tournament.py --engine ./halcyon \
        --nets net_v1=v1.hnn net_v2=v2.hnn net_colab=colab.hnn net_power=power.hnn \
        --games-per-pair 4 --depth 10 --out tournament_results.csv
"""
import argparse
import itertools
import subprocess
import sys

import chess


class EngineProcess:
    def __init__(self, path, net_file, depth):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1
        )
        self.depth = depth
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name NNUEFile value {net_file}")
        self._send("setoption name UseNNUE value true")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token):
        lines = []
        for line in self.proc.stdout:
            lines.append(line.rstrip("\n"))
            if token in line:
                break
        return lines

    def best_move(self, fen):
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else None
        return None

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# A handful of varied, roughly balanced starting points so a 4-game match
# isn't the exact same game mirrored twice with colours swapped.
OPENING_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",            # 1.e4 e5
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",              # 1.d4
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",        # Petrov-ish
]


def play_one_game(engine_white, engine_black, start_fen, max_plies=120):
    board = chess.Board(start_fen)
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        mover = engine_white if board.turn == chess.WHITE else engine_black
        uci = mover.best_move(board.fen())
        if uci is None or uci == "0000":
            break
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            board.push(move)
        except ValueError:
            break
        ply += 1
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        result = "1/2-1/2"  # hit max_plies without a decision -- treat as a draw
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--nets", nargs="+", required=True,
                     help="name=path.hnn pairs, e.g. net_v1=v1.hnn net_v2=v2.hnn")
    ap.add_argument("--games-per-pair", type=int, default=4,
                     help="total games per pairing, split evenly across opening FENs and colours")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--out", default="tournament_results.csv")
    args = ap.parse_args()

    nets = {}
    for spec in args.nets:
        name, path = spec.split("=", 1)
        nets[name] = path
    names = list(nets.keys())
    print(f"networks: {names}", file=sys.stderr)

    score = {n: 0.0 for n in names}
    games_played = {n: 0 for n in names}
    rows = []

    for a, b in itertools.combinations(names, 2):
        print(f"\n=== {a} vs {b} ===", file=sys.stderr)
        eng_a = EngineProcess(args.engine, nets[a], args.depth)
        eng_b = EngineProcess(args.engine, nets[b], args.depth)

        for g in range(args.games_per_pair):
            fen = OPENING_FENS[g % len(OPENING_FENS)]
            a_is_white = (g % 2 == 0)
            white, black = (eng_a, eng_b) if a_is_white else (eng_b, eng_a)
            white_name, black_name = (a, b) if a_is_white else (b, a)

            result = play_one_game(white, black, fen)
            if result == "1-0":
                score[white_name] += 1.0
            elif result == "0-1":
                score[black_name] += 1.0
            else:
                score[white_name] += 0.5
                score[black_name] += 0.5
            games_played[white_name] += 1
            games_played[black_name] += 1

            print(f"  game {g+1}: {white_name}(W) vs {black_name}(B) -> {result}", file=sys.stderr)
            rows.append({"white": white_name, "black": black_name, "result": result, "start_fen": fen})

        eng_a.close()
        eng_b.close()

    print("\n=== FINAL STANDINGS ===")
    ranked = sorted(names, key=lambda n: score[n] / max(1, games_played[n]), reverse=True)
    for n in ranked:
        gp = games_played[n]
        pct = 100.0 * score[n] / gp if gp else 0.0
        print(f"  {n}: {score[n]:.1f} / {gp} games ({pct:.1f}%)")

    with open(args.out, "w") as f:
        f.write("white,black,result,start_fen\n")
        for r in rows:
            f.write(f"{r['white']},{r['black']},{r['result']},{r['start_fen']}\n")
    print(f"\nGame-by-game results written to {args.out}")


if __name__ == "__main__":
    main()
