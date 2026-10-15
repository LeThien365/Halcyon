"""
Self-play data generator for NNUE training data.

Plays games of Halcyon against itself (random opening moves for diversity,
then engine search for the rest of the game) and records one training sample
per position: the FEN, the engine's search score in centipawns (from the side
to move's perspective), and the eventual game result.

Usage:
    python3 selfplay_gen.py --engine ../halcyon --games 20 --depth 6 --out data.csv

This is a small reference implementation to prove the pipeline works end to
end. For a network that actually plays well you will want FAR more data
(hobby NNUE nets are typically trained on tens of millions of positions) and
probably a higher search depth per move — run this for much longer, and
ideally in parallel, on your own machine.
"""
import argparse
import random
import subprocess
import sys

import chess


class EngineProcess:
    def __init__(self, path):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name UseNNUE value false")
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

    def search(self, fen, depth):
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        last_score = 0
        best_move = None
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("info") and " score cp " in line:
                try:
                    last_score = int(line.split(" score cp ")[1].split(" ")[0])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("info") and " score mate " in line:
                mate_in = int(line.split(" score mate ")[1].split(" ")[0])
                last_score = 30000 - abs(mate_in) if mate_in > 0 else -(30000 - abs(mate_in))
            if line.startswith("bestmove"):
                parts = line.split()
                best_move = parts[1] if len(parts) > 1 else None
                break
        return best_move, last_score

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_one_game(engine, random_plies, search_depth, max_plies):
    board = chess.Board()
    samples = []  # (fen, cp_score_side_to_move) collected as we go

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        fen = board.fen()

        if ply < random_plies:
            move = random.choice(list(board.legal_moves))
            samples.append((fen, None))  # no engine score for pure random-opening plies
        else:
            best_uci, score_cp = engine.search(fen, search_depth)
            if best_uci is None:
                break
            move = chess.Move.from_uci(best_uci)
            samples.append((fen, score_cp))

        board.push(move)
        ply += 1

    result_str = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    if result_str == "1-0":
        result = 1.0
    elif result_str == "0-1":
        result = 0.0
    else:
        result = 0.5

    return samples, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="../halcyon", help="path to the compiled Halcyon binary")
    ap.add_argument("--games", type=int, default=2000000)
    ap.add_argument("--depth", type=int, default=10, help="search depth per move during data generation")
    ap.add_argument("--random-plies", type=int, default=6, help="random opening plies for game diversity")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--out", default="selfplay_data.csv")
    args = ap.parse_args()

    engine = EngineProcess(args.engine)
    import subprocess as _sp
    written = 0
    with open(args.out, "w") as f:
        f.write("fen;cp;result\n")
        for g in range(args.games):
            samples, result = play_one_game(engine, args.random_plies, args.depth, args.max_plies)
            for fen, cp in samples:
                if cp is None:
                    continue  # skip the random-opening plies themselves (no meaningful engine score)
                f.write(f"{fen};{cp};{result}\n")
                written += 1
            print(f"game {g+1}/{args.games} done, result={result}, samples so far={written}", file=sys.stderr)
    engine.close()
    print(f"Wrote {written} samples to {args.out}")


if __name__ == "__main__":
    main()
