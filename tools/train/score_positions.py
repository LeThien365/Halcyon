"""
Stage 2 of the Lichess-PGN training pipeline: reads the FEN list produced by
pgn_sample_positions.py, scores each position with the Halcyon engine itself
(NOT with whatever the game's eventual result was -- each position gets its
own fresh centipawn score from Halcyon's own search), and writes a CSV in
exactly the `fen;cp;result` format nnue_format.py / train_nnue_numpy.py /
train_nnue.py already expect.

"result" here is still needed by the trainer's CSV format, but since these
positions come from arbitrary points in real games (not from a self-play
process that tracks its own game outcome), there is no single well-defined
"game result" to attach per-position that means the same thing self-play's
column means. Rather than invent a fake one, this script fills `result` with
the side-to-move-relative sign of Halcyon's own score (clamped into
{0.0, 0.5, 1.0}), i.e. it derives both training columns from the SAME engine
evaluation at that position -- there is deliberately no second, different
signal being mixed in. If you want the trainer to use *only* the centipawn
column (ignoring `result` during training), that's a one-line change in
train_nnue.py / train_nnue_numpy.py, not something this script needs to
decide on your behalf.

Resumable: positions already present in --out are skipped on a re-run (keyed
by exact FEN string), so a long scoring run can be stopped and restarted.

Usage:
    python3 score_positions.py --engine ../halcyon --positions positions.txt \
        --out training.csv --depth 10
"""
import argparse
import subprocess
import sys


class EngineProcess:
    """Same minimal UCI wrapper as selfplay_gen.py's, reused as-is rather
    than reimplemented, so both scripts talk to the engine identically."""

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

    def eval_fen(self, fen, depth):
        """Returns (cp_from_side_to_move_perspective, is_mate_score)."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        last_cp = None
        last_mate = None
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("info") and " score cp " in line:
                try:
                    last_cp = int(line.split(" score cp ")[1].split(" ")[0])
                    last_mate = None
                except (IndexError, ValueError):
                    pass
            elif line.startswith("info") and " score mate " in line:
                try:
                    mate_in = int(line.split(" score mate ")[1].split(" ")[0])
                    last_mate = mate_in
                    last_cp = None
                except (IndexError, ValueError):
                    pass
            if line.startswith("bestmove"):
                break
        if last_mate is not None:
            cp = 30000 - abs(last_mate) if last_mate > 0 else -(30000 - abs(last_mate))
            return cp, True
        return (last_cp if last_cp is not None else 0), False

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def load_already_scored(out_path):
    seen = set()
    try:
        with open(out_path) as f:
            next(f, None)  # header
            for line in f:
                parts = line.rstrip("\n").split(";")
                if parts:
                    seen.add(parts[0])
    except FileNotFoundError:
        pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to the compiled Halcyon binary")
    ap.add_argument("--positions", required=True, help="FEN list from pgn_sample_positions.py")
    ap.add_argument("--out", required=True, help="output CSV (fen;cp;result)")
    ap.add_argument("--depth", type=int, default=10, help="search depth per position")
    ap.add_argument("--cp-clip", type=int, default=1000, help="clip reported cp to +/- this before writing (matches train_nnue*.py's own clipping, applied here too so inspected CSVs already reflect what training will see)")
    args = ap.parse_args()

    already = load_already_scored(args.out)
    if already:
        print(f"resuming: {len(already)} positions already scored in {args.out}", file=sys.stderr)

    engine = EngineProcess(args.engine)
    write_header = not already
    scored = 0
    skipped = 0

    with open(args.positions) as pf, open(args.out, "a") as out_f:
        if write_header:
            out_f.write("fen;cp;result\n")
        for line in pf:
            fen = line.strip()
            if not fen:
                continue
            if fen in already:
                skipped += 1
                continue

            cp, is_mate = engine.eval_fen(fen, args.depth)
            cp_clipped = max(-args.cp_clip, min(args.cp_clip, cp))

            # Derive `result` from the SAME score (see module docstring) --
            # side-to-move-relative cp > 0 => that side is doing better.
            if cp_clipped > 50:
                result = 1.0
            elif cp_clipped < -50:
                result = 0.0
            else:
                result = 0.5

            out_f.write(f"{fen};{cp_clipped};{result}\n")
            out_f.flush()
            scored += 1
            if scored % 200 == 0:
                print(f"...{scored} scored, {skipped} skipped (already done)", file=sys.stderr)

    engine.close()
    print(f"Done: {scored} positions scored, {skipped} skipped, written to {args.out}")


if __name__ == "__main__":
    main()
