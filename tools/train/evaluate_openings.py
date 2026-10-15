"""
Evaluates every opening line in Lichess's public ECO database (the
chess-openings project: eco_a.tsv..eco_e.tsv, columns eco/name/pgn) using
Halcyon's own search, and writes one row per opening with its engine score.

For each opening:
  1. Replay its PGN move sequence from the starting position to get the FEN.
  2. Run `go depth N` on that FEN.
  3. Record the final centipawn score (mate scores are mapped to a large
     capped centipawn value, same convention as score_positions.py) from the
     side-to-move's own perspective, AND translated to absolute White/Black
     perspective so openings can be ranked for either side directly.

Output CSV columns: eco, name, pgn, side_to_move, depth, cp_side_to_move,
cp_white, cp_black, nodes, time_ms

Resumable: rows already present in --out (keyed by the exact `pgn` string,
which is unique per row in the source TSVs even when several rows share the
same opening `name`) are skipped on a re-run.

Usage:
    python3 evaluate_openings.py --engine ./halcyon --eco-dir . \
        --out openings_eval.csv --depth 17
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

import chess


class EngineProcess:
    """Same minimal UCI wrapper used by the other pipeline scripts."""

    def __init__(self, path, threads=1, hash_mb=64):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name UseNNUE value false")
        if threads > 1:
            self._send(f"setoption name Threads value {threads}")
        if hash_mb != 64:
            self._send(f"setoption name Hash value {hash_mb}")
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
        """Returns dict with cp (side-to-move perspective), is_mate, nodes, time_ms."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        last = {"cp": 0, "is_mate": False, "nodes": 0, "time_ms": 0}
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("info") and f" depth {depth} " in f" {line} ".replace(f"depth {depth}", f" depth {depth} ", 1):
                pass  # depth filter not strictly needed; we just keep the LAST info line below
            if line.startswith("info"):
                if " score cp " in line:
                    try:
                        last["cp"] = int(line.split(" score cp ")[1].split(" ")[0])
                        last["is_mate"] = False
                    except (IndexError, ValueError):
                        pass
                elif " score mate " in line:
                    try:
                        mate_in = int(line.split(" score mate ")[1].split(" ")[0])
                        last["cp"] = 29000 - abs(mate_in) if mate_in > 0 else -(29000 - abs(mate_in))
                        last["is_mate"] = True
                    except (IndexError, ValueError):
                        pass
                if " nodes " in line:
                    try:
                        last["nodes"] = int(line.split(" nodes ")[1].split(" ")[0])
                    except (IndexError, ValueError):
                        pass
                if " time " in line:
                    try:
                        last["time_ms"] = int(line.split(" time ")[1].split(" ")[0])
                    except (IndexError, ValueError):
                        pass
            if line.startswith("bestmove"):
                break
        return last

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def load_openings(eco_dir):
    """Reads eco_a.tsv..eco_e.tsv (or any *.tsv matching the eco/name/pgn
    header) from eco_dir, yielding (eco, name, pgn_string) tuples."""
    paths = sorted(glob.glob(os.path.join(eco_dir, "*.tsv")))
    if not paths:
        print(f"no .tsv files found in {eco_dir}", file=sys.stderr)
        return
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames != ["eco", "name", "pgn"]:
                print(f"skipping {path}: unexpected header {reader.fieldnames}", file=sys.stderr)
                continue
            for row in reader:
                yield row["eco"], row["name"], row["pgn"]


def pgn_to_board(pgn_string):
    """Replays a short SAN move sequence like '1. e4 e5 2. Nf3 Nc6' from the
    starting position and returns the resulting chess.Board, or None if any
    move in it is illegal (defensively handled -- the Lichess ECO data is
    expected to be clean, but we don't want one bad row to crash a
    multi-hour run)."""
    board = chess.Board()
    # Strip move numbers like "1." / "12..." and split on whitespace.
    tokens = pgn_string.replace(".", ". ").split()
    for tok in tokens:
        if tok.endswith(".") or tok == "":
            continue
        try:
            board.push_san(tok)
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
            return None
    return board


def load_already_done(out_path):
    seen = set()
    try:
        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen.add(row["pgn"])
    except FileNotFoundError:
        pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="path to the compiled Halcyon binary")
    ap.add_argument("--eco-dir", required=True, help="directory containing eco_a.tsv..eco_e.tsv")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--depth", type=int, default=17)
    ap.add_argument("--threads", type=int, default=1, help="engine Threads option (Colab CPU: 2 is a reasonable value)")
    ap.add_argument("--hash", type=int, default=64, help="engine Hash (MB)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many NEW rows (0 = no limit, process everything)")
    args = ap.parse_args()

    already = load_already_done(args.out)
    if already:
        print(f"resuming: {len(already)} openings already evaluated in {args.out}", file=sys.stderr)

    openings = list(load_openings(args.eco_dir))
    print(f"loaded {len(openings)} opening rows from {args.eco_dir}", file=sys.stderr)

    engine = EngineProcess(args.engine, threads=args.threads, hash_mb=args.hash)
    write_header = not already
    done = 0
    skipped_already = 0
    skipped_illegal = 0

    fieldnames = ["eco", "name", "pgn", "side_to_move", "depth",
                  "cp_side_to_move", "cp_white", "cp_black",
                  "is_mate", "nodes", "time_ms"]

    with open(args.out, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for eco, name, pgn in openings:
            if pgn in already:
                skipped_already += 1
                continue

            board = pgn_to_board(pgn)
            if board is None:
                skipped_illegal += 1
                print(f"skipping illegal/unparseable pgn: {eco} {name!r} {pgn!r}", file=sys.stderr)
                continue

            fen = board.fen()
            side_to_move = fen.split()[1]  # 'w' or 'b'

            if board.is_checkmate():
                # The opening line itself ends in mate (e.g. "Fool's Mate")
                # -- nothing to search; the side to move has just been
                # mated, so its score is the worst possible, not a neutral
                # 0 (which engine.eval_fen would otherwise report, since no
                # search actually runs on a position with no legal moves).
                cp_stm = -29000
                result = {"is_mate": True, "nodes": 0, "time_ms": 0}
            elif board.is_stalemate() or board.is_insufficient_material():
                cp_stm = 0
                result = {"is_mate": False, "nodes": 0, "time_ms": 0}
            else:
                result = engine.eval_fen(fen, args.depth)
                cp_stm = result["cp"]

            cp_white = cp_stm if side_to_move == "w" else -cp_stm
            cp_black = -cp_white

            writer.writerow({
                "eco": eco, "name": name, "pgn": pgn,
                "side_to_move": side_to_move, "depth": args.depth,
                "cp_side_to_move": cp_stm, "cp_white": cp_white, "cp_black": cp_black,
                "is_mate": result["is_mate"], "nodes": result["nodes"], "time_ms": result["time_ms"],
            })
            out_f.flush()
            done += 1
            if done % 50 == 0:
                print(f"...{done} evaluated, {skipped_already} already-done skipped, {skipped_illegal} illegal skipped", file=sys.stderr)
            if args.limit and done >= args.limit:
                print(f"stopping at --limit {args.limit}", file=sys.stderr)
                break

    engine.close()
    print(f"Done: {done} openings evaluated, {skipped_already} skipped (already done), "
          f"{skipped_illegal} skipped (illegal pgn), written to {args.out}")


if __name__ == "__main__":
    main()
