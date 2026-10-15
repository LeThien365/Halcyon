"""
Streams a Lichess .pgn.zst database and samples positions from real games'
actual moves, writing one FEN per line to a plain text file. This is stage 1
of a two-stage pipeline:

  stage 1 (this script):  .pgn.zst  -> positions.txt   (just FEN strings)
  stage 2 (score_positions.py): positions.txt + engine -> training.csv  (fen;cp;result)

Splitting it this way means the slow part (stage 2, which runs the engine on
every position) can be stopped/resumed/parallelized independently of PGN
parsing, and you can inspect/filter positions.txt by hand before spending any
engine time on it.

Sampling strategy (deliberately simple and transparent, not hidden inside
python-chess's own iteration):
  - skip the first `--skip-plies` plies of every game (opening book moves are
    highly repetitive and contribute little as training signal)
  - take one position every `--stride` plies after that
  - skip games shorter than `--min-plies` total (too short to be informative,
    often early resignations/aborts)
  - skip positions where the side to move is already in checkmate/stalemate
    (nothing to learn from a terminal position with no eval)

No Elo filtering is applied (every game in the file is used), per the
explicit choice to take the full rating spectrum rather than a curated
subset -- this script does NOT filter on WhiteElo/BlackElo at all.

Accepts either a compressed `.pgn.zst` file OR a plain, already-decompressed
`.pgn` file -- detected automatically from the file extension, so you can
point --pgn at whichever one you actually have (e.g. if you unzipped the
.zst yourself before uploading) without needing to re-compress anything.

Usage:
    python3 pgn_sample_positions.py --pgn games.pgn.zst --out positions.txt \
        --target 10000
    # or, with an already-decompressed file:
    python3 pgn_sample_positions.py --pgn games.pgn --out positions.txt \
        --target 10000
"""
import argparse
import io
import sys

import chess
import chess.pgn


def iter_games_from_pgn_source(path):
    """Yields python-chess Game objects streamed from `path`, which may be
    either a zstandard-compressed .pgn.zst or a plain, already-decompressed
    .pgn file -- picked automatically by file extension. Either way, nothing
    is ever written to a temporary decompressed file on disk."""
    if path.lower().endswith(".zst"):
        import zstandard
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        with open(path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                while True:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    yield game
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as text_stream:
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                yield game


def sample_positions_from_game(game, skip_plies, stride, min_plies):
    """Yields FEN strings sampled from one game's actual move sequence."""
    moves = list(game.mainline_moves())
    if len(moves) < min_plies:
        return

    board = game.board()
    for ply, move in enumerate(moves):
        board.push(move)
        if ply < skip_plies:
            continue
        if (ply - skip_plies) % stride != 0:
            continue
        if board.is_checkmate() or board.is_stalemate():
            continue
        yield board.fen()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help="path to a Lichess .pgn.zst file, OR an already-decompressed plain .pgn file")
    ap.add_argument("--out", required=True, help="output path for sampled FENs (one per line)")
    ap.add_argument("--target", type=int, default=10000, help="stop once this many positions are sampled")
    ap.add_argument("--skip-plies", type=int, default=10, help="skip this many opening plies per game")
    ap.add_argument("--stride", type=int, default=8, help="take one position every N plies after the skip")
    ap.add_argument("--min-plies", type=int, default=20, help="skip games shorter than this (too short to be useful)")
    ap.add_argument("--max-games", type=int, default=0, help="stop after reading this many games (0 = no limit besides --target)")
    args = ap.parse_args()

    written = 0
    games_read = 0
    with open(args.out, "w") as out_f:
        for game in iter_games_from_pgn_source(args.pgn):
            games_read += 1
            for fen in sample_positions_from_game(game, args.skip_plies, args.stride, args.min_plies):
                out_f.write(fen + "\n")
                written += 1
                if written >= args.target:
                    break
            if written >= args.target:
                break
            if args.max_games and games_read >= args.max_games:
                break
            if games_read % 500 == 0:
                print(f"...{games_read} games read, {written} positions sampled so far", file=sys.stderr)

    print(f"Done: {games_read} games read, {written} positions written to {args.out}")


if __name__ == "__main__":
    main()
