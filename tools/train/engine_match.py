"""
A/B match between two Halcyon binaries (e.g. old vs new evaluation) at fixed
depth. Openings come from the ECO database (varied 6-10 ply lines); every
opening is played twice with colours swapped so neither side gets the
better colour. Games are adjudicated when clearly decided / dead drawn.

    python3 engine_match.py --a ./halcyon_new --b ./halcyon_old --eco-dir eco_data \
        --pairs 100 --depth 5 --opt UseNNUE=false
"""
import argparse, csv, glob, math, os, random, subprocess, sys, time
import chess

class Eng:
    def __init__(self, path, opts):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci"); self.until("uciok")
        for k, v in opts: self.send(f"setoption name {k} value {v}")
        self.send("isready"); self.until("readyok")
    def send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()
    def until(self, tok):
        for line in self.p.stdout:
            if tok in line: return
    def newgame(self): self.send("ucinewgame"); self.send("isready"); self.until("readyok")
    def go(self, moves, depth):
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self.send(f"go depth {depth}")
        score = None; best = None
        for line in self.p.stdout:
            if line.startswith("info") and " score " in line:
                t = line.split(); i = t.index("score")
                if t[i+1] == "cp": score = int(t[i+2])
                elif t[i+1] == "mate": score = 30000 if int(t[i+2]) > 0 else -30000
            elif line.startswith("bestmove"):
                best = line.split()[1]; break
        return best, score
    def close(self):
        try: self.send("quit"); self.p.wait(timeout=3)
        except Exception: self.p.kill()

def load_openings(eco_dir, n, seed):
    rows = []
    for path in sorted(glob.glob(os.path.join(eco_dir, "*.tsv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"): rows.append(r["pgn"])
    rnd = random.Random(seed); rnd.shuffle(rows)
    out = []
    for pgn in rows:
        b = chess.Board(); ucis = []
        try:
            for tok in pgn.replace(".", ". ").split():
                if tok.endswith(".") or not tok: continue
                m = b.parse_san(tok); ucis.append(m.uci()); b.push(m)
        except Exception: continue
        if len(ucis) < 6: continue
        ucis = ucis[:rnd.randint(6, 10)]
        if tuple(ucis) not in [tuple(o) for o in out]: out.append(ucis)
        if len(out) >= n: break
    return out

def play(white, black, opening, depth, max_plies=220):
    board = chess.Board(); moves = []
    for u in opening: board.push_uci(u); moves.append(u)
    white.newgame(); black.newgame()
    wp = []
    while True:
        if board.is_game_over(claim_draw=True): return board.result(claim_draw=True)
        if len(moves) >= max_plies: return "1/2-1/2"
        eng = white if board.turn == chess.WHITE else black
        mv, sc = eng.go(moves, depth)
        if not mv or mv == "0000": return "1/2-1/2"
        try:
            m = chess.Move.from_uci(mv)
            if m not in board.legal_moves: return "0-1" if board.turn == chess.WHITE else "1-0"
        except ValueError: return "1/2-1/2"
        if sc is not None: wp.append(sc if board.turn == chess.WHITE else -sc)
        board.push(m); moves.append(mv)
        if len(wp) >= 6 and all(abs(x) >= 900 and (x > 0) == (wp[-1] > 0) for x in wp[-6:]):
            return "1-0" if wp[-1] > 0 else "0-1"
        if len(moves) >= 100 and len(wp) >= 20 and all(abs(x) <= 12 for x in wp[-20:]): return "1/2-1/2"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--eco-dir", required=True); ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--depth", type=int, default=5); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", action="append", default=[], help="Name=Value applied to both engines")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    opts = [tuple(o.split("=", 1)) for o in a.opt] + [("Hash", "16")]
    A, B = Eng(a.a, opts), Eng(a.b, opts)
    ops = load_openings(a.eco_dir, a.pairs, a.seed)
    w = d = l = 0; t0 = time.time()
    for i, op in enumerate(ops, 1):
        for aw in (True, False):
            r = play(A if aw else B, B if aw else A, op, a.depth)
            if r == "1/2-1/2": d += 1
            elif (r == "1-0") == aw: w += 1
            else: l += 1
        n = w + d + l; s = (w + 0.5 * d) / n
        line = f"pair {i}/{len(ops)}  A(new): +{w} ={d} -{l}  score {100*s:.1f}%  [{time.time()-t0:.0f}s]"
        print(line, flush=True)
        if a.log: open(a.log, "w").write(line + "\n")
    n = w + d + l; s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    se = math.sqrt(var / n)
    elo = lambda x: -400 * math.log10(1 / min(max(x, 1e-6), 1 - 1e-6) - 1)
    print(f"FINAL  A vs B: +{w} ={d} -{l} over {n} games  score {100*s:.1f}%  Elo {elo(s):+.0f}  (95% CI {elo(s-1.96*se):+.0f} .. {elo(s+1.96*se):+.0f})")
    A.close(); B.close()

if __name__ == "__main__": main()
