"""
Summarizes the CSV produced by evaluate_openings.py into human-readable
rankings: which openings score best/worst for White, and separately for
Black, by Halcyon's own search evaluation.

Usage:
    python3 rank_openings.py --csv openings_eval.csv --top 20
"""
import argparse
import csv


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["cp_white"] = int(r["cp_white"])
            r["cp_black"] = int(r["cp_black"])
            rows.append(r)
    return rows


def print_ranking(rows, key, label, top_n):
    ranked = sorted(rows, key=lambda r: r[key], reverse=True)
    print(f"\n=== {label}: TOP {top_n} (mạnh nhất) ===")
    for r in ranked[:top_n]:
        print(f"  {r[key]:+6d}cp  [{r['eco']}] {r['name']}  ({r['pgn']})")
    print(f"\n=== {label}: BOTTOM {top_n} (yếu nhất) ===")
    for r in ranked[-top_n:][::-1]:
        print(f"  {r[key]:+6d}cp  [{r['eco']}] {r['name']}  ({r['pgn']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="output CSV from evaluate_openings.py")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-csv", default=None, help="optional: write full sorted rankings to this CSV (two sheets' worth: by white then by black, concatenated with a separator column)")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    print(f"loaded {len(rows)} evaluated openings from {args.csv}")

    print_ranking(rows, "cp_white", "Theo góc nhìn TRẮNG", args.top)
    print_ranking(rows, "cp_black", "Theo góc nhìn ĐEN", args.top)

    avg_white = sum(r["cp_white"] for r in rows) / len(rows)
    print(f"\nĐiểm trung bình toàn bộ khai cuộc (góc nhìn Trắng): {avg_white:+.1f}cp")

    if args.out_csv:
        by_white = sorted(rows, key=lambda r: r["cp_white"], reverse=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank_by_white", "eco", "name", "pgn", "cp_white", "cp_black"])
            for i, r in enumerate(by_white, 1):
                writer.writerow([i, r["eco"], r["name"], r["pgn"], r["cp_white"], r["cp_black"]])
        print(f"\nĐã ghi bảng xếp hạng đầy đủ vào {args.out_csv}")


if __name__ == "__main__":
    main()
