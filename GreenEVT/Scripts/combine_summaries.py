"""
Merge summary.csv files from several charge_eval.py runs (e.g. one sim job
per model) into one table. Rows are keyed by policy; a policy that appears
in more than one run (base/risk, which every job can include) is taken from
the first run listed.

    python combine_summaries.py $WORK/charge_eval_out/seed_42_* --out combined.csv
"""
import argparse
import csv
import os

from charge_eval import print_table, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="charge_eval.py output folders (or summary.csv files)")
    p.add_argument("--out", default="combined_summary.csv")
    args = p.parse_args()

    rows = {}
    for run in args.runs:
        path = os.path.join(run, "summary.csv") if os.path.isdir(run) else run
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.setdefault(row["policy"], row)

    rows = list(rows.values())
    write_csv(args.out, rows)
    print_table(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
