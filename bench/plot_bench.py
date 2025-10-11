#!/usr/bin/env python3
import argparse, os, json, glob
import matplotlib.pyplot as plt

def load_summaries(pattern):
    paths = sorted(glob.glob(pattern))
    rows = []
    for p in paths:
        try:
            with open(p, "r") as f:
                data = json.load(f)
            s = data.get("summary", {})
            s["path"] = p
            rows.append(s)
        except Exception as e:
            print("Skip", p, ":", e)
    return rows

def bar_by_mode(rows, out_png):
    # group by (device, mode)
    labels, vals = [], []
    for r in rows:
        labels.append(f'{r.get("device")}-{r.get("mode")}')
        vals.append(r.get("ms_median", 0.0))
    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.ylabel("Median step time (ms)")
    plt.title("AdamW vs Tiger — Median Step Time")
    plt.tight_layout()
    plt.savefig(out_png, dpi=144)
    print("Saved plot:", out_png)

def line_loss(paths, out_png, max_points=200):
    plt.figure()
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
        loss = data.get("series", {}).get("loss", [])
        if not loss: 
            continue
        n = len(loss)
        step = max(1, n // max_points)
        xs = list(range(0, n, step))
        ys = [loss[i] for i in xs]
        plt.plot(xs, ys, label=os.path.basename(p))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=144)
    print("Saved plot:", out_png)

def main():
    ap = argparse.ArgumentParser("Plot bench results")
    ap.add_argument("--pattern", default="bench/results/compare-*.json")
    ap.add_argument("--out-dir", default="bench/plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_summaries(args.pattern)
    if not rows:
        print("No results found for", args.pattern)
        return
    bar_by_mode(rows, os.path.join(args.out_dir, "median_step_time.png"))
    line_loss(sorted([r["path"] for r in rows]), os.path.join(args.out_dir, "loss_curves.png"))

if __name__ == "__main__":
    main()
