"""Table of the two-view diagnostics per epoch, parsed from a training log.

Usage: python tools_cos_table.py logs/training_<ts>.log
Acceptance item for WP-C: cosine(z_c, z_d) on validation across epochs.
"""
import re
import sys

PAT = re.compile(
    r"Epoch (\d+)/\d+ .*?Train: Loss ([\d.]+).*?Val: Loss ([\d.]+),.*?AUC ([\d.nan]+)"
    r".*?cons ([\d.]+), mse ([\d.]+), inst ([\d.]+), cos_tr ([\d.]+), acc_deg_tr ([\d.]+)"
    r" \| cos_val ([\d.]+), acc_deg_val ([\d.]+)"
)

def main(path):
    rows = []
    for line in open(path):
        m = PAT.search(line)
        if m:
            rows.append(m.groups())
    if not rows:
        print(f"no two-view epoch lines found in {path}")
        return
    hdr = ("ep", "train_loss", "val_loss", "val_auc", "cons", "cons_mse",
           "inst", "cos_tr", "acc_deg_tr", "cos_val", "acc_deg_val")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    best = min(rows, key=lambda r: float(r[2]))
    print(f"\nbest val loss at epoch {best[0]}: val_loss={best[2]} val_auc={best[3]} "
          f"cos_val={best[9]} acc_deg_val={best[10]}")
    print(f"cos_val over training: {rows[0][9]} -> {rows[-1][9]}")

if __name__ == "__main__":
    main(sys.argv[1])
