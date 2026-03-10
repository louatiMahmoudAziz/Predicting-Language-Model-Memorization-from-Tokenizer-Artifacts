"""Print a readable final summary of a pipeline run."""
import json
import os
import sys
import pandas as pd

run_id = sys.argv[1] if len(sys.argv) > 1 else "local_cpu"

# Pipeline manifest
pipe_path = f"results/{run_id}/{run_id}_pipeline.json"
m = json.load(open(pipe_path))
print("=== PIPELINE SUMMARY ===")
total = 0.0
for s in m["steps"]:
    e = s.get("elapsed_s", 0.0) or 0.0
    total += e
    art = os.path.basename(s.get("artifact", "") or "")
    print(f"  Step {s['step']:02d}  {s['name']:<22}  {s['status']:<8}  {e:5.1f}s  {art}")
print(f"  TOTAL: {total:.1f}s  ({total/60:.1f} min)")
print()

# Labels / ΔBPC
labels_path = f"labels/{run_id}_labels.parquet"
df = pd.read_parquet(labels_path)
all_valid = bool(df["valid_label"].all())
n_valid = int(df["valid_label"].sum())
all_positive = bool((df.loc[df["valid_label"], "delta_bpc"] > 0).all())
min_dbpc = float(df.loc[df["valid_label"], "delta_bpc"].min())
max_dbpc = float(df.loc[df["valid_label"], "delta_bpc"].max())
mean_dbpc = float(df.loc[df["valid_label"], "delta_bpc"].mean())

print("=== SPEC TEST 1 CRITERIA (repeated canaries show positive delta_bpc) ===")
print(f"  valid rows            : {n_valid} / {len(df)}")
print(f"  all delta_bpc > 0     : {all_positive}  (min={min_dbpc:.2f}, mean={mean_dbpc:.2f}, max={max_dbpc:.2f})")
print(f"  SPEC TEST 1           : {'PASS' if all_valid and all_positive else 'FAIL'}")
print()

# Eval
comp_path = f"results/{run_id}/eval/comparison.parquet"
comp = pd.read_parquet(comp_path)

reg = comp[comp["task"] == "regression"].copy()
if not reg.empty:
    cols = [c for c in ["task", "model_name", "feature_subset", "n_test", "auroc",
                        "reg_pearson_r", "reg_spearman_rho"] if c in reg.columns]
    print("=== REGRESSION METRICS (test split) ===")
    print(reg[cols].to_string(index=False))
    print()

clf = comp[comp["task"] == "classification"].copy()
if not clf.empty:
    cols = [c for c in ["task", "model_name", "feature_subset", "n_test", "n_pos", "auroc", "auprc"]
            if c in clf.columns]
    print("=== CLASSIFICATION METRICS (test split) ===")
    print(clf[cols].to_string(index=False))
