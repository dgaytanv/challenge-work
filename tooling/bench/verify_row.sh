#!/bin/bash
# verify_row.sh <tag> [expected_train_data]
# Answers the only question that matters before a row goes in a table:
# did this bench ACTUALLY produce a JSON, and does that JSON say what the row claims?
# Three states, never two: NEVER-RAN / INCOMPLETE / OK.
tag="${1:?usage: verify_row.sh <tag> [train_data]}"
want_td="${2:-}"
shopt -s nullglob
files=(/home/jovyan/hackathon-shared/runs/${tag}_*.json)
if [ ${#files[@]} -eq 0 ]; then
  echo "NEVER-RAN  $tag  (no runs/${tag}_*.json; the wrapper's rc says nothing)"
  exit 2
fi
f="${files[-1]}"
python3 - "$f" "$tag" "$want_td" <<'PY'
import json,sys
f,tag,want=sys.argv[1],sys.argv[2],sys.argv[3]
try: d=json.load(open(f))
except Exception as e:
    print(f"INCOMPLETE {tag}  (JSON unreadable: {e})"); sys.exit(2)
missing=[k for k in ("mean_area","tag","train_data","seed","data") if k not in d]
if missing:
    print(f"INCOMPLETE {tag}  (missing fields: {','.join(missing)})  {f}"); sys.exit(2)
if d["tag"]!=tag:
    print(f"INCOMPLETE {tag}  (JSON tag is {d['tag']})  {f}"); sys.exit(2)
if want and d.get("train_data")!=want:
    print(f"INCOMPLETE {tag}  (train_data {d.get('train_data')} != expected {want})  {f}"); sys.exit(2)
print(f"OK  {tag}  mean_area {d['mean_area']:.6f} +- {d.get('mean_area_std',float('nan')):.6f}  "
      f"clean {d.get('auc_clean_full',float('nan')):.4f}  seed {d.get('seed')}  R={d.get('probe_repeats')}  {f.split('/')[-1]}")
PY
