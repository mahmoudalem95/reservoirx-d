#!/usr/bin/env bash
# Worked examples against a local ReservoirX-D instance.
#   uvicorn app.main:app --port 8080
set -euo pipefail
BASE="${RXD_BASE:-http://localhost:8080}/v1/reservoirx-d"

echo "== 1. Generate a directed degree-regular reservoir =="
MATRIX_ID=$(curl -sS -X POST "$BASE/generate" -H 'Content-Type: application/json' -d '{
  "node_count": 200, "mean_degree": 3, "wiring_scheme": "degree_regular",
  "spectral_radius": 0.9, "weight_distribution": "uniform", "seed": 42
}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["primary"]["matrix_id"])')
echo "matrix_id=$MATRIX_ID"

echo
echo "== 2. Fractional degree is rejected, with the alternative named =="
curl -sS -X POST "$BASE/generate" -H 'Content-Type: application/json' \
  -d '{"node_count": 200, "mean_degree": 2.7, "wiring_scheme": "degree_regular"}' \
  | python3 -m json.tool | head -20

echo
echo "== 3. Same fractional degree via near_regular =="
curl -sS -X POST "$BASE/generate" -H 'Content-Type: application/json' \
  -d '{"node_count": 200, "mean_degree": 2.7, "wiring_scheme": "near_regular", "seed": 42}' \
  | python3 -c 'import sys,json; p=json.load(sys.stdin)["primary"]; print("edges", p["n_edges"], "degree_cv %.4f" % p["metrics"]["degree_cv"])'

echo
echo "== 4. Diagnose against both null models =="
curl -sS -X POST "$BASE/diagnose" -H 'Content-Type: application/json' \
  -d "{\"matrix_id\": \"$MATRIX_ID\", \"iterations\": 50}" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("reciprocity %.4f" % d["raw_metrics"]["reciprocity"]); print(d["directedness"]["assessment"])'

echo
echo "== 5. Evaluate memory capacity =="
curl -sS -X POST "$BASE/evaluate" -H 'Content-Type: application/json' \
  -d "{\"matrix_id\": \"$MATRIX_ID\", \"task\": \"memory_capacity\", \"seed\": 0,
       \"evaluation_params\": {\"sequence_length\": 3000, \"max_lag\": 60}}" \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print("score %.3f" % r["score"]); print("warnings:", r["warnings"])'

echo
echo "== 6. Compare regular vs skewed, two tasks, with the symmetrisation control =="
curl -sS -X POST "$BASE/compare" -H 'Content-Type: application/json' -d '{
  "node_count": 200, "mean_degree": 3, "iterations": 8,
  "tasks": ["memory_capacity", "delay_xor"],
  "include_symmetrization_control": true
}' | python3 -c '
import sys, json
b = json.load(sys.stdin)
for name, t in b["per_task"].items():
    s = t["statistics"]
    d, p1, ph = s["effect_size"], s["p_value"], s["p_holm"]
    print("%-16s d=%+.2f p=%.4f p_holm=%.4f" % (name, d, p1, ph))
print(b["summary"]["verdict"])'

echo
echo "== 7. Optimize width, with a lag budget that actually discriminates =="
curl -sS -X POST "$BASE/optimize" -H 'Content-Type: application/json' -d '{
  "base_nodes": 200, "mean_degree": 3, "selection_seeds": 8, "confirmation_seeds": 4,
  "evaluation_params": {"sequence_length": 4000, "max_lag": 120}
}' | python3 -c '
import sys, json
b = json.load(sys.stdin)
sel = b["selection"]
print("optimal_ratio", sel["optimal_ratio"], "| saving %.1f%%" % sel["measured_connection_saving_percent"],
      "| saturated", sel["metric_saturated"], "| confirmed", b["confirmation"]["holds"])'

echo
echo "== 8. Evidence behind the service =="
curl -sS "$BASE/claims" | python3 -c '
import sys, json
for c in json.load(sys.stdin)["claims"]:
    print("%-22s %s" % (c["status"], c["id"]))'
