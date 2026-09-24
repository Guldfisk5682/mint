#!/usr/bin/env bash
set -Eeuo pipefail

# Seed-100 Office-Home replication. The first source-only probe may already be
# running; pass its PID as WAIT_FOR_A_PID to avoid launching it twice.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export DATA_ROOT="${DATA_ROOT:-/hy-tmp/mint/data}"
export PYTHON_BIN="${PYTHON_BIN:-/root/mint-venv/bin/python}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/root/mint-safe-results/officehome_seed100}"
SCORE_ROOT="${ROOT}/results/officehome_mtda"
OUTPUT_ROOT="${ROOT}/output/officehome_mtda"
CURRENT_RUN_DIR=""

backup_tree() {
  local source="$1" destination="$2"
  [[ -d "${source}" ]] || return 0
  mkdir -p "${destination}"
  cp -a "${source}/." "${destination}/"
  (cd "${destination}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum > SHA256SUMS)
}

backup_score() {
  local code="$1" tag="$2"
  backup_tree "${OUTPUT_ROOT}/source_probe/${code}2${tag}/seed100" \
    "${ARCHIVE_ROOT}/source_probe/${code}2${tag}/seed100"
  cp -a "${SCORE_ROOT}/difficulty_${code}_seed100.json" "${ARCHIVE_ROOT}/scores/"
  sha256sum "${ARCHIVE_ROOT}/scores/difficulty_${code}_seed100.json" \
    >> "${ARCHIVE_ROOT}/scores/SHA256SUMS"
}

on_error() {
  local status=$?
  trap - ERR
  printf 'Campaign failed: status=%s line=%s command=%s\n' \
    "${status}" "${BASH_LINENO[0]}" "${BASH_COMMAND}" | tee -a "${ARCHIVE_ROOT}/FAILURE.txt" >&2
  if [[ -n "${CURRENT_RUN_DIR}" ]]; then
    backup_tree "${CURRENT_RUN_DIR}" "${ARCHIVE_ROOT}/partial/$(basename "$(dirname "${CURRENT_RUN_DIR}")")/$(basename "${CURRENT_RUN_DIR}")" || true
  fi
  [[ ! -f "${SCORE_ROOT}/order_audit_seed100.json" ]] || \
    cp -a "${SCORE_ROOT}/order_audit_seed100.json" "${ARCHIVE_ROOT}/" || true
  exit "${status}"
}
trap on_error ERR

mkdir -p "${ARCHIVE_ROOT}/scores" "${SCORE_ROOT}"
[[ -d "${DATA_ROOT}/office_home" ]]
[[ -s /root/.cache/clip/ViT-B-16.pt ]]
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available()'
[[ -z "$(git status --porcelain)" ]]
git rev-parse HEAD > "${ARCHIVE_ROOT}/code_commit.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${ARCHIVE_ROOT}/started_utc.txt"

if [[ -n "${WAIT_FOR_A_PID:-}" ]]; then
  echo "Waiting for existing A probe PID ${WAIT_FOR_A_PID}"
  while [[ ! -s "${SCORE_ROOT}/difficulty_A_seed100.json" ]]; do
    kill -0 "${WAIT_FOR_A_PID}" 2>/dev/null || {
      echo "A probe exited without a difficulty manifest" >&2
      exit 1
    }
    sleep 60
  done
fi

for code in A C P R; do
  case "${code}" in
    A) tag=CPR ;; C) tag=APR ;; P) tag=ACR ;; R) tag=ACP ;;
  esac
  if [[ ! -s "${SCORE_ROOT}/difficulty_${code}_seed100.json" ]]; then
    echo "Starting source-only entropy probe ${code} at $(date -u +'%FT%TZ')"
    CURRENT_RUN_DIR="${OUTPUT_ROOT}/source_probe/${code}2${tag}/seed100"
    bash scripts/mint/run_officehome_probe.sh "${code}" 100
    CURRENT_RUN_DIR=""
  fi
  backup_score "${code}" "${tag}"
done

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path("results/officehome_mtda")
expected = {
    "A": ["clipart", "product", "real_world"],
    "C": ["art", "real_world", "product"],
    "P": ["clipart", "art", "real_world"],
    "R": ["clipart", "art", "product"],
}
audit = {"reference": "archived seed-100 full-MINT stage order", "sources": {}}
ok = True
for code, old_order in expected.items():
    score = json.loads((root / f"difficulty_{code}_seed100.json").read_text())
    actual = score["hard_to_easy"]
    matches = actual == old_order
    audit["sources"][code] = {
        "archived_full_order": old_order,
        "recomputed_hard_to_easy": actual,
        "recomputed_easy_to_hard": score["easy_to_hard"],
        "scores": score["scores"],
        "source_only_checkpoint_sha256": score["checkpoint_sha256"],
        "matches_archived_full": matches,
    }
    ok &= matches
audit["all_match"] = ok
output = root / "order_audit_seed100.json"
output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(json.dumps(audit, indent=2, sort_keys=True))
if not ok:
    raise SystemExit("Entropy order differs from archived full MINT; formal runs not launched")
PY
cp -a "${SCORE_ROOT}/order_audit_seed100.json" "${ARCHIVE_ROOT}/"

for code in A C P R; do
  case "${code}" in
    A) tag=CPR ;; C) tag=APR ;; P) tag=ACR ;; R) tag=ACP ;;
  esac
  echo "Starting full MINT hard-to-easy ${code} at $(date -u +'%FT%TZ')"
  CURRENT_RUN_DIR="${OUTPUT_ROOT}/mint/${code}2${tag}/seed100"
  bash scripts/mint/run_officehome.sh "${code}" 100
  backup_tree "${CURRENT_RUN_DIR}" "${ARCHIVE_ROOT}/mint/${code}2${tag}/seed100"
  CURRENT_RUN_DIR=""
done

for code in A C P R; do
  case "${code}" in
    A) tag=CPR ;; C) tag=APR ;; P) tag=ACR ;; R) tag=ACP ;;
  esac
  echo "Starting easy-to-hard ablation ${code} at $(date -u +'%FT%TZ')"
  CURRENT_RUN_DIR="${OUTPUT_ROOT}/officehome_ablation_e2h_seed100/${code}2${tag}/seed100"
  bash scripts/mint/run_officehome_ablation.sh e2h "${code}" 100
  backup_tree "${CURRENT_RUN_DIR}" "${ARCHIVE_ROOT}/e2h/${code}2${tag}/seed100"
  CURRENT_RUN_DIR=""
done

for code in A C P R; do
  case "${code}" in
    A) tag=CPR ;; C) tag=APR ;; P) tag=ACR ;; R) tag=ACP ;;
  esac
  echo "Starting no-student-soft ablation ${code} at $(date -u +'%FT%TZ')"
  CURRENT_RUN_DIR="${OUTPUT_ROOT}/officehome_ablation_hard_only_seed100/${code}2${tag}/seed100"
  bash scripts/mint/run_officehome_ablation.sh hard_only "${code}" 100
  backup_tree "${CURRENT_RUN_DIR}" "${ARCHIVE_ROOT}/hard_only/${code}2${tag}/seed100"
  CURRENT_RUN_DIR=""
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "${ARCHIVE_ROOT}/finished_utc.txt"
echo "Office-Home campaign completed at $(date -u +'%FT%TZ')"
