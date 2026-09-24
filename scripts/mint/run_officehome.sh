#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_officehome.sh <A|C|P|R> [seed]}"
SEED="${2:-100}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${SOURCE}" in
  A) source=art; targets=(clipart product real_world); tag=CPR ;;
  C) source=clipart; targets=(art product real_world); tag=APR ;;
  P) source=product; targets=(art clipart real_world); tag=ACR ;;
  R) source=real_world; targets=(art clipart product); tag=ACP ;;
  *) echo "Unknown Office-Home source: ${SOURCE}" >&2; exit 2 ;;
esac

score_file="${ROOT}/results/officehome_mtda/difficulty_${SOURCE}_seed${SEED}.json"
order_text="$("${PYTHON_BIN}" scripts/mint/order_from_score.py "${score_file}" \
  --source "${source}" --seed "${SEED}" --targets "${targets[@]}")"
mapfile -t order <<< "${order_text}"
order_cfg="['${order[0]}','${order[1]}','${order[2]}']"
run_dir="${ROOT}/output/officehome_mtda/mint/${SOURCE}2${tag}/seed${SEED}"
opts=(TRAINER.MAPLE_MTDA.CURRICULUM.DOMAIN_ORDER "${order_cfg}")

"${PYTHON_BIN}" scripts/experiment_guard.py \
  --output-dir "${run_dir}" --method-tag mint --source "${SOURCE}" \
  --targets "${tag:0:1}" "${tag:1:1}" "${tag:2:1}" --seed "${SEED}" \
  --trainer CurriculumContinuousSharedProjMaPLeMTDA \
  --trainer-config config/trainers/mint.yaml \
  --dataset-config config/datasets/office_home_mtda.yaml \
  --data "${DATA_ROOT}" --score-manifest "${score_file}" \
  --effective-opts "${opts[*]}"
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed matching MINT run: ${run_dir}"
  exit 0
fi

"${PYTHON_BIN}" train.py --root "${DATA_ROOT}" --seed "${SEED}" \
  --trainer CurriculumContinuousSharedProjMaPLeMTDA \
  --dataset-config-file config/datasets/office_home_mtda.yaml \
  --config-file config/trainers/mint.yaml \
  --source-domains "${source}" --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" "${opts[@]}"
"${PYTHON_BIN}" scripts/mint/export_metrics.py --run-dir "${run_dir}" \
  --dataset Office-Home --method MINT --source "${SOURCE}" \
  --seed "${SEED}" --targets "${targets[@]}"
