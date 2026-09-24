#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_office31_probe.sh <A|D|W> [seed]}"
SEED="${2:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${SOURCE}" in
  A) source_domain=amazon; targets=(dslr webcam); target_code=DW ;;
  D) source_domain=dslr; targets=(amazon webcam); target_code=AW ;;
  W) source_domain=webcam; targets=(amazon dslr); target_code=AD ;;
  *) echo "Unknown Office-31 source: ${SOURCE}" >&2; exit 2 ;;
esac

run_dir="${ROOT_DIR}/output/office31_mtda/source_probe/${SOURCE}2${target_code}/seed${SEED}"
score_file="${ROOT_DIR}/results/office31_mtda/difficulty_${SOURCE}_seed${SEED}.json"
checkpoint="${run_dir}/ContinuousSharedProjMaPLeMTDA/model.pth.tar-5"
if [[ ! -s "${checkpoint}" ]]; then
  "${PYTHON_BIN}" train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --trainer ContinuousSharedProjMaPLeMTDA \
    --dataset-config-file config/datasets/office31_mtda.yaml \
    --config-file config/trainers/esmaple.yaml \
    --source-domains "${source_domain}" \
    --target-domains "${targets[@]}" \
    --output-dir "${run_dir}" TEST.NO_TEST True
fi

"${PYTHON_BIN}" scripts/mint/score_office31.py \
  --source "${SOURCE}" \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --model-dir "${run_dir}" \
  --output "${score_file}"
