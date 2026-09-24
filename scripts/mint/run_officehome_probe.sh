#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_officehome_probe.sh <A|C|P|R> [seed]}"
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

run_dir="${ROOT}/output/officehome_mtda/source_probe/${SOURCE}2${tag}/seed${SEED}"
score_file="${ROOT}/results/officehome_mtda/difficulty_${SOURCE}_seed${SEED}.json"
checkpoint="${run_dir}/ContinuousSharedProjMaPLeMTDA/model.pth.tar-5"
if [[ ! -s "${checkpoint}" ]]; then
  "${PYTHON_BIN}" train.py --root "${DATA_ROOT}" --seed "${SEED}" \
    --trainer ContinuousSharedProjMaPLeMTDA \
    --dataset-config-file config/datasets/office_home_mtda.yaml \
    --config-file config/trainers/esmaple.yaml \
    --source-domains "${source}" --target-domains "${targets[@]}" \
    --output-dir "${run_dir}" TEST.NO_TEST True
fi
"${PYTHON_BIN}" scripts/mint/score_officehome.py \
  --source "${SOURCE}" --root "${DATA_ROOT}" --seed "${SEED}" \
  --model-dir "${run_dir}" --output "${score_file}"
