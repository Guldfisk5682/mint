#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_domainnet_probe.sh <C|I|P|Q|R|S> [seed]}"
SEED="${2:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${SOURCE}" in
  C) source_domain=clipart; targets=(infograph painting quickdraw real sketch) ;;
  I) source_domain=infograph; targets=(clipart painting quickdraw real sketch) ;;
  P) source_domain=painting; targets=(clipart infograph quickdraw real sketch) ;;
  Q) source_domain=quickdraw; targets=(clipart infograph painting real sketch) ;;
  R) source_domain=real; targets=(clipart infograph painting quickdraw sketch) ;;
  S) source_domain=sketch; targets=(clipart infograph painting quickdraw real) ;;
  *) echo "Unknown DomainNet source: ${SOURCE}" >&2; exit 2 ;;
esac

run_dir="${ROOT_DIR}/output/domainnet_mtda/source_probe/${SOURCE}2O/seed${SEED}"
score_file="${ROOT_DIR}/results/domainnet_mtda/difficulty_${SOURCE}_seed${SEED}.json"
checkpoint="${run_dir}/ContinuousSharedProjMaPLeMTDA/model.pth.tar-10"

if [[ ! -s "${checkpoint}" ]]; then
  "${PYTHON_BIN}" train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --trainer ContinuousSharedProjMaPLeMTDA \
    --dataset-config-file config/datasets/domainnet_mtda.yaml \
    --config-file config/trainers/esmaple.yaml \
    --source-domains "${source_domain}" \
    --target-domains "${targets[@]}" \
    --output-dir "${run_dir}" \
    OPTIM.MAX_EPOCH 10 \
    TRAIN.MAX_BATCHES_PER_EPOCH 500 \
    TRAIN.SOURCE_ONLY True \
    TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS True \
    TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT 0.0 \
    TEST.NO_TEST True
fi

"${PYTHON_BIN}" scripts/mint/score_domainnet.py \
  --source "${SOURCE}" \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --model-dir "${run_dir}" \
  --output "${score_file}"
