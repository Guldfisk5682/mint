#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?usage: run.sh <coop|cocoop|maple|esmaple|damp> <office31|officehome|domainnet> <source-code> [seed]}"
DATASET="${2:?missing dataset}"
SOURCE="${3:?missing source code}"
SEED="${4:-100}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${METHOD}" in
  coop) trainer=CoOpMTDA ;;
  cocoop) trainer=CoCoOpMTDA ;;
  maple) trainer=MaPLeMTDA ;;
  esmaple) trainer=ContinuousSharedProjMaPLeMTDA ;;
  damp) trainer=DAMP ;;
  *) echo "Unknown baseline method: ${METHOD}" >&2; exit 2 ;;
esac

case "${DATASET}:${SOURCE}" in
  officehome:A) source=art; targets=(clipart product real_world); tag=CPR ;;
  officehome:C) source=clipart; targets=(art product real_world); tag=APR ;;
  officehome:P) source=product; targets=(art clipart real_world); tag=ACR ;;
  officehome:R) source=real_world; targets=(art clipart product); tag=ACP ;;
  office31:A) source=amazon; targets=(dslr webcam); tag=DW ;;
  office31:D) source=dslr; targets=(amazon webcam); tag=AW ;;
  office31:W) source=webcam; targets=(amazon dslr); tag=AD ;;
  domainnet:C) source=clipart; targets=(infograph painting quickdraw real sketch); tag=IPQRS ;;
  domainnet:I) source=infograph; targets=(clipart painting quickdraw real sketch); tag=CPQRS ;;
  domainnet:P) source=painting; targets=(clipart infograph quickdraw real sketch); tag=CIQRS ;;
  domainnet:Q) source=quickdraw; targets=(clipart infograph painting real sketch); tag=CIPRS ;;
  domainnet:R) source=real; targets=(clipart infograph painting quickdraw sketch); tag=CIPQS ;;
  domainnet:S) source=sketch; targets=(clipart infograph painting quickdraw real); tag=CIPQR ;;
  *) echo "Unknown dataset/source: ${DATASET}/${SOURCE}" >&2; exit 2 ;;
esac

case "${DATASET}" in
  officehome) dataset_config=config/datasets/office_home_mtda.yaml; dataset_name=Office-Home ;;
  office31) dataset_config=config/datasets/office31_mtda.yaml; dataset_name=Office-31 ;;
  domainnet) dataset_config=config/datasets/domainnet_mtda.yaml; dataset_name=DomainNet ;;
esac

run_dir="${ROOT}/output/${DATASET}_mtda/baselines/${METHOD}/${SOURCE}2${tag}/seed${SEED}"
target_codes=()
for ((index=0; index<${#tag}; index++)); do
  target_codes+=("${tag:index:1}")
done
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Baseline result already exists: ${run_dir}/mtda_metrics.json"
  exit 0
fi

overrides=(TEST.FINAL_MODEL last_step)
if [[ "${METHOD}" == damp ]]; then
  overrides+=(TRAIN.SOURCE_ONLY False DATALOADER.TRAIN_U.SAME_AS_X False)
  if [[ "${DATASET}" == domainnet ]]; then
    overrides+=(TRAIN.MAX_BATCHES_PER_EPOCH 500)
  fi
else
  overrides+=(TRAIN.SOURCE_ONLY True TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS True)
  if [[ "${DATASET}" == domainnet && "${METHOD}" == esmaple ]]; then
    overrides+=(OPTIM.MAX_EPOCH 10 TRAIN.MAX_BATCHES_PER_EPOCH 500)
  fi
fi

"${PYTHON_BIN}" scripts/experiment_guard.py \
  --output-dir "${run_dir}" --method-tag "${METHOD}_source_to_rest" \
  --source "${SOURCE}" --targets "${target_codes[@]}" \
  --seed "${SEED}" --trainer "${trainer}" \
  --trainer-config "config/trainers/${METHOD}.yaml" \
  --dataset-config "${dataset_config}" --data "${DATA_ROOT}" \
  --effective-opts "${overrides[*]}"

"${PYTHON_BIN}" train.py --root "${DATA_ROOT}" --seed "${SEED}" \
  --trainer "${trainer}" --dataset-config-file "${dataset_config}" \
  --config-file "config/trainers/${METHOD}.yaml" \
  --source-domains "${source}" --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" "${overrides[@]}"

"${PYTHON_BIN}" scripts/mint/export_metrics.py --run-dir "${run_dir}" \
  --dataset "${dataset_name}" --method "${METHOD}" \
  --source "${SOURCE}" --seed "${SEED}" --targets "${targets[@]}"
