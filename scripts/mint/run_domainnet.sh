#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?usage: run_domainnet.sh <C|I|P|Q|R|S> [seed]}"
SEED="${2:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUDGET_MODE="${BUDGET_MODE:-fixed10k}"

case "${BUDGET_MODE}" in
  fixed10k)
    max_epoch=10
    max_batches_per_epoch=1000
    default_method_tag=ours_fixed10k
    ;;
  maple_full)
    max_epoch=5
    max_batches_per_epoch=-1
    default_method_tag=ours_maple_full
    ;;
  *)
    echo "Unknown BUDGET_MODE=${BUDGET_MODE}; expected fixed10k or maple_full" >&2
    exit 2
    ;;
esac
METHOD_TAG="${METHOD_TAG:-${default_method_tag}}"

case "${SOURCE}" in
  C) source_domain=clipart; targets=(infograph painting quickdraw real sketch); target_codes=(I P Q R S) ;;
  I) source_domain=infograph; targets=(clipart painting quickdraw real sketch); target_codes=(C P Q R S) ;;
  P) source_domain=painting; targets=(clipart infograph quickdraw real sketch); target_codes=(C I Q R S) ;;
  Q) source_domain=quickdraw; targets=(clipart infograph painting real sketch); target_codes=(C I P R S) ;;
  R) source_domain=real; targets=(clipart infograph painting quickdraw sketch); target_codes=(C I P Q S) ;;
  S) source_domain=sketch; targets=(clipart infograph painting quickdraw real); target_codes=(C I P Q R) ;;
  *) echo "Unknown DomainNet source: ${SOURCE}" >&2; exit 2 ;;
esac

difficulty_file="${ROOT_DIR}/results/domainnet_mtda/difficulty_${SOURCE}_seed${SEED}.json"
order_text="$("${PYTHON_BIN}" scripts/mint/order_from_score.py "${difficulty_file}" \
  --source "${source_domain}" --seed "${SEED}" --targets "${targets[@]}")"
mapfile -t order <<< "${order_text}"
order_cfg="['${order[0]}','${order[1]}','${order[2]}','${order[3]}','${order[4]}']"
run_dir="${ROOT_DIR}/output/domainnet_mtda/${METHOD_TAG}/${SOURCE}2O/seed${SEED}"

cfg_opts=(
  OPTIM.MAX_EPOCH "${max_epoch}"
  TRAIN.MAX_BATCHES_PER_EPOCH "${max_batches_per_epoch}"
  TRAINER.MAPLE_MTDA.CURRICULUM.DOMAIN_ORDER "${order_cfg}"
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.ENABLED True
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS 8
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LAMBDA 0.75
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.SELECTION_MODE online
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.LABEL_SOURCE pseudo
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TRAVERSAL cycle
  TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.NORMALIZATION none
  TRAINER.MAPLE_MTDA.CURRICULUM.DIAGNOSTICS.ENABLED False
  TRAINER.MAPLE_MTDA.CURRICULUM.RESET_OPTIM_PER_STAGE False
  TRAINER.MAPLE_MTDA.PL_VARIANT agreement_hard_student_soft
  TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3
  TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.3
  TRAINER.MAPLE_MTDA.PL_DUAL_CONF_THRESHOLD 0.7
  TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.5
  TRAINER.MAPLE_MTDA.PL_STRONG_AUGMENT randaugment_fixmatch
)

echo "DomainNet ours budget: mode=${BUDGET_MODE}, epochs=${max_epoch}, max_batches_per_epoch=${max_batches_per_epoch}"

"${PYTHON_BIN}" scripts/experiment_guard.py \
  --output-dir "${run_dir}" \
  --method-tag "${METHOD_TAG}" \
  --source "${SOURCE}" \
  --targets "${target_codes[@]}" \
  --seed "${SEED}" \
  --trainer CurriculumContinuousSharedProjMaPLeMTDA \
  --trainer-config config/trainers/mint.yaml \
  --dataset-config config/datasets/domainnet_mtda.yaml \
  --data "${DATA_ROOT}" \
  --effective-opts "${cfg_opts[*]}"

"${PYTHON_BIN}" train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer CurriculumContinuousSharedProjMaPLeMTDA \
  --dataset-config-file config/datasets/domainnet_mtda.yaml \
  --config-file config/trainers/mint.yaml \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  "${cfg_opts[@]}"

"${PYTHON_BIN}" scripts/mint/export_metrics.py --run-dir "${run_dir}" \
  --dataset DomainNet --method MINT --source "${SOURCE}" \
  --seed "${SEED}" --targets "${targets[@]}"
