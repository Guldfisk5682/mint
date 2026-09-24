#!/usr/bin/env bash
set -euo pipefail

# Paired Office-Home ablations use the same label-free score manifest as full MINT.

VARIANT="${1:?usage: run_officehome_ablation.sh <e2h|joint|hard_only|soft_only|no_replay|topk16|topk32> <A|C|P|R> [seed]}"
SOURCE="${2:?missing Office-Home source domain code}"
SEED="${3:-100}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/officehome_mtda}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "${SOURCE}" in
  A) source_domain=art; targets=(clipart product real_world); target_codes=(C P R) ;;
  C) source_domain=clipart; targets=(art product real_world); target_codes=(A P R) ;;
  P) source_domain=product; targets=(art clipart real_world); target_codes=(A C R) ;;
  R) source_domain=real_world; targets=(art clipart product); target_codes=(A C P) ;;
  *) echo "Unknown Office-Home source: ${SOURCE}" >&2; exit 2 ;;
esac
case "${VARIANT}" in
  e2h|joint|hard_only|soft_only|no_replay|topk16|topk32) ;;
  *) echo "Unknown ablation variant: ${VARIANT}" >&2; exit 2 ;;
esac

score_file="${ROOT_DIR}/results/officehome_mtda/difficulty_${SOURCE}_seed${SEED}.json"
order_text="$("${PYTHON_BIN}" scripts/mint/order_from_score.py "${score_file}" \
  --source "${source_domain}" --seed "${SEED}" --targets "${targets[@]}")"
mapfile -t h2e <<< "${order_text}"
order=("${h2e[@]}")
[[ "${VARIANT}" == "e2h" ]] && order=("${h2e[2]}" "${h2e[1]}" "${h2e[0]}")
order_cfg="['${order[0]}','${order[1]}','${order[2]}']"
target_tag="$(IFS=''; echo "${target_codes[*]}")"
run_dir="${OUTPUT_ROOT}/officehome_ablation_${VARIANT}_seed${SEED}/${SOURCE}2${target_tag}/seed${SEED}"

trainer="CurriculumContinuousSharedProjMaPLeMTDA"
trainer_config="config/trainers/mint.yaml"
cfg_opts=(
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
  TRAIN.CHECKPOINT_FREQ 0
)

case "${VARIANT}" in
  joint)
    trainer="ContinuousSharedProjMaPLeMTDA"
    trainer_config="config/trainers/esmaple.yaml"
    cfg_opts=(
      TRAIN.SOURCE_ONLY False
      TRAINER.MAPLE_MTDA.PL_VARIANT agreement_hard_student_soft
      TRAINER.MAPLE_MTDA.LAMBDA_PL 0.3
      TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.3
      TRAINER.MAPLE_MTDA.PL_DUAL_CONF_THRESHOLD 0.7
      TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.5
      TRAINER.MAPLE_MTDA.PL_STRONG_AUGMENT randaugment_fixmatch
      TRAIN.CHECKPOINT_FREQ 0
    )
    ;;
  hard_only)
    cfg_opts+=(TRAINER.MAPLE_MTDA.PL_STUDENT_SOFT_LAMBDA 0.0)
    ;;
  soft_only)
    cfg_opts+=(TRAINER.MAPLE_MTDA.LAMBDA_PL 0.0 TRAINER.MAPLE_MTDA.LAMBDA_PL_FINAL 0.0)
    ;;
  no_replay)
    cfg_opts+=(TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.ENABLED False)
    ;;
  topk16)
    cfg_opts+=(TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS 16)
    ;;
  topk32)
    cfg_opts+=(TRAINER.MAPLE_MTDA.CURRICULUM.REPLAY.TOPK_PER_CLASS 32)
    ;;
esac

"${PYTHON_BIN}" scripts/experiment_guard.py \
  --output-dir "${run_dir}" \
  --method-tag "officehome_ablation_${VARIANT}_seed${SEED}" \
  --source "${SOURCE}" \
  --targets "${target_codes[@]}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --trainer-config "${trainer_config}" \
  --dataset-config config/datasets/office_home_mtda.yaml \
  --score-manifest "${score_file}" \
  --data "${DATA_ROOT}" \
  --effective-opts "${cfg_opts[*]}"
if [[ -s "${run_dir}/mtda_metrics.json" ]]; then
  echo "Completed matching ablation run: ${run_dir}"
  exit 0
fi

"${PYTHON_BIN}" train.py \
  --root "${DATA_ROOT}" \
  --seed "${SEED}" \
  --trainer "${trainer}" \
  --dataset-config-file config/datasets/office_home_mtda.yaml \
  --config-file "${trainer_config}" \
  --source-domains "${source_domain}" \
  --target-domains "${targets[@]}" \
  --output-dir "${run_dir}" \
  "${cfg_opts[@]}" 2>&1 | tee "${run_dir}/console.log"

"${PYTHON_BIN}" scripts/mint/export_metrics.py \
  --run-dir "${run_dir}" --dataset Office-Home --method "MINT_${VARIANT}" \
  --source "${SOURCE}" --seed "${SEED}" --targets "${targets[@]}"
