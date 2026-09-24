#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_DATA_ROOT="${REPO_ROOT}/data"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
TARGET_DIR="${DATA_ROOT%/}/office_home"
DOMAINS=(art clipart product real_world)

if [ ! -d "${TARGET_DIR}" ]; then
  echo "Office-Home directory not found: ${TARGET_DIR}" >&2
  exit 2
fi

reference_classes=""

for domain_name in "${DOMAINS[@]}"; do
  domain_dir="${TARGET_DIR}/${domain_name}"
  if [ ! -d "${domain_dir}" ]; then
    echo "Missing Office-Home domain directory: ${domain_dir}" >&2
    exit 3
  fi

  class_count="$(find "${domain_dir}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  image_count="$(find "${domain_dir}" -type f | wc -l | tr -d ' ')"
  classes_now="$(find "${domain_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)"

  if [ -z "${reference_classes}" ]; then
    reference_classes="${classes_now}"
  elif [ "${reference_classes}" != "${classes_now}" ]; then
    echo "Class folder mismatch detected in ${domain_dir}" >&2
    exit 4
  fi

  echo "${domain_name}: classes=${class_count} images=${image_count}"
done

echo "Office-Home layout looks consistent under ${TARGET_DIR}"
