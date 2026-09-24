#!/bin/bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
TARGET_DIR="${DATA_ROOT%/}/office31"
DOMAINS=(amazon dslr webcam)

if [ "${DATA_ROOT}" = "/path/to/datasets" ]; then
  echo "Please set DATA_ROOT to a real dataset root before verifying Office-31." >&2
  exit 1
fi

if [ ! -d "${TARGET_DIR}" ]; then
  echo "Office-31 root not found: ${TARGET_DIR}" >&2
  exit 2
fi

echo "Verifying Office-31 layout under ${TARGET_DIR}"
echo

for domain in "${DOMAINS[@]}"; do
  domain_dir="${TARGET_DIR}/${domain}"
  if [ ! -d "${domain_dir}" ]; then
    echo "[FAIL] missing domain directory: ${domain_dir}" >&2
    exit 3
  fi

  class_count="$(find "${domain_dir}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
  image_count="$(find "${domain_dir}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' ')"

  echo "[OK] ${domain}"
  echo "  classes: ${class_count}"
  echo "  images:  ${image_count}"

  if [ "${class_count}" = "0" ] || [ "${image_count}" = "0" ]; then
    echo "[FAIL] ${domain} has no classes or images" >&2
    exit 4
  fi
done

echo
echo "Office-31 layout verification passed."
