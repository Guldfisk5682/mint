#!/bin/bash

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
OFFICE31_ARCHIVE="${OFFICE31_ARCHIVE:-}"
OFFICE31_FILE_ID="${OFFICE31_FILE_ID:-0B4IapRTv9pJ1WGZVd1VDMmhwdlE}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT%/}/downloads}"
TARGET_DIR="${DATA_ROOT%/}/office31"
ARCHIVE_PATH=""

find_domain_dir() {
  local search_root="$1"
  local domain_name="$2"
  find "${search_root}" -type d -name "${domain_name}" | head -n 1
}

if [ "${DATA_ROOT}" = "/path/to/datasets" ]; then
  echo "Please set DATA_ROOT to a real dataset root before downloading Office-31." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}" "${DOWNLOAD_DIR}"

if [ -d "${TARGET_DIR}/amazon" ] && [ -d "${TARGET_DIR}/dslr" ] && [ -d "${TARGET_DIR}/webcam" ]; then
  echo "Office-31 already looks prepared under ${TARGET_DIR}"
  exit 0
fi

if [ -d "${TARGET_DIR}/amazon" ] || [ -d "${TARGET_DIR}/dslr" ] || [ -d "${TARGET_DIR}/webcam" ]; then
  echo "Found a partial Office-31 directory under ${TARGET_DIR}. Please clean it up before rerunning." >&2
  exit 1
fi

if [ -n "${OFFICE31_ARCHIVE}" ]; then
  ARCHIVE_PATH="${OFFICE31_ARCHIVE}"
else
  ARCHIVE_PATH="${DOWNLOAD_DIR}/domain_adaptation_images.tar.gz"
  if [ ! -f "${ARCHIVE_PATH}" ]; then
    cat <<EOF
Office-31 download link note

This script downloads the image archive using a Google Drive file id commonly
used by third-party Office-31 dataset loaders:

  ${OFFICE31_FILE_ID}

If this mirror becomes unavailable, provide a local archive explicitly:

  OFFICE31_ARCHIVE=/path/to/domain_adaptation_images.tar.gz DATA_ROOT=${DATA_ROOT} $0
EOF
    if python -m gdown --help 2>&1 | grep -q -- "--fuzzy"; then
      python -m gdown --fuzzy "https://drive.google.com/uc?id=${OFFICE31_FILE_ID}" -O "${ARCHIVE_PATH}"
    else
      python -m gdown "https://drive.google.com/uc?id=${OFFICE31_FILE_ID}" -O "${ARCHIVE_PATH}"
    fi
  fi
fi

if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "Office-31 archive not found: ${ARCHIVE_PATH}" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d /tmp/office31_extract.XXXXXX)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

case "${ARCHIVE_PATH}" in
  *.tar.gz|*.tgz)
    tar -xzf "${ARCHIVE_PATH}" -C "${TMP_DIR}"
    ;;
  *.zip)
    unzip -q "${ARCHIVE_PATH}" -d "${TMP_DIR}"
    ;;
  *)
    echo "Unsupported archive format: ${ARCHIVE_PATH}" >&2
    exit 3
    ;;
esac

AMAZON_DIR="$(find_domain_dir "${TMP_DIR}" amazon)"
DSLR_DIR="$(find_domain_dir "${TMP_DIR}" dslr)"
WEBCAM_DIR="$(find_domain_dir "${TMP_DIR}" webcam)"

if [ -z "${AMAZON_DIR}" ] || [ -z "${DSLR_DIR}" ] || [ -z "${WEBCAM_DIR}" ]; then
  cat <<EOF
Could not locate all three Office-31 domains after extraction.

Archive used:
  ${ARCHIVE_PATH}

Found:
  amazon=${AMAZON_DIR:-missing}
  dslr=${DSLR_DIR:-missing}
  webcam=${WEBCAM_DIR:-missing}
EOF
  exit 4
fi

cp -a "${AMAZON_DIR}" "${TARGET_DIR}/amazon"
cp -a "${DSLR_DIR}" "${TARGET_DIR}/dslr"
cp -a "${WEBCAM_DIR}" "${TARGET_DIR}/webcam"

echo "Office-31 is ready under ${TARGET_DIR}"
