#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEFAULT_DATA_ROOT="${REPO_ROOT}/data"
OFFICEHOME_GDRIVE_URL_DEFAULT="https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw&usp=sharing"
HF_ENDPOINT_DEFAULT="https://hf-mirror.com"
HF_DATASET_REPO_DEFAULT="flwrlabs/office-home"
HF_DATASET_SPLIT_DEFAULT="train"

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
OFFICEHOME_ARCHIVE="${OFFICEHOME_ARCHIVE:-}"
OFFICEHOME_URL="${OFFICEHOME_URL:-}"
OFFICEHOME_ARCHIVE_NAME="${OFFICEHOME_ARCHIVE_NAME:-OfficeHomeDataset_10072016.zip}"
HF_ENDPOINT="${HF_ENDPOINT:-${HF_ENDPOINT_DEFAULT}}"
HF_DATASET_REPO="${HF_DATASET_REPO:-${HF_DATASET_REPO_DEFAULT}}"
HF_DATASET_SPLIT="${HF_DATASET_SPLIT:-${HF_DATASET_SPLIT_DEFAULT}}"
OFFICEHOME_SOURCE="${OFFICEHOME_SOURCE:-auto}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT%/}/downloads}"
TARGET_DIR="${DATA_ROOT%/}/office_home"
ARCHIVE_PATH=""

canonical_domain_name() {
  local raw_name
  raw_name="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  raw_name="${raw_name// /_}"
  raw_name="${raw_name//-/_}"
  case "${raw_name}" in
    art) echo "art" ;;
    clipart) echo "clipart" ;;
    product) echo "product" ;;
    real_world|realworld|real_worlds) echo "real_world" ;;
    *)
      echo ""
      ;;
  esac
}

find_domain_dir() {
  local search_root="$1"
  local wanted="$2"
  find "${search_root}" -type d | while read -r candidate; do
    local base
    base="$(basename "${candidate}")"
    if [ "$(canonical_domain_name "${base}")" = "${wanted}" ]; then
      echo "${candidate}"
      break
    fi
  done
}

download_from_hf() {
  cat <<EOF
Falling back to Hugging Face Office-Home source

HF dataset:
  ${HF_DATASET_REPO}
HF split:
  ${HF_DATASET_SPLIT}
HF endpoint:
  ${HF_ENDPOINT}
EOF

  TARGET_DIR="${TARGET_DIR}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_DATASET_REPO="${HF_DATASET_REPO}" \
  HF_DATASET_SPLIT="${HF_DATASET_SPLIT}" \
  python "${REPO_ROOT}/scripts/datasets/prepare_officehome_from_hf.py"
}

mkdir -p "${TARGET_DIR}" "${DOWNLOAD_DIR}"

if [ -d "${TARGET_DIR}/art" ] && [ -d "${TARGET_DIR}/clipart" ] && [ -d "${TARGET_DIR}/product" ] && [ -d "${TARGET_DIR}/real_world" ]; then
  echo "Office-Home already looks prepared under ${TARGET_DIR}"
  exit 0
fi

if [ -d "${TARGET_DIR}/art" ] || [ -d "${TARGET_DIR}/clipart" ] || [ -d "${TARGET_DIR}/product" ] || [ -d "${TARGET_DIR}/real_world" ]; then
  echo "Found a partial Office-Home directory under ${TARGET_DIR}. Please clean it up before rerunning." >&2
  exit 1
fi

if [ "${OFFICEHOME_SOURCE}" = "hf" ]; then
  download_from_hf
  exit 0
fi

if [ -n "${OFFICEHOME_ARCHIVE}" ]; then
  ARCHIVE_PATH="${OFFICEHOME_ARCHIVE}"
else
  if [ "${OFFICEHOME_SOURCE}" = "archive" ]; then
    echo "OFFICEHOME_SOURCE=archive requires OFFICEHOME_ARCHIVE to be set." >&2
    exit 2
  fi

  ARCHIVE_PATH="${DOWNLOAD_DIR}/${OFFICEHOME_ARCHIVE_NAME}"
  if [ ! -f "${ARCHIVE_PATH}" ]; then
    if [ -z "${OFFICEHOME_URL}" ]; then
      OFFICEHOME_URL="${OFFICEHOME_GDRIVE_URL_DEFAULT}"
      cat <<EOF
Office-Home download source

Official page:
  http://hemanthdv.org/OfficeHome-Dataset/

Default archive source for this script:
  ${OFFICEHOME_URL}

You can override it with either:

1. a local archive:
   OFFICEHOME_ARCHIVE=/path/to/${OFFICEHOME_ARCHIVE_NAME} DATA_ROOT=${DATA_ROOT} $0

2. another direct URL:
   OFFICEHOME_URL=https://.../${OFFICEHOME_ARCHIVE_NAME} DATA_ROOT=${DATA_ROOT} $0

3. HF fallback dataset:
   ${HF_DATASET_REPO} via ${HF_ENDPOINT}
EOF
    fi

    if [ "${OFFICEHOME_SOURCE}" = "gdrive" ] || [ "${OFFICEHOME_SOURCE}" = "auto" ]; then
      if python -m gdown --help >/dev/null 2>&1; then
        set +e
        if python -m gdown --help 2>&1 | grep -q -- "--fuzzy"; then
          python -m gdown --fuzzy "${OFFICEHOME_URL}" -O "${ARCHIVE_PATH}"
        else
          python -m gdown "${OFFICEHOME_URL}" -O "${ARCHIVE_PATH}"
        fi
        GDOWN_STATUS=$?
        set -e
        if [ "${GDOWN_STATUS}" -ne 0 ]; then
          rm -f "${ARCHIVE_PATH}"
          if [ "${OFFICEHOME_SOURCE}" = "auto" ]; then
            download_from_hf
            exit 0
          fi
          exit "${GDOWN_STATUS}"
        fi
      elif command -v curl >/dev/null 2>&1 && [[ "${OFFICEHOME_URL}" =~ ^https?:// ]]; then
        set +e
        curl -L "${OFFICEHOME_URL}" -o "${ARCHIVE_PATH}"
        CURL_STATUS=$?
        set -e
        if [ "${CURL_STATUS}" -ne 0 ]; then
          rm -f "${ARCHIVE_PATH}"
          if [ "${OFFICEHOME_SOURCE}" = "auto" ]; then
            download_from_hf
            exit 0
          fi
          exit "${CURL_STATUS}"
        fi
      elif command -v wget >/dev/null 2>&1 && [[ "${OFFICEHOME_URL}" =~ ^https?:// ]]; then
        set +e
        wget -O "${ARCHIVE_PATH}" "${OFFICEHOME_URL}"
        WGET_STATUS=$?
        set -e
        if [ "${WGET_STATUS}" -ne 0 ]; then
          rm -f "${ARCHIVE_PATH}"
          if [ "${OFFICEHOME_SOURCE}" = "auto" ]; then
            download_from_hf
            exit 0
          fi
          exit "${WGET_STATUS}"
        fi
      elif [ "${OFFICEHOME_SOURCE}" = "auto" ]; then
        download_from_hf
        exit 0
      else
        echo "Could not use Google Drive source with OFFICEHOME_SOURCE=${OFFICEHOME_SOURCE}." >&2
        exit 2
      fi
    else
      download_from_hf
      exit 0
    fi
  fi
fi

if [ ! -f "${ARCHIVE_PATH}" ]; then
  cat <<EOF >&2
Office-Home archive not found after primary download attempt: ${ARCHIVE_PATH}

Will not continue with archive extraction.
If you intended to use HF fallback directly, please run:

  OFFICEHOME_SOURCE=hf HF_ENDPOINT=${HF_ENDPOINT} HF_DATASET_REPO=${HF_DATASET_REPO} bash $0
EOF
  exit 4
fi

TMP_DIR="$(mktemp -d /tmp/officehome_extract.XXXXXX)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

case "${ARCHIVE_PATH}" in
  *.zip)
    unzip -q "${ARCHIVE_PATH}" -d "${TMP_DIR}"
    ;;
  *.tar.gz|*.tgz)
    tar -xzf "${ARCHIVE_PATH}" -C "${TMP_DIR}"
    ;;
  *.tar)
    tar -xf "${ARCHIVE_PATH}" -C "${TMP_DIR}"
    ;;
  *)
    echo "Unsupported archive format: ${ARCHIVE_PATH}" >&2
    exit 5
    ;;
esac

for domain_name in art clipart product real_world; do
  found_dir="$(find_domain_dir "${TMP_DIR}" "${domain_name}")"
  if [ -z "${found_dir}" ]; then
    echo "Could not locate domain '${domain_name}' after extraction from ${ARCHIVE_PATH}" >&2
    exit 6
  fi

  source_dir="${found_dir}"
  if [ -d "${found_dir}/images" ]; then
    source_dir="${found_dir}/images"
  fi

  mkdir -p "${TARGET_DIR}/${domain_name}"

  copied_any=0
  find "${source_dir}" -mindepth 1 -maxdepth 1 -type d | while read -r class_dir; do
    cp -a "${class_dir}" "${TARGET_DIR}/${domain_name}/"
    copied_any=1
  done

  if [ "$(find "${TARGET_DIR}/${domain_name}" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 0 ]; then
    echo "No class folders were copied for domain '${domain_name}' from ${source_dir}" >&2
    exit 7
  fi
done

echo "Office-Home is ready under ${TARGET_DIR}"
