#!/usr/bin/env bash
set -euo pipefail

# Download the cleaned DomainNet release and the official train/test lists used
# by DomainNetMTDA. The archives total roughly 18 GB; allow substantially more
# free space for extraction.

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT%/}/downloads/domainnet}"
TARGET_DIR="${DATA_ROOT%/}/DomainNet"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"
VERIFY_LAYOUT="${VERIFY_LAYOUT:-1}"
BASE_URL="${DOMAINNET_BASE_URL:-https://csr.bu.edu/ftp/visda/2019/multi-source}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${DOMAINNET_DOMAINS:-}" ]]; then
  # Space-separated subset for resumable parallel staging. The default remains
  # the complete official six-domain release.
  read -r -a domains <<< "${DOMAINNET_DOMAINS}"
else
  domains=(clipart infograph painting quickdraw real sketch)
fi

if [[ "${DATA_ROOT}" == "/path/to/datasets" ]]; then
  echo "Set DATA_ROOT to the dataset parent directory." >&2
  exit 1
fi

mkdir -p "${DOWNLOAD_DIR}" "${TARGET_DIR}/image_list"

download_file() {
  local url="$1"
  local destination="$2"
  local partial="${destination}.part"

  if [[ -s "${destination}" ]]; then
    # A non-empty interrupted zip used to be mistaken for a completed file,
    # causing a later unzip failure. Listing the central directory is a fast
    # completeness check without decompressing the archive.
    if [[ "${destination}" == *.zip ]] && ! unzip -Z -1 "${destination}" >/dev/null 2>&1; then
      echo "Resuming incomplete archive ${destination}"
      mv "${destination}" "${partial}"
    else
      echo "Using existing ${destination}"
      return
    fi
  fi
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 8 --retry-all-errors --continue-at - \
      --output "${partial}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --tries=8 --output-document="${partial}" "${url}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 2
  fi
  mv "${partial}" "${destination}"
}

extract_archive() {
  local archive="$1"
  local destination="$2"
  local windows_tar="/mnt/c/Windows/System32/tar.exe"

  if [[ "${destination}" == /mnt/?/* ]] && [[ -x "${windows_tar}" ]] \
      && command -v wslpath >/dev/null 2>&1; then
    echo "Using Windows-native extraction for mounted drive"
    "${windows_tar}" -xf "$(wslpath -w "${archive}")" \
      -C "$(wslpath -w "${destination}")"
  else
    unzip -q "${archive}" -d "${destination}"
  fi
}

for domain in "${domains[@]}"; do
  case "${domain}" in
    clipart|infograph|painting|quickdraw|real|sketch) ;;
    *) echo "Unsupported DomainNet domain: ${domain}" >&2; exit 3 ;;
  esac
  archive="${DOWNLOAD_DIR}/${domain}.zip"
  if [[ "${domain}" == "clipart" || "${domain}" == "painting" ]]; then
    archive_url="${BASE_URL}/groundtruth/${domain}.zip"
  else
    archive_url="${BASE_URL}/${domain}.zip"
  fi
  download_file "${archive_url}" "${archive}"

  if [[ ! -d "${TARGET_DIR}/${domain}" ]]; then
    echo "Extracting ${archive}"
    extract_archive "${archive}" "${TARGET_DIR}"
  else
    echo "Domain directory already exists; not extracting again: ${TARGET_DIR}/${domain}"
  fi

  for split in train test; do
    download_file \
      "${BASE_URL}/domainnet/txt/${domain}_${split}.txt" \
      "${TARGET_DIR}/image_list/${domain}_${split}.txt"
  done

  if [[ "${KEEP_ARCHIVES}" == "0" ]]; then
    rm -f "${archive}"
  fi
done

if [[ "${VERIFY_LAYOUT}" == "1" ]]; then
  python "${REPO_ROOT}/scripts/datasets/verify_domainnet_layout.py" \
    --root "${DATA_ROOT}"
fi
