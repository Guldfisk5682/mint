#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DASSL_DIR="${DASSL_DIR:-${REPO_ROOT}/../Dassl.pytorch}"
DASSL_REV="${DASSL_REV:-c61a1b570ac6333bd50fb5ae06aea59002fb20bb}"

if [ ! -d "${DASSL_DIR}" ]; then
  git clone https://github.com/KaiyangZhou/Dassl.pytorch.git "${DASSL_DIR}"
fi

if [ "$(git -C "${DASSL_DIR}" rev-parse HEAD)" != "${DASSL_REV}" ]; then
  echo "Dassl revision mismatch: expected ${DASSL_REV}, found $(git -C "${DASSL_DIR}" rev-parse HEAD)" >&2
  echo "Set DASSL_DIR to a clean pinned checkout; the installer will not alter an existing dependency." >&2
  exit 2
fi

python -m pip install -e "${DASSL_DIR}"
python -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "Dassl and project dependencies are installed."
