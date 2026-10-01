#!/usr/bin/env bash
# Download the OpenKBP dataset (AAPM OpenKBP Grand Challenge, head-and-neck IMRT, 340 patients:
# 200 train / 40 validation / 100 test) by cloning the official repository (~2 GB of CSV files).
# The data ends up in dose_prediction/data/open-kbp/provided-data/{train,validation,test}-pats/pt_*/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${OPENKBP_DIR:-${HERE}/open-kbp}"   # override the location with OPENKBP_DIR=...
if [ ! -d "${DEST}" ]; then
    git clone https://github.com/ababier/open-kbp.git "${DEST}"
    # Commit the pipeline was tested with (the dataset has not changed since the challenge);
    # an existing folder is used as it is.
    git -C "${DEST}" checkout -q ce625e6
fi
for s in train-pats validation-pats test-pats; do
    echo "${s}: $(ls "${DEST}/provided-data/${s}" | wc -l) patients"
done
