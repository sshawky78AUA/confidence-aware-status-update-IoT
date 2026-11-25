#!/usr/bin/env bash
# Downloads the hand-labelled Sen1Floods11 chips (about 1 GB) needed by prep_sen1floods11.py.
# Needs the Google Cloud SDK (gsutil); the bucket is public.
set -euo pipefail
DATA=${1:-data}
mkdir -p "$DATA/flood_handlabeled"
gsutil -q cp 'gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_*_data.csv' "$DATA/flood_handlabeled/"
gsutil -m -q cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand \
                   gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand "$DATA/"
echo "Sen1Floods11 data in $DATA"
