#!/usr/bin/env bash
# Reproduces every result, figure and table of both papers.
#   ./run_all.sh            use the included traces (traces_sen1floods11.npz)
#   ./run_all.sh --traces   first rebuild the traces from the raw Sen1Floods11 data (downloads ~1 GB)
set -euo pipefail
cd "$(dirname "$0")"
RES=${RES:-results_new}; FIG=${FIG:-figures_new}; mkdir -p "$RES" "$FIG"
if [[ "${1:-}" == "--traces" ]]; then
  ./download_sen1floods11.sh data
  python3 prep_sen1floods11.py data traces_sen1floods11.npz      # step 1: classifiers + inference traces
fi
python3 experiments.py "$RES"                                     # step 2: all experiments (about 75 min on 4 cores)
python3 plots.py "$RES" "$FIG"                                    # step 3: Figs. budget, sensitivity, V/mix
python3 table.py "$RES" "$FIG/table_results.tex"                  #         results table
python3 regions.py 0.25 "$FIG"                                    #         decision-regions figure
python3 fig_inference.py traces_sen1floods11.npz "$FIG/fig_inference.pdf"   # journal only
python3 fig_path.py "$FIG/fig_path.pdf"                                      # journal only
echo "Results in $RES, figures and table in $FIG"
