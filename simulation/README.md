# Simulation code

Everything needed to regenerate the results, figures and tables of both papers, starting either from the included inference traces or from the raw Sen1Floods11 satellite data.

## Quick start

```bash
cd simulation
pip install -r requirements.txt
./run_all.sh              # uses the included traces; about 1 hour on 4 cores
./run_all.sh --traces     # also rebuilds the traces from raw data first (downloads ~1 GB)
```

New outputs go to `results_new/` and `figures_new/` (change with `RES=... FIG=... ./run_all.sh`), so the shipped results are never overwritten. Copy figures into `../journal/figures/` or `../conference/figures/` when you want the papers to use them.

## Pipeline

| Step | Script | Input | Output |
|---|---|---|---|
| 0 | `download_sen1floods11.sh [DIR]` | public bucket `gs://sen1floods11` | hand-labelled Sentinel-2 chips and labels |
| 1 | `prep_sen1floods11.py DATA OUT.npz` | raw chips | `traces_sen1floods11.npz`: per-footprint true state, on-board and ground classifier outputs |
| 2 | `experiments.py RES_DIR [what]` | traces | `budget.json`, `acc.json`, `block.json`, `payload.json`, `V.json`, `mix.json`, `chosen_05.json` |
| 3 | `plots.py RES_DIR FIG_DIR` | results | `fig_budget.pdf`, `fig_sensitivity.pdf`, `fig_V_mix.pdf` |
| 3 | `table.py RES_DIR OUT.tex` | `budget.json` | `table_results.tex` |
| 3 | `regions.py [NU] FIG_DIR` | traces (model parameters) | `regions.pdf` (decision regions, nu = 0.25 in the papers) |
| 3 | `fig_inference.py traces.npz OUT.pdf` | traces | `fig_inference.pdf` (journal) |
| 3 | `fig_path.py OUT.pdf` | traces | `fig_path.pdf` (journal, sample path) |

`experiments.py` takes an optional comma-separated list of experiments as its second argument, for example `python3 experiments.py results_new budget,V` (`V` needs `budget` first, since it reuses `chosen_05.json`).

### What each piece does

- **`prep_sen1floods11.py`** scans each 512x512 chip with a 64x64 footprint along a boustrophedon path (one footprint per slot) and marks a footprint flooded if at least 10% of its labelled pixels are water. It trains the on-board classifier (logistic regression on 8x8 pooled RGB+NIR, 257 parameters) and the ground classifier (gradient-boosted trees on 13-band statistics), calibrates both with isotonic regression on the validation split, and also makes the weaker on-board variants `n2`..`n8` by adding sensor noise.
- **`sim.py`** is the trace-driven simulator: UAV circling the area, probabilistic LoS air-to-ground channel with Rayleigh fading and blockage, energy per action, belief update at the ground station, and the policies: CA-DPP (proposed) with its ablations `cadpp_blind` and `cadpp_label`, the baselines adapted from Zakeri et al. 2023, Gan et al. 2025, Pehlivanoglu et al. 2026 and Basnayaka et al. 2025, and a genie `oracle`. All system parameters are in the dictionary `P` at the top.
- **`experiments.py`** runs 5-fold chip-level cross-validation over the 174 validation and test chips. For each fold every policy parameter (V and belief model for CA-DPP, thresholds for the baselines) is tuned on the other four folds and evaluated on the held-out fold over 4 seeds, giving 20 held-out runs per policy and paired statistics (95% CI, paired t-test, Wilcoxon).

## Shipped results

`results/` holds the JSON files behind the figures and table in the papers. Running `plots.py results`, `table.py results`, `regions.py 0.25`, `fig_inference.py` and `fig_path.py` on them reproduces the paper figures pixel for pixel and `table_results.tex` exactly.

## Archive

`archive/` keeps earlier versions for the record: `sim2.py`, `experiments2.py`, `experiments3.py` (single train/test split, superseded by cross-validation), `plots2.py`, `plots3.py`, `results_v2/`, and `results_beta/` (a discounted look-ahead variant of CA-DPP that did not help and was dropped). They are not needed to reproduce the papers.
