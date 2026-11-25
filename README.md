# Direction 1: correctness-aware silent/label/raw updates for UAV sensing

LaTeX sources for two versions of the paper by Sameh Hosny and John Tadrous.

| Folder | What it is | Main file |
|---|---|---|
| `journal/` | Journal master draft (IEEE Internet of Things Journal format) | `journal.tex` |
| `conference/` | Conference version (IEEEtran conference, 7 pages) | `main.tex` |
| `simulation/` | Full simulation pipeline: data preparation, simulator, experiments, figures and table | `run_all.sh` |

Each paper folder compiles on its own with `pdflatex` + `bibtex`, and each is also an Overleaf project.
The `sim/` folder in each paper holds only the TikZ sources of the scenario figures.

To reproduce the results, see [`simulation/README.md`](simulation/README.md). In short:

```bash
cd simulation
pip install -r requirements.txt
./run_all.sh            # from the included inference traces, about 1 hour on 4 cores
./run_all.sh --traces   # from the raw Sen1Floods11 data (downloads about 1 GB)
```
