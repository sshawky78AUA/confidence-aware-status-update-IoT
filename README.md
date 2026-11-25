# Confidence-Aware Status Updates from On-Board Classifiers in Aerial IoT

Paper sources and simulation code by Sameh Hosny and John Tadrous.

| Folder | What it is |
|---|---|
| `paper/` | The LaTeX project, same layout as the Overleaf project |
| `simulation/` | Full simulation pipeline: data preparation, simulator, experiments, figures and table |
| `figure-sources/` | TikZ sources of the scenario and Markov-chain figures (standalone; compile to the PDFs in `paper/figures/`) |

## Paper

`paper/` holds two self-contained documents that share one `refs.bib` and one `figures/` folder:

| File | Version |
|---|---|
| `journal.tex` | Journal master draft (IEEE Internet of Things Journal, double column) |
| `conference.tex` | Conference version (IEEEtran conference, 7 pages), cut from the journal |

Compile either with `pdflatex` + `bibtex` (in Overleaf, pick it as the main document).
The simulation code and figure sources live only here on GitHub, not in Overleaf, since the papers only need the figure PDFs.

## Simulation

To reproduce the results, see [`simulation/README.md`](simulation/README.md). In short:

```bash
cd simulation
pip install -r requirements.txt
./run_all.sh            # from the included inference traces, about 75 minutes on 4 cores
./run_all.sh --traces   # from the raw Sen1Floods11 data (downloads about 1 GB)
```
