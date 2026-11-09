# run18 figure scripts

Producers of the figures of the run18 manuscript. They read only files of this repository (`results/run18/` and `configs/`), need no model and no GPU, and run in a few seconds. Each script checks its counts against the values printed in the manuscript and writes nothing if one of them differs. Shared inputs, colours and helpers are in `run18_style.py`.

| Script | Figure | Main input |
|---|---|---|
| `fig_assessments_run18.py` | `fig:assessments`, the 364 delivered relations under seven assessments | delivered graph, provenance table of 187, outputs of 188 |
| `fig_text_to_graph_run18.py` | `fig:texttograph`, from four sentences of Alves et al. (2014) to the cleaned graph | graph with located provenance (187), removed relations (188) |
| `fig_overview_zoom_run18.py` | `fig:overview`, the cleaned graph by family and the neighbourhood of mass transport deposits | cleaned graph (188) |
| `fig_lb2019_vs_graph_run18.py` | `fig:lb2019`, the 26 expert relations and their status in the cleaned graph | cleaned graph, `configs/lb_reference_edges_original26.json`, `configs/entity_families_run18.csv` |
| `fig_pressure_families_run18.py` | `fig:pressurefamilies`, the neighbourhood of slope failure by family | cleaned graph (188) |

Run all five from the repository root:

```
for f in assessments text_to_graph overview_zoom lb2019_vs_graph pressure_families; do
    python scripts/figures/run18/fig_${f}_run18.py --out output/figures/fig_${f}_run18
done
```

Each script writes a PDF and a PNG. Python 3.10, matplotlib and networkx (see `environment/requirements-cpu.txt`).
