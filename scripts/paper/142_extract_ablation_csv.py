#!/usr/bin/env python3
"""
142_extract_ablation_csv.py

Extrait ablation_results.csv depuis la source confirmee :
output/paper_results_final/tables/ablation.tex

Confirmation faite le 2026-09-03 (diagnostic 141) : ce fichier est
byte-identique a paper_results_v3/tables/ablation.tex et
paper_results_E/tables/ablation.tex (474 octets, 5 lignes A-E),
et sa colonne Recall/34 donne 25,25,22,21,18 -- la sequence deja
citee dans le manuscrit. paper_results/ et paper_results_v2/ ne
contiennent que A-D (anterieurs a l'ajout de E) et sont donc
naturellement supersedes, pas en conflit.

L'autre fichier trouve au diagnostic (retrieval_ablation_table.json,
et figures/paper_final/fig_paper_retrieval_ablation_values.json) est
une ablation differente (retrieval stage : memory-only / BM25 /
BM25+reranker) -- il n'est pas touche par ce script.

Usage :
    python 142_extract_ablation_csv.py
    python 140_generate_manuscript_figures.py --outdir figures/ --only ablation
"""

import csv
import re
import sys
from pathlib import Path

SOURCE = Path.home() / "ontogeorag" / "output" / "paper_results_final" / "tables" / "ablation.tex"
OUTPUT = Path.home() / "ontogeorag" / "output" / "run16E" / "deliverable" / "ablation_results.csv"

# Une ligne de donnees ressemble a :
# A & 13 imposed & 249 & 207 & 21 & 25/34 & 19/26 \\
ROW_RE = re.compile(
    r"^([A-E])\s*&\s*(.+?)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)/34\s*&\s*(\d+)/26"
)


def main():
    if not SOURCE.exists():
        print("ERROR: source file not found: " + str(SOURCE), file=sys.stderr)
        print("Ce script ne fabrique aucune valeur ; verifier le chemin.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for line in SOURCE.read_text().splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            config, vocab, queries, triples, distinct_desc, recall34, recall26 = m.groups()
            rows.append({
                "config_label": config,
                "prompt_vocabulary": vocab.strip(),
                "n_queries": queries,
                "n_triples": triples,
                "n_distinct_descriptors": distinct_desc,
                "recall_34": recall34,
                "recall_26": recall26,
            })

    if not rows:
        print("ERROR: no data rows matched in " + str(SOURCE), file=sys.stderr)
        print("Le format du .tex a peut-etre change depuis l'ecriture de ce parseur.", file=sys.stderr)
        sys.exit(1)

    if len(rows) != 5:
        print("WARNING: " + str(len(rows)) + " lignes trouvees, 5 attendues (A-E).", file=sys.stderr)
        print("Verifier " + str(SOURCE) + " avant de continuer.", file=sys.stderr)

    expected = ["25", "25", "22", "21", "18"]
    got = [r["recall_34"] for r in rows]
    if got != expected:
        print("WARNING: sequence recall_34 = " + str(got) +
              ", differente de la sequence citee dans le manuscrit " + str(expected) + ".",
              file=sys.stderr)
        print("Ne pas continuer sans comprendre pourquoi.", file=sys.stderr)
        sys.exit(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("wrote " + str(OUTPUT) + " (" + str(len(rows)) + " rows)")
    print("source: " + str(SOURCE))
    print("recall_34 sequence confirmed: " + str(got))


if __name__ == "__main__":
    main()