# À faire avant publication (généré par 153, à compléter)

## Figures sans producteur sur le cluster
- Figure 2 `fig_layers.png`, Figure 4 `fig_descriptor_vocabulary.png`, Figure 5 `fig_vocabulary_gap.png` : script à réécrire.
- Figure 6 `fig_sensitivity.png` : comparer avec `figures_new/fig_corruption_class.png` (98).

## Non copié volontairement
- Graphe livré et graphe à provenance localisée : contiennent des phrases d'articles (décision de licence).
- `reference/Table_Supplementary_1_V2.xlsx` : matériel supplémentaire publié (licence à vérifier).
- Corpus, index, `chunks.jsonl` : texte sous droits.

## Chemins personnels restant dans les fichiers copiés
- `61_extend_queries.py` : {"~/ontogeorag": 1}
- `m4/m4_aggregate.py` : {"~/ontogeorag": 2}
- `m4/m4_direction_check.py` : {"~/ontogeorag": 3}
- `m4/m4_integrate_tiers.py` : {"~/ontogeorag": 3}
- `m4/m4_metrics.py` : {"~/ontogeorag": 3}
- `m4/m4_negatives.py` : {"~/ontogeorag": 7}
- `m4/m4_panel.py` : {"~/ontogeorag": 3}
- `m4/m4_verify.py` : {"~/ontogeorag": 3}
- `results/run16/manifest.json` : {"/home/talbi": 52, ".cache/huggingface": 6}
- `results/run16B/manifest.json` : {"/home/talbi": 52, ".cache/huggingface": 6}
- `results/run16C/manifest.json` : {"/home/talbi": 52, ".cache/huggingface": 6}
- `results/run16D/manifest.json` : {"/home/talbi": 53, ".cache/huggingface": 6}
- `results/run16E/manifest.json` : {"/home/talbi": 55, ".cache/huggingface": 6}
- `scripts/analysis/93_graph_validation_run16E.py` : {"~/ontogeorag": 4}
- `scripts/analysis/94_failure_modes.py` : {"~/ontogeorag": 6}
- `scripts/analysis/95p_commands.sh` : {"/home/talbi": 21, "kg_test": 3, ".cache/huggingface": 3}
- `scripts/analysis/96_descriptor_frequencies_run16E.py` : {"~/ontogeorag": 4}
- `scripts/figures/95y_fig_tradeoff.py` : {"~/ontogeorag": 1}
- `scripts/paper/67_paper_results.py` : {"~/ontogeorag": 10}
- `scripts/paper/68_classify_vocabulary.py` : {"~/ontogeorag": 2}
- `scripts/tools/48e_preflight_41.py` : {"~/ontogeorag": 3}
- `slurm/run16_full.sh` : {"/home/talbi": 4, "kg_test": 1}

## Scripts dont un chemin relatif peut changer après rangement sous scripts/
- Les scripts de scripts/ se lancent depuis la racine du dossier : `PYTHONPATH=. python scripts/<dossier>/<script>.py`.
- `95g_derniers_controles.py` se réfère à `96_descriptor_frequencies_run16E.py` : les deux sont dans scripts/analysis/, vérifier le chemin utilisé.
