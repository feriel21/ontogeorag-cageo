# run18 self-audit outputs

These files are the recorded outputs of the self-audit of the manuscript. They were copied unchanged from the authors' cluster on 10 November 2026; `SHA256SUMS` gives the checksum of each copy. They were written before the audit scripts were translated, so their file names, column names and summaries (`_resume.txt`) are in French. A rerun of the translated scripts writes the same content under English names (`scripts/run18/audit/README.md`).

The evidence sentences that some files contain are short quotations from the source papers and remain subject to their publishers' terms, as in the delivered graph.

## Stress test (`stress_test/`)

Produced by `slurm/stress18.sh` (SLURM job 110415, 2 h 47 min on one A100 slice), at commit `b0e91f5` of the run, from the delivered graph `results/run18/kg_final/flat_run18_enforced_dedup.json` (sha256 starting `caccdca00ff3`). The two scripts it calls, `m4/m4_negatives.py` and `m4/m4_verify.py`, have the checksums recorded in `PROVENANCE.tsv`.

| File | Content | Used in the manuscript |
|---|---|---|
| `controls.jsonl`, `controls_meta.json` | the 728 judged records: 364 delivered relations and one corrupted copy of each (159 replaced entities, 91 swapped evidence sentences, 91 changed relation types, 23 inverted directions), seed 13 | Section on controlled perturbations |
| `llama/`, `mistral/` | verdicts of each verifying-LLM on the 728 records (`m4_verdicts.jsonl`), run settings (`m4_run_meta.json`: greedy decoding, evidence read up to 1,500 characters) and per-verifier report (`negatives_report.json`) | `tab:stress`, columns Llama and Mistral |
| `regles_decision.json`, `.txt` | rates of the panel under the decision rules compared by `181_decision_rules.py` | `tab:stress`, columns Evidence check and Plausibility only |
| `_resume.txt` | summary written by the driver | |

## Audit folders (`audit/`)

Each folder is the output of the audit script with the same number in `scripts/run18/audit/` (English name in brackets).

| Folder | Script | Used in the manuscript |
|---|---|---|
| `174_gate_run18_20261022_111908` | `174_audit_fuite_benchmark.py` (repository root) | leak audit of the 210 questions; 16 questions name both arguments of an expert relation in another form |
| `179_trace_run18` | `179_trace_missing_expert_relations.py` | status of the 26 expert relations (`fig:lb2019`) |
| `180_passage_panel` | `180_passage_seen_by_panel.py` | evidence passage given to the verifying-LLMs |
| `182_mesures` | `182_name_families_run18.py`, subcommand `measure` | name families and the relations that lead to slope failure |
| `186_portee` | `186_scope_markers.py` | markers of condition, modality and scope (`tab:markers`) |
| `189_elements_article` | `189_manuscript_elements.py` | `tab:graphs` and the counts of the results |
| `190_controles_et_coherence` | `190_controls_and_consistency.py` | the 25 ambiguous copies and the replay of the verdicts (362 of 362 identical) |
| `191_regles_et_sensibilite` | `191_rules_and_sensitivity.py` | rates of `tab:stress` by type of corruption and their sensitivity to the decision rule |
| `192_infos_manquantes` | `192_missing_information.py` | information that the run files do not record |
| `192_contenu_slope_failure` | summary tables of 4 November 2026, written by a script that is not in this repository | relations of the cleaned graph by type, and the causes, descriptors and settings of mass transport deposits |
| `193_couverture_descripteurs` | `193_descriptor_coverage.py` | coverage of the 13 expert descriptors: 13 of 13, 12 of 13 in Tier 1 |

`results/run18/kg_final/article_metrics.json`, written by `pipeline/07_final_metrics.py` during the run, reports 12 of 13 expert descriptors (11 in Tier 1) and a recall on 34 reference edges. The manuscript uses neither value: `07_final_metrics.py` maps *stratified* to *layered* before counting, whereas the expert graph lists them as two descriptors, and the manuscript compares the graph with 26 expert relations only. `193_couverture_descripteurs/_resume.txt` gives the direct count used in the manuscript.
