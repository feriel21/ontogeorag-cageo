# run18 audit scripts

Read-only CPU scripts run on the recorded run18 outputs after the run. No model is called and no run output is modified. Each script refuses to overwrite its output folder.

The files were copied from the cluster; `PROVENANCE.tsv` records the SHA-256 of each copy. After the run, we translated them into English and renamed them: comments, docstrings, printed messages, identifiers and output file names. `tools/check_code_unchanged.py` with `--allow-strings --allow-renames` confirmed that the syntax tree is unchanged apart from string constants and consistent renames. `PROVENANCE.tsv` records the original name and both versions. Two numbers are used twice (189 and 190), because the scripts were numbered in the order they were written.

Some scripts read files written by the cleaning scripts 184, 187 and 188 (`graphe_nettoye.json`, `graphe_avec_provenance.json`, the folder `188_nettoyage_v2`). These inputs keep their French names, because the names are written by the scripts 184, 187 and 188, which must stay byte-identical (the postprocess manifest records their checksum); `docs/run18_cleaning_outputs.md` explains each of them.

The output names changed with the translation. A rerun writes the new names, and some scripts read the outputs of others, so we rerun them in this order:

1. `181_decision_rules.py` before `191_rules_and_sensitivity.py`, which reads `decision_rules.json` in the self-audit folder;
2. `186_scope_markers.py` before `189_manuscript_elements.py`, which reads the folder `186_scope`.

`190_controls_and_consistency.py` lists the files of the folders given by `--folders`. On the authors' copy, two of these folders were written before the translation, under the names `output/audit/181_regles_llama` and `output/audit/181_regles_mistral`. A folder that does not exist is reported as `[absent]`.

| Script | What it computes |
|---|---|
| `179_trace_missing_expert_relations.py` | For each of the 26 expert relations missing from the run18 graph, the step where it is lost: no sentence in the corpus, passage never retrieved, retrieved but not extracted, extracted in another form, or removed at a later step |
| `180_passage_seen_by_panel.py` | Whether the evidence sentence of each relation is in the text given to the verifying-LLMs, crossed with the panel decision |
| `181_decision_rules.py` | Three decision rules applied to the verdicts of the self-audit records; only the evidence-only rule is used in the manuscript, the other two were not reproduced by 191 |
| `182_name_families_run18.py` | Proposed name family for each entity, validated by hand into `configs/entity_families_run18.csv` |
| `183_compare_passes.py` | Distinct relations of a replayed deterministic pass compared with the recorded pass |
| `185_relation_trace_sheet.py` | Trace sheet of one relation, from the query to the delivered graph, with the origin of each item |
| `186_scope_markers.py` | Markers of condition, modality and scope in the evidence sentences, crossed with tier and verdicts |
| `189_manuscript_elements.py` | Delivered and cleaned graphs, rejection and retention rates of the self-audit with Wilson intervals, ambiguous corrupted records |
| `189_vocabulary_frequency.py` | Number of papers and passages that contain each term of the extraction vocabulary |
| `190_controls_and_consistency.py` | Consistency of tiers, verdicts and decisions in both graphs; inventory and join of the self-audit files |
| `190_retrieval_replay_run18.py` | Replay of the run18 retrieval without any LLM: questions dropped by the BM25 threshold, effect of the cross-encoder |
| `191_rules_and_sensitivity.py` | Decision rules recomputed from the joined verdicts, rates per corruption type, sensitivity without the ambiguous records |
| `192_missing_information.py` | Evidence check applied to the delivered and cleaned graphs, contributing papers, settings of the verifying-LLMs, counts of each stage |
| `192_graph_content.py` | Content of a graph about one target found by name substrings (mass transport deposits by default): relations by type, descriptors, causes and triggers, context |
| `193_descriptor_coverage.py` | Coverage of the 13 expert descriptors under three string-matching rules, and the rule used by `pipeline/07_final_metrics.py` |
