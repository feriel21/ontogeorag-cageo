# run18 audit scripts

Read-only CPU scripts run on the recorded run18 outputs after the run. No model is called and no run output is modified. Each script refuses to overwrite its output folder.

The files are copied unchanged from the cluster; `PROVENANCE.tsv` records the origin and SHA-256 of each one. Their comments, messages and output file names are in French. Two numbers are used twice (189 and 190), because the scripts were numbered in the order they were written.

| Script | What it computes |
|---|---|
| `179_trace_relations_absentes.py` | For each of the 26 expert relations missing from the run18 graph, the step where it is lost: no sentence in the corpus, passage never retrieved, retrieved but not extracted, extracted in another form, or removed at a later step |
| `180_passage_vu_par_le_panel.py` | Whether the evidence sentence of each relation is in the text given to the verifying-LLMs, crossed with the panel decision |
| `181_regles_decision.py` | Three decision rules applied to the verdicts of the self-audit records; only the evidence-only rule is used in the manuscript, the other two were not reproduced by 191 |
| `182_familles_noms_run18.py` | Proposed name family for each entity, validated by hand into `configs/entity_families_run18.csv` |
| `183_comparer_passes.py` | Distinct relations of a replayed deterministic pass compared with the recorded pass |
| `185_fiche_triplet.py` | Traceability sheet of one relation, from the query to the delivered graph, with the origin of each item |
| `186_marqueurs_portee.py` | Markers of condition, modality and scope in the evidence sentences, crossed with tier and verdicts |
| `189_elements_article.py` | Delivered and cleaned graphs, rejection and retention rates of the self-audit with Wilson intervals, ambiguous corrupted records |
| `189_frequence_vocabulaire.py` | Number of papers and passages that contain each term of the extraction vocabulary |
| `190_controles_et_coherence.py` | Consistency of tiers, verdicts and decisions in both graphs; inventory and join of the self-audit files |
| `190_rejeu_recherche_run18.py` | Replay of the run18 retrieval without any LLM: questions dropped by the BM25 threshold, effect of the cross-encoder |
| `191_regles_et_sensibilite.py` | Decision rules recomputed from the joined verdicts, rates per corruption type, sensitivity without the ambiguous records |
| `192_infos_manquantes.py` | Evidence check applied to the delivered and cleaned graphs, contributing papers, settings of the verifying-LLMs, counts of each stage |
| `193_couverture_descripteurs.py` | Coverage of the 13 expert descriptors under three string-matching rules, and the rule used by `pipeline/07_final_metrics.py` |
