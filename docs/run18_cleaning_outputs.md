# run18 cleaning scripts and outputs: reading guide

The three scripts that turn the delivered run18 graph into the cleaned graph are written in French: `184_trace_requetes_et_pertes.py`, `187_provenance_articles.py` and `188_nettoyage_familles.py`. Their code, comments and messages are not translated, because the postprocess manifest (`results/run18/postprocess/postprocess_manifest.txt`) records their SHA-256 and `tools/verify_provenance.py --postprocess` checks it. Their outputs in `results/run18/postprocess/` keep French file and column names for the same reason.

This guide gives, in English, what each script does and what each output column means. All three scripts are read-only on the run outputs, run on a CPU, call no model and refuse to overwrite their output folder. `workflows/postprocess_run18.sh` runs them in order.

## Chain

```
delivered graph (results/run18/kg_final/flat_run18_enforced_dedup.json)
  -> 184  trace of every extraction            -> postprocess/184_trace/
  -> 187  source paper of every relation       -> postprocess/187_provenance/  (graph with provenance)
  -> 188  rule-based cleaning, name families   -> postprocess/188_nettoyage/   (cleaned graph)
```

## 184 — suspicious queries, losses and evidence provenance

- **A. Suspicious queries.** Finds the queries of `configs/descriptor_queries_run18.jsonl` whose text and anchor together name both arguments of one of the 26 expert edges. Matching accepts the exact term, plurals (+s, +es), hyphen or space, and *mtd*, *mtds*, *mass-transport deposit(s)* for *mass transport deposit*. The script stops unless exactly 16 queries are found. The class of each query is a reading by hand, written in the script with its reason.
- **B. Losses, record by record.** Replays each line of `pass_X/verified_triples.jsonl` through the functions of `pipeline/04_clean_validate_run18.py`, then the name map, the rules of 05a, the lexicon rule of 04c and the fusion key of 06. Duplicates are followed to the relation that absorbs them. Every replayed count is compared with the files of the run and printed as OK or ECHEC (failure).
- **C. Fate after fusion.** Panel decision, quarantine, and tier in the delivered graph, for each merged relation.
- **D. Provenance.** Looks for each evidence sentence in its five passages (exact, partial, or not found).
- **E. Sensitivity.** Delivered graph without the relations obtained only from suspicious queries.

| File | Content |
|---|---|
| `requetes_suspectes.tsv` | the suspicious queries: `index_requete` query index, `strategie` strategy, `focus` anchor, `requete` query text, `classe` class, `motif` reason for the class, `arete_sujet` / `arete_relation` / `arete_objet` expert edge named, `forme_vue_sujet` / `forme_vue_objet` surface form found |
| `suivi_extractions.tsv` | one line per extraction (see columns below) |
| `extractions_requetes_suspectes.tsv` | same columns, extractions from suspicious queries only |
| `pertes_par_etape.tsv` | losses per step: `passe` pass, `etape` step, `raison` reason, `n_extractions` count |
| `relations_fusionnees.tsv` | one line per merged relation (see columns below) |
| `normalisations_ambigues.tsv` | out-of-vocabulary objects with several candidate descriptors: `objet_brut` raw object, `candidats` candidates, `choix_du_run` choice made by the run, `origine_du_choix` where the choice was read, `choix_rejoue` choice of the replay |
| `sensibilite.json` | `graphe_livre` delivered graph, `sans_relations_uniquement_R` without relations obtained only from R-class queries, `retirees_uniquement_R` those relations, `sans_relations_uniquement_16` without relations obtained only from the 16 queries, `avertissement` warning |
| `_resume.txt` | printed summary and checks |

Columns of `suivi_extractions.tsv`: `passe` pass (a or b), `ligne` line in `verified_triples.jsonl`, `index_requete` query index, `suspecte` suspicious query, `classe_requete` query class, `strategie` strategy, `requete` query, `brut_sujet` / `brut_relation` / `brut_objet` raw triple, `brut_arete_reference` raw triple is an expert edge, `verdict_qwen` self-check verdict, `etape_sortie` step where the extraction leaves the chain, `raison` reason, `absorbe_par_ligne` line of the record that absorbs this duplicate, `cle_sujet` / `cle_relation` / `cle_objet` normalised key, `arete_reference` key is an expert edge, `statut_final_de_la_relation` final status of the relation, `preuve_dans_ses_passages` evidence found in its passages, `preuve_chunks` passages, `preuve_articles` papers, `articles_recuperes` papers retrieved.

Columns of `relations_fusionnees.tsv`: `sujet` subject, `relation`, `objet` object, `tier_06` tier at fusion, `origine` pass(es) of origin, `decision_panel` panel decision, `statut_final` final status (`livre Tier 1`, `livre Tier 2` delivered in tier 1 or 2; `quarantaine` quarantined), `arete_reference` expert edge, `requete_affichee_suspecte` displayed query is suspicious, `n_extractions_retenues` extractions kept, `n_par_requetes_suspectes` / `n_par_requetes_non_suspectes` from suspicious / other queries, `provenance_requetes` queries, `provenance_requetes_R` R-class queries, `non_suspectes_extraites_mais_ecartees` other queries extracted but set aside, `strong_absorbe_comme_doublon` STRONG record absorbed as a duplicate, `verdict` self-check verdicts.

Coded values: `etape_sortie` = `04_verdict` (self-check), `04_validation` (vocabulary and validation rules), `doublon_04` (duplicate), empty (enters the fusion); `raison` = `verif_not_supported`, `type_constraint`, `entity_too_long_subject`, `entity_too_long_object`, `self_loop`, `meme cle avant fusion des noms` (same key before name merging), `meme cle apres fusion des noms` (same key after name merging), `entre_en_fusion` (enters the fusion). Query classes: `R-reponse` the expected answer is written in the question, `R-presuppose` the question presupposes the expert relation, `R-candidat` the subject is one candidate among others, `V-vocabulaire` both terms occur but the question does not link them.

## 187 — paper, reference and section of each relation

For each delivered relation, the evidence sentence is located in the index (exact, or partial when one of three 40-character windows matches), then mapped to its paper (`data/pdf_to_ref.tsv`, `data/lb2019_41_refs.txt`). The section is **deduced**: the last heading found in the text before the sentence. The PDF layout is lost, so this is an inference, not a field of the file. A probable figure caption is flagged the same way.

| File | Content |
|---|---|
| `provenance_par_relation.tsv` | one line per delivered relation |
| `provenance_detail.tsv` | one line per (relation, passage where the evidence is located) |
| `graphe_avec_provenance.json` | copy of the delivered graph with a `provenance` field per relation |
| `references_non_resolues.txt` | passages without a reference (empty) |
| `_resume.txt` | printed summary |

Columns: `sujet` subject, `objet` object, `tier`, `decision_panel` panel decision, `n_articles` number of papers, `articles` papers, `phrase_affichee_*` for the displayed evidence sentence (`statut` status, `article` paper, `ref_n` reference number, `passage` passage position in the paper, `section_deduite` deduced section, `type_titre` heading type, `legende_figure` probable caption, `apres_references` located after the reference list), `preuve` evidence sentence; in the detail file, `doc_id` file, `reference_courte` short reference, `reference` full reference, `doi`, `chunk_id` passage, `legende_figure_probable` probable caption, `statut` exact or partial, `origine` source of the sentence (`phrase affichee` displayed sentence, or `extraction a:N` / `extraction b:N` another extraction of pass a or b, line N).

## 188 — rule-based cleaning and name families

Rules fixed in the script; each removed relation lists its rules. A *source sentence* is an evidence sentence located by 187: the displayed sentence and those of the other extractions that support the relation.

- **R1 reference list**: every source sentence is an entry of the reference list.
- **R2 local label**: the subject or object is a label specific to one paper (*am1*, *unit a*, *mass transport deposit 5*), not a geological class.
- **R3 missing subject**: no source sentence names the subject, its head word or a synonym from its family.
- Flagged, not removed: **S1** named place or formation, **S2** the displayed sentence starts with a comment of the model instead of a quotation, **S3** probable figure caption, **S4** subject absent but a pronoun in the first three words of a source sentence.

Families: each entity receives its family from `configs/entity_families_run18.csv`; an entity absent from the table keeps its own name. Relations are grouped by (subject family, relation, object family). Three family labels of the table are in French: `contexte de marge` margin setting, `position bathymetrique` bathymetric position, `systeme sedimentaire` sedimentary system.

| File | Content |
|---|---|
| `graphe_nettoye.json` | cleaned graph (281 relations); each relation adds `famille_sujet` and `famille_objet` (subject and object family); `meta_188` gives `source` input file, `source_sha256`, `regles` rules applied, `n_avant` / `n_apres` relations before and after |
| `relations_retirees.tsv` | removed relations: `n_phrases_sources` number of source sentences, `preuve` evidence, `regles` rules |
| `relations_signalees.tsv` | flagged relations: `signalements` flags, `retiree` removed |
| `relations_par_famille.tsv` | groups: `famille_sujet`, `relation`, `famille_objet`, `n_relations`, `n_tier1`, `n_articles` papers, `membres` members |
| `entites_hors_table.tsv` | entities absent from the family table: `entite`, `n_relations` |
| `_resume.txt` | printed summary |

In the `provenance` field of both graphs: `method` method of 187 (in French), `n_articles` number of papers, `sources` list with `ref_n`, `reference`, `doi`, `doc_id`, `chunk_id`, `passage` (position in the paper, e.g. 28/62), `section_deduite` deduced section, `legende_figure_probable` probable caption, `statut` exact or partial, `origine` as in 187.
