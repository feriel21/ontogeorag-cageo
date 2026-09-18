# Provenance of the published code

The delivered run is run16E. Its manifest, `results/run16E/manifest.json`, records commit `9c03730` (clean tree: True), and for each step the script, its sha256 and its command line. That commit is not published because its tree contains the corpus PDFs; the published files are tied to it by the checksums below. `tools/verify_provenance.py` checks them.

In the options, `$OUT` is the output directory of the run and `HF:repo@revision` a Hugging Face model snapshot.

## Recorded steps

| # | Step | Script | sha256 | Options |
|---|---|---|---|---|
| 1 | index | `49_build_index_v2.py` | `fe35d2a60642` | `--pdf-dir data/full_corpus --outdir $OUT/index --chunk-size 2000 --overlap 200 --expect-docs 41` |
| 2 | extract_pass_a | `pipeline/02_extract_triples.py` | `f2555decd4de` | `--index-dir $OUT/index --schema configs/ontology_schema.json --queries configs/descriptor_queries_v2.jsonl --output $OUT/pass_a/raw_triples.jsonl --model HF:Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 --backend hf --top-k 5 --bm25-topn 20 --min-bm25 2.0 --max-chars 2800 --temperature 0.0 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 3 | extract_pass_b | `pipeline/02_extract_triples.py` | `f2555decd4de` | `--index-dir $OUT/index --schema configs/ontology_schema.json --queries configs/descriptor_queries_v2.jsonl --output $OUT/pass_b/raw_triples.jsonl --model HF:Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 --backend hf --top-k 5 --bm25-topn 20 --min-bm25 2.0 --max-chars 2800 --temperature 0.3 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 4 | repair_provenance | `52_repair_provenance.py` | `19e0101d3d0d` | `--chunks $OUT/index/chunks.jsonl --triples $OUT/pass_a/raw_triples.jsonl $OUT/pass_b/raw_triples.jsonl --outdir $OUT --expect-chunks 5` |
| 5 | verify_pass_a | `53_verify_repaired.py` | `e1fcedb7452c` | `--input $OUT/pass_a/triples_repaired.jsonl --output $OUT/pass_a/verified_triples.jsonl --model HF:Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 --backend hf --max-chunks 5 --chunk-chars 2000` |
| 6 | verify_pass_b | `53_verify_repaired.py` | `e1fcedb7452c` | `--input $OUT/pass_b/triples_repaired.jsonl --output $OUT/pass_b/verified_triples.jsonl --model HF:Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28 --backend hf --max-chunks 5 --chunk-chars 2000` |
| 7 | clean_pass_a | `pipeline/04_clean_validate.py` | `21383881fdf0` | `--input $OUT/pass_a/verified_triples.jsonl --outdir $OUT/pass_a --verif-policy normal --cluster-threshold 0.06` |
| 8 | rulecanon_pass_a | `analysis_suite/05a_rule_canonicalize.py` | `8f5823443a71` | `--input $OUT/pass_a/canonical_triples_v5.jsonl` |
| 9 | lexicon_pass_a | `analysis_suite/04c_lexicon_enforce.py` | `7c311aaf90f7` | `--input $OUT/pass_a/canonical_triples_v5.rulecanon.jsonl --enforce` |
| 10 | clean_pass_b | `pipeline/04_clean_validate.py` | `21383881fdf0` | `--input $OUT/pass_b/verified_triples.jsonl --outdir $OUT/pass_b --verif-policy normal --cluster-threshold 0.06` |
| 11 | rulecanon_pass_b | `analysis_suite/05a_rule_canonicalize.py` | `8f5823443a71` | `--input $OUT/pass_b/canonical_triples_v5.jsonl` |
| 12 | lexicon_pass_b | `analysis_suite/04c_lexicon_enforce.py` | `7c311aaf90f7` | `--input $OUT/pass_b/canonical_triples_v5.rulecanon.jsonl --enforce` |
| 13 | fusion | `pipeline/06_tiered_fusion.py` | `f0c45d490de8` | `--iter-a $OUT/pass_a/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl --iter-b $OUT/pass_b/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl --output $OUT/kg/tiered_kg_run16E.json` |
| 14 | m4_verify_llama | `m4/m4_verify.py` | `7350d11274a4` | `--kg $OUT/kg/tiered_kg_run16E.json --index $OUT/index --output $OUT/m4/llama --model HF:meta-llama/Llama-3.1-8B-Instruct@0e9e39f249a16976918f6564b8830bc894c89659` |
| 15 | m4_aggregate_llama | `m4/m4_aggregate.py` | `148d44c13502` | `--verdicts $OUT/m4/llama/m4_verdicts.jsonl --output $OUT/m4/llama` |
| 16 | m4_verify_mistral | `m4/m4_verify.py` | `7350d11274a4` | `--kg $OUT/kg/tiered_kg_run16E.json --index $OUT/index --output $OUT/m4/mistral --model HF:mistralai/Mistral-7B-Instruct-v0.2@63a8b081895390a26e140280378bc85ec8bce07a` |
| 17 | m4_aggregate_mistral | `m4/m4_aggregate.py` | `148d44c13502` | `--verdicts $OUT/m4/mistral/m4_verdicts.jsonl --output $OUT/m4/mistral` |
| 18 | m4_panel | `m4/m4_panel.py` | `e659b652e999` | `--judges llama=$OUT/m4/llama/m4_verdicts.jsonl mistral=$OUT/m4/mistral/m4_verdicts.jsonl --output $OUT/m4_panel` |
| 19 | m4_integrate | `m4/m4_integrate_tiers.py` | `7bc55b09b2ad` | `--kg $OUT/kg/tiered_kg_run16E.json --decisions $OUT/m4_panel/m4_panel_decisions.jsonl --output $OUT/m4_panel` |
| 20 | schema_enforce | `analysis_suite/04b_schema_enforce.py` | `63f294bb4a88` | `--kg $OUT/m4_panel/tiered_kg_m4.json --outdir $OUT/kg_enforced --default-type Geological_Object --descriptor-fallback Environmental_Control` |
| 21 | flatten | `analysis_suite/24_kg_to_flat.py` | `26f70b35f60c` | `--kg $OUT/kg_enforced/tiered_kg_enforced.json --include tier1,tier2 --dedup --out $OUT/kg_final/flat_run16E_enforced_dedup.json` |
| 22 | metrics | `pipeline/07_final_metrics.py` | `de92306533c7` | `--kg $OUT/kg_final/flat_run16E_enforced_dedup.json --output $OUT/kg_final/article_metrics.json` |

## Stage scripts

| Script | sha256 | Same on disk | Same in PROVENANCE.tsv |
|---|---|---|---|
| `49_build_index_v2.py` | `fe35d2a60642182a163ea09f36b9f7f34a9d34c502349e7813bb33e106192c87` | yes | yes |
| `52_repair_provenance.py` | `19e0101d3d0df548fa4425069c2f04ace0a806779b3b7cfc72444e25b07a5fca` | yes | yes |
| `53_verify_repaired.py` | `e1fcedb7452ccb8381d23de34c9e1e76b7e358d136c07d1e1c9618cd86f3ce25` | yes | yes |
| `analysis_suite/04b_schema_enforce.py` | `63f294bb4a88a3fb6394b894aa3a68ec265c6dd44be7a73a4ee2a1c812c347f9` | yes | yes |
| `analysis_suite/04c_lexicon_enforce.py` | `7c311aaf90f7f457354b2b646e395b8a41c873edc9ff380d2202b7af795332ed` | yes | yes |
| `analysis_suite/05a_rule_canonicalize.py` | `8f5823443a7153e47adcf3583ee10dfb3f2b1a9eaae532cb7fee12a1e6b6200d` | yes | yes |
| `analysis_suite/24_kg_to_flat.py` | `26f70b35f60c9b53238dea963d6764f04e6db563916a8774c73426eeb0d92d1f` | yes | yes |
| `m4/m4_aggregate.py` | `148d44c135026e510a6a5eae5cfc69c994e825c2afb12937fd66d1345793af71` | yes | yes |
| `m4/m4_integrate_tiers.py` | `7bc55b09b2addb6e51c2c49d615aacce13607d751d1a9dca9c17ffad8369ae3b` | yes | yes |
| `m4/m4_panel.py` | `e659b652e99977ffdf5cf4d263a00bbb0b5d9448b0704b3f3744cf034b11c94e` | yes | yes |
| `m4/m4_verify.py` | `7350d11274a4a9e39232d5fbf7f898b97d59e7c86d285b20ae1420f06b1a788f` | yes | yes |
| `pipeline/02_extract_triples.py` | `f2555decd4de68a7cc8c5630c944990a256d8f14cf5ad4809ef0dd03dd7a5eb6` | yes | yes |
| `pipeline/04_clean_validate.py` | `21383881fdf0ce953ceb63cdc7e404f5a34c4ec5917e373b31d74ba20645b276` | yes | yes |
| `pipeline/06_tiered_fusion.py` | `f0c45d490de8d756d56e7e5d078e5cfa93082511d4b2de1a26bcac2a8c9839a9` | yes | yes |
| `pipeline/07_final_metrics.py` | `de92306533c7c710be8e854235bf8b576baa91585f863936fa6ff6d915651b49` | yes | yes |

## Files listed in PROVENANCE.tsv

Files not present in this repository were not published (for example the delivered graph and outputs containing text from the publications).

| File | Origin | sha256 | Check | Present |
|---|---|---|---|---|
| `49_build_index_v2.py` | git:9c03730:49_build_index_v2.py | `fe35d2a60642` | sha256 = manifest | yes |
| `52_repair_provenance.py` | git:9c03730:52_repair_provenance.py | `19e0101d3d0d` | sha256 = manifest | yes |
| `53_verify_repaired.py` | git:9c03730:53_verify_repaired.py | `e1fcedb7452c` | sha256 = manifest | yes |
| `58v2_freeze_benchmark.py` | git:9c03730:58v2_freeze_benchmark.py | `326a1eb826b2` | not in manifest | yes |
| `61_extend_queries.py` | git:9c03730:61_extend_queries.py | `8736049336c4` | not in manifest | yes |
| `analysis_suite/04b_schema_enforce.py` | git:9c03730:analysis_suite/04b_schema_enforce.py | `63f294bb4a88` | sha256 = manifest | yes |
| `analysis_suite/04c_lexicon_enforce.py` | git:9c03730:analysis_suite/04c_lexicon_enforce.py | `7c311aaf90f7` | sha256 = manifest | yes |
| `analysis_suite/05a_rule_canonicalize.py` | git:9c03730:analysis_suite/05a_rule_canonicalize.py | `8f5823443a71` | sha256 = manifest | yes |
| `analysis_suite/24_kg_to_flat.py` | git:9c03730:analysis_suite/24_kg_to_flat.py | `26f70b35f60c` | sha256 = manifest | yes |
| `analysis_suite/kg_io.py` | git:9c03730:analysis_suite/kg_io.py | `10bea6d38fd7` | not in manifest | yes |
| `configs/ablation_50q.jsonl` | git:9c03730:configs/ablation_50q.jsonl | `f6cf2238526f` | config au commit | no |
| `configs/archive/benchmark_v1.json` | git:9c03730:configs/archive/benchmark_v1.json | `198554c48729` | config au commit | no |
| `configs/archive/ontology_v1.json` | git:9c03730:configs/archive/ontology_v1.json | `e3497b8c5b50` | config au commit | no |
| `configs/benchmark/benchmark_v2.json` | git:9c03730:configs/benchmark/benchmark_v2.json | `1f8b19c37780` | config au commit | no |
| `configs/benchmark/benchmark_v3.json` | git:9c03730:configs/benchmark/benchmark_v3.json | `515033ed9189` | config au commit | yes |
| `configs/deprecated/lb_extended_benchmark.json` | git:9c03730:configs/deprecated/lb_extended_benchmark.json | `d172467ab0b4` | config au commit | yes |
| `configs/descriptor_queries.jsonl` | git:9c03730:configs/descriptor_queries.jsonl | `0febd2a38124` | config au commit | yes |
| `configs/descriptor_queries_v2.jsonl` | git:9c03730:configs/descriptor_queries_v2.jsonl | `f4269b33fc47` | config au commit | yes |
| `configs/descriptor_queries_v2.meta.json` | git:9c03730:configs/descriptor_queries_v2.meta.json | `4ef329e7078f` | config au commit | yes |
| `configs/descriptor_synonyms_v2.json` | git:9c03730:configs/descriptor_synonyms_v2.json | `d72bd70bf702` | config au commit | no |
| `configs/lb_26_original.json` | git:9c03730:configs/lb_26_original.json | `54593857bb12` | config au commit | yes |
| `configs/lb_26_original_canonrel.json` | git:9c03730:configs/lb_26_original_canonrel.json | `96bb3c5e1eb3` | config au commit | no |
| `configs/lb_dev.json` | git:9c03730:configs/lb_dev.json | `f253a7762fb4` | config au commit | yes |
| `configs/lb_reference_edges.json` | git:9c03730:configs/lb_reference_edges.json | `d172467ab0b4` | config au commit | yes |
| `configs/lb_reference_edges_original26.json` | git:9c03730:configs/lb_reference_edges_original26.json | `c3a8324eacab` | config au commit | yes |
| `configs/lb_test.json` | git:9c03730:configs/lb_test.json | `2583139441b2` | config au commit | yes |
| `configs/ontology_run15.json` | git:9c03730:configs/ontology_run15.json | `094bfe747e32` | config au commit | no |
| `configs/ontology_schema.json` | git:9c03730:configs/ontology_schema.json | `1c4bc885ba4f` | config au commit | yes |
| `configs/ontology_v1_consolidated.json` | git:9c03730:configs/ontology_v1_consolidated.json | `5ef4d806d861` | config au commit | no |
| `configs/run13_protocol_declaration.json` | git:9c03730:configs/run13_protocol_declaration.json | `02ed5acbf266` | config au commit | no |
| `configs/type_overrides.json` | git:9c03730:configs/type_overrides.json | `f5a9d62323a9` | config au commit | no |
| `data/README.md` | git:9c03730:data/README.md | `3495b783dfa3` | not in manifest | yes |
| `data/lb2019_41_refs.txt` | git:9c03730:data/lb2019_41_refs.txt | `d433a2cee75a` | not in manifest | yes |
| `m4/m4_aggregate.py` | git:9c03730:m4/m4_aggregate.py | `148d44c13502` | sha256 = manifest | yes |
| `m4/m4_config.py` | git:9c03730:m4/m4_config.py | `e8c930b55219` | not in manifest | yes |
| `m4/m4_direction_check.py` | git:9c03730:m4/m4_direction_check.py | `80856373d570` | not in manifest | yes |
| `m4/m4_integrate_tiers.py` | git:9c03730:m4/m4_integrate_tiers.py | `7bc55b09b2ad` | sha256 = manifest | yes |
| `m4/m4_metrics.py` | git:9c03730:m4/m4_metrics.py | `0949eb018e79` | not in manifest | yes |
| `m4/m4_negatives.py` | git:9c03730:m4/m4_negatives.py | `26cf81a9b311` | not in manifest | yes |
| `m4/m4_panel.py` | git:9c03730:m4/m4_panel.py | `e659b652e999` | sha256 = manifest | yes |
| `m4/m4_verify.py` | git:9c03730:m4/m4_verify.py | `7350d11274a4` | sha256 = manifest | yes |
| `pipeline/02_extract_triples.py` | git:9c03730:pipeline/02_extract_triples.py | `f2555decd4de` | sha256 = manifest | yes |
| `pipeline/03_verify_triples.py` | git:9c03730:pipeline/03_verify_triples.py | `d009df2bcbd6` | not in manifest | yes |
| `pipeline/04_clean_validate.py` | git:9c03730:pipeline/04_clean_validate.py | `21383881fdf0` | sha256 = manifest | yes |
| `pipeline/06_tiered_fusion.py` | git:9c03730:pipeline/06_tiered_fusion.py | `f0c45d490de8` | sha256 = manifest | yes |
| `pipeline/07_final_metrics.py` | git:9c03730:pipeline/07_final_metrics.py | `de92306533c7` | sha256 = manifest | yes |
| `pipeline/__init__.py` | git:9c03730:pipeline/__init__.py | `e3b0c44298fc` | not in manifest | yes |
| `pipeline/rag/__init__.py` | git:9c03730:pipeline/rag/__init__.py | `e3b0c44298fc` | not in manifest | yes |
| `pipeline/rag/constants.py` | git:9c03730:pipeline/rag/constants.py | `08b6bcca1aa8` | not in manifest | yes |
| `pipeline/rag/hybrid_retriever.py` | git:9c03730:pipeline/rag/hybrid_retriever.py | `9eb60ed0f656` | not in manifest | yes |
| `pipeline/rag/llm_hf.py` | git:9c03730:pipeline/rag/llm_hf.py | `c7331891a8e6` | not in manifest | yes |
| `reference/reference_kg.json` | git:9c03730:reference/reference_kg.json | `5ee55a40749a` | not in manifest | yes |
| `results/run16/manifest.json` | file:output/run16/manifest.json | `98f612513284` | manifeste de run | yes |
| `results/run16B/manifest.json` | file:output/run16B/manifest.json | `d00cfe4a4f21` | manifeste de run | yes |
| `results/run16C/manifest.json` | file:output/run16C/manifest.json | `6ea7fd9d8829` | manifeste de run | yes |
| `results/run16D/manifest.json` | file:output/run16D/manifest.json | `24c7ba6f72a0` | manifeste de run | yes |
| `results/run16E/manifest.json` | file:output/run16E/manifest.json | `bc5e8c163675` | manifeste de run | yes |
| `scripts/analysis/100_family_corroboration.py` | file:100_family_corroboration.py | `8e36efc5ca7d` | sha = inventaire 148 | no |
| `scripts/analysis/93_graph_validation_run16E.py` | file:93_graph_validation_run16E.py | `8aef4e98f7c4` | sha = inventaire 148 | no |
| `scripts/analysis/94_failure_modes.py` | file:94_failure_modes.py | `0ccb4b99aa14` | sha = inventaire 148 | no |
| `scripts/analysis/95d_fermer_partition_descripteurs.py` | file:95d_fermer_partition_descripteurs.py | `8a208a7bf4ed` | sha = inventaire 148 | no |
| `scripts/analysis/95g_derniers_controles.py` | file:95g_derniers_controles.py | `ba118cf73078` | sha = inventaire 148 | no |
| `scripts/analysis/95h_mesurer_graphe_reference.py` | file:95h_mesurer_graphe_reference.py | `5e646368e9a7` | sha = inventaire 148 | no |
| `scripts/analysis/95i_provenance_benchmark26.py` | file:95i_provenance_benchmark26.py | `1bf1ae6d1f6c` | sha = inventaire 148 | no |
| `scripts/analysis/95m_argv_manifeste_et_ciblage.py` | file:95m_argv_manifeste_et_ciblage.py | `0f23715be05e` | sha = inventaire 148 | no |
| `scripts/analysis/95p_commands.sh` | file:output/audit/95p_commands.sh | `df7e6feba6ea` | sha = inventaire 148 | no |
| `scripts/analysis/95p_comparaison_controlee_modeles.py` | file:95p_comparaison_controlee_modeles.py | `1e0067b99bd8` | sha = inventaire 148 | no |
| `scripts/analysis/96_descriptor_frequencies_run16E.py` | file:96_descriptor_frequencies_run16E.py | `ce5a41da4f3b` | sha = inventaire 148 | no |
| `scripts/evaluation/106b_join_resolve.py` | file:106b_join_resolve.py | `b06f634c322a` | sha = inventaire 148 | no |
| `scripts/evaluation/106c_join_verify.py` | file:106c_join_verify.py | `65937ec49580` | sha = inventaire 148 | no |
| `scripts/evaluation/106d_metrics_stratified.py` | file:106d_metrics_stratified.py | `4c516fc14aa9` | sha = inventaire 148 | no |
| `scripts/evaluation/143_benchmark_leakage_and_retrieval_recall.py` | file:143_benchmark_leakage_and_retrieval_recall.py | `ac20f9e40c38` | sha = inventaire 148 | no |
| `scripts/evaluation/144_replay_fidelity_and_sentence_cooccurrence.py` | file:144_replay_fidelity_and_sentence_cooccurrence.py | `07c3acd722c0` | sha = inventaire 148 | no |
| `scripts/evaluation/145_graph_with_located_provenance.py` | file:145_graph_with_located_provenance.py | `cf68da200e57` | sha = inventaire 148 | no |
| `scripts/evaluation/95_nli_verifier_substitution.py` | file:95_nli_verifier_substitution.py | `603d9e088638` | sha = inventaire 148 | no |
| `scripts/figures/95q_fig_pipeline.py` | file:95q_fig_pipeline.py | `14774202c2ca` | sha = inventaire 148 | no |
| `scripts/figures/95y_fig_tradeoff.py` | file:95y_fig_tradeoff.py | `81cf49a5a583` | sha = inventaire 148 | no |
| `scripts/figures/98_make_figures.py` | file:98_make_figures.py | `c865fec42721` | sha = inventaire 148 | no |
| `scripts/paper/142_extract_ablation_csv.py` | file:142_extract_ablation_csv.py | `aacd818089a8` | sha = inventaire 148 | no |
| `scripts/paper/67_paper_results.py` | file:67_paper_results.py | `11e961d825f5` | sha = inventaire 148 | no |
| `scripts/paper/68_classify_vocabulary.py` | file:68_classify_vocabulary.py | `c10b02cc3a81` | sha = inventaire 148 | no |
| `scripts/tools/48e_preflight_41.py` | file:48e_preflight_41.py | `623900990a08` | sha = inventaire 148 | no |
| `scripts/tools/50_extract_lb26_reference.py` | file:50_extract_lb26_reference.py | `28d356697f41` | sha = inventaire 148 | no |
| `slurm/run16_full.sh` | git:9c03730:slurm/run16_full.sh | `db12a9fc265e` | not in manifest | yes |
| `scripts/paper/154_table5_ablation.py` | file:154_table5_ablation.py | `bccc3b64dc98` | Table 5 identique au manuscrit et a 67 | no |
| `scripts/paper/table5_ablation.py` | new (repository tooling, English) | `09876e76133e` | validated on cluster | yes |
| `tools/check_code_unchanged.py` | new (repository tooling, English) | `a5fb37fbcded` | validated on cluster | yes |
| `tools/replay_cpu_steps.py` | new (repository tooling, English) | `0b2ffc85507b` | validated on cluster | yes |
| `tools/verify_provenance.py` | new (repository tooling, English) | `a1ec67a78153` | validated on cluster | yes |
| `scripts/analysis/graph_structure.py` | renamed from scripts/analysis/93_graph_validation_run16E.py (English text; code equivalent) | `dce7548099f2` | check_code_unchanged | yes |
| `scripts/analysis/unrecovered_benchmark_edges.py` | renamed from scripts/analysis/94_failure_modes.py (English text; code equivalent) | `2b9b208523bb` | check_code_unchanged | yes |
| `scripts/analysis/descriptor_frequencies_literal.py` | renamed from scripts/analysis/96_descriptor_frequencies_run16E.py (English text; code equivalent) | `b8b5e193b5e3` | check_code_unchanged | yes |
| `scripts/analysis/descriptor_partition.py` | renamed from scripts/analysis/95d_fermer_partition_descripteurs.py (English text; code equivalent) | `30113e7b1741` | check_code_unchanged | yes |
| `scripts/analysis/surface_forms_and_entity_counts.py` | renamed from scripts/analysis/95g_derniers_controles.py (English text; code equivalent) | `6c87f767882e` | check_code_unchanged | yes |
| `scripts/analysis/reference_graph_inventory.py` | renamed from scripts/analysis/95h_mesurer_graphe_reference.py (English text; code equivalent) | `72966498ea50` | check_code_unchanged | yes |
| `scripts/analysis/reference_graph_and_benchmark_provenance.py` | renamed from scripts/analysis/95i_provenance_benchmark26.py (English text; code equivalent) | `e7eb78a6dacd` | check_code_unchanged | yes |
| `scripts/analysis/manifest_parameters_and_benchmark_targeting.py` | renamed from scripts/analysis/95m_argv_manifeste_et_ciblage.py (English text; code equivalent) | `cfcca9a902ca` | check_code_unchanged | yes |
| `scripts/analysis/model_comparison.py` | renamed from scripts/analysis/95p_comparaison_controlee_modeles.py (English text; code equivalent) | `69f9c8809a77` | check_code_unchanged | yes |
| `scripts/analysis/model_comparison_commands.sh` | renamed from scripts/analysis/95p_commands.sh (English header; commands identical) | `1ed35e3539d7` | diff | yes |
| `tools/check_code_unchanged.py` | new version (--allow-messages, --allow-renames) | `61918155a7c9` | equivalence of eb205c4 verified | yes |
| `scripts/analysis/family_corroboration.py` | renamed from scripts/analysis/100_family_corroboration.py (English text; code equivalent) | `80bedbf14f62` | check_code_unchanged --allow-messages | yes |
| `configs/entity_families.csv` | file:families_v2.csv | `3e6d1f192ace` | reproduces 26 families, 17/42, 135, tier means 9.4/10.1 (156) | yes |
| `scripts/evaluation/nli_verifier_comparison.py` | renamed from scripts/evaluation/95_nli_verifier_substitution.py (English; checked by 157) | `feae469d58c4` | 157 | yes |
| `scripts/evaluation/control_verdict_alignment.py` | renamed from scripts/evaluation/106c_join_verify.py (English; checked by 157) | `f7e093b11166` | 157 | yes |
| `scripts/evaluation/graph_with_located_provenance.py` | renamed from scripts/evaluation/145_graph_with_located_provenance.py (English; checked by 157) | `5ae843ca371c` | 157 | yes |
| `scripts/figures/fig_tradeoff.py` | renamed from scripts/figures/95y_fig_tradeoff.py (English; checked by 157) | `c56fc22b90cb` | 157 | yes |
| `scripts/tools/benchmark26_from_pipeline_constant.py` | renamed from scripts/tools/50_extract_lb26_reference.py (English; checked by 157) | `93750d8e4748` | 157 | yes |
| `scripts/tools/corpus_preflight.py` | renamed from scripts/tools/48e_preflight_41.py (English; checked by 157) | `b80fe21da5b3` | 157 | yes |
| `scripts/figures/fig_pipeline.py` | renamed from scripts/figures/95q_fig_pipeline.py (English; checked by 157) | `14d248ebe243` | 157 | yes |
| `scripts/evaluation/retrieval_replay_and_benchmark_targeting.py` | renamed from scripts/evaluation/143_benchmark_leakage_and_retrieval_recall.py (English docstring; checked by 158) | `60d15810652d` | 158 | yes |
| `scripts/evaluation/replay_fidelity_and_sentence_cooccurrence.py` | renamed from scripts/evaluation/144_replay_fidelity_and_sentence_cooccurrence.py (English docstring; checked by 158) | `9f61a0c4ff76` | 158 | yes |
| `scripts/evaluation/verifier_decision_rules.py` | renamed from scripts/evaluation/106d_metrics_stratified.py (English docstring; checked by 158) | `87f7462bae4f` | 158 | yes |
| `scripts/paper/classify_open_vocabulary.py` | renamed from scripts/paper/68_classify_vocabulary.py (English; reproduces the Section 3.6 counts, checked by 159) | `2a1f7ed601ad` | 159 | yes |
| `scripts/figures/fig_pipeline.py` | evaluation row subtitles shortened to one line (159) | `b60acc86a03b` | 159 | yes |
| `scripts/figures/fig_corroboration.py` | new producer of Figure 7 (checks published bins and tier statistics) | `1488b619ef05` | validated on cluster | yes |
| `scripts/figures/fig_sensitivity.py` | new producer of Figure 6 (checks published counts; labels say evidence sentence) | `ce0e83aaab31` | validated on cluster | yes |
| `scripts/figures/fig_vocabulary_gap.py` | new producer of Figure 5 (checks published counts; says extraction vocabulary) | `b7daac710149` | validated on cluster | yes |
| `scripts/figures/fig_descriptor_vocabulary.py` | new producer of Figure 4 (checks the 40 published counts) | `a8a5d93e9730` | validated on cluster | yes |
| `scripts/figures/fig_layers.py` | new producer of Figure 2 (provenance split; 12 directional relations not checked shown separately) | `0e3afe44912b` | validated on cluster | yes |
| `data/README.md` | written for the release | `ee88135c8007` | rewritten, replaces the copy from 9c03730 | yes |
