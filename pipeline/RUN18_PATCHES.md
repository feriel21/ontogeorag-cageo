# Scripts de production run18

Generes par `176_generer_scripts_run18.py`. Les originaux ne sont pas modifies.

| fichier | sha256 |
|---|---|
| pipeline/02_extract_triples.py | `f2555decd4de68a7cc8c5630c944990a256d8f14cf5ad4809ef0dd03dd7a5eb6` |
| pipeline/02_extract_triples_run18.py | `7fdc6cc1a44224a0a85451ce89e2d61bcd313b9e6e7361cec4665e0d27d419df` |
| pipeline/04_clean_validate.py | `21383881fdf0ce953ceb63cdc7e404f5a34c4ec5917e373b31d74ba20645b276` |
| pipeline/04_clean_validate_run18.py | `e5a8db171df78204a27325a6f8d185496a23e668c1ad2cb34e879091ec61748a` |
| configs/prompt_examples_run18.json | `f9210b2584c11c5b5ca3386483d363e2c3f208b9cc815d8e0be0849b0ea289ca` |

Source des exemples : Exemples de format tires de 11 articles hors du corpus evalue (index output/generalization/step1, candidats 177, graine 20261021). Chaque exemple est affirme par la phrase citee dans 'provenance'. Aucun exemple ne reprend un couple des relations de comparaison (verifie par 176 et 174).

Changements : 02 E1 exemples lus dans le fichier ci-dessus ; 02 E2 contre-exemples generiques ; 04 N1 protection = KNOWN_DESCRIPTORS ; 04 N2 cadres = KNOWN_SETTINGS ; 04 N3 aucune arete d'evaluation.

```diff
--- pipeline/02_extract_triples.py
+++ pipeline/02_extract_triples_run18.py
@@ -1,4 +1,6 @@
 #!/usr/bin/env python3
+# GENERE par 176_generer_scripts_run18.py a partir de pipeline/02_extract_triples.py (sha f2555decd4de).
+# Voir pipeline/RUN18_PATCHES.md. Ne pas editer a la main.
 """
 pipeline/02_extract_triples.py — RAG Triple Extraction (Stage 2)
 ==================================================================
@@ -91,16 +93,15 @@
   "source", "source_type", "relation", "target", "target_type"
   "salience": "typical" | "common" | "occasional"  (REQUIRED for hasDescriptor)
 
-GOLD EXAMPLES (from peer-reviewed MTD literature):
+EXAMPLES (format only):
 [
-  {{"source": "mass transport deposit", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "chaotic", "target_type": "Descriptor", "salience": "typical"}},
-  {{"source": "mass transport deposit", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "transparent", "target_type": "Descriptor", "salience": "typical"}},
-  {{"source": "mass transport deposit", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "hummocky", "target_type": "Descriptor", "salience": "common"}},
-  {{"source": "turbidite", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "parallel", "target_type": "Descriptor", "salience": "typical"}},
-  {{"source": "turbidite", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "high-amplitude", "target_type": "Descriptor", "salience": "common"}},
-  {{"source": "debris flow", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "chaotic", "target_type": "Descriptor", "salience": "typical"}},
-  {{"source": "slide", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "blocky", "target_type": "Descriptor", "salience": "typical"}},
-  {{"source": "hemipelagite", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "continuous", "target_type": "Descriptor", "salience": "typical"}}
+  {{"source": "slump", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "folded", "target_type": "Descriptor", "salience": "typical"}},
+  {{"source": "mass transport deposit", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "deformed", "target_type": "Descriptor", "salience": "typical"}},
+  {{"source": "mass transport deposit", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "irregular", "target_type": "Descriptor", "salience": "common"}},
+  {{"source": "mass transport deposit", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "smooth", "target_type": "Descriptor", "salience": "occasional"}},
+  {{"source": "basal shear surface", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "parallel", "target_type": "Descriptor", "salience": "common"}},
+  {{"source": "block", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "undeformed", "target_type": "Descriptor", "salience": "occasional"}},
+  {{"source": "landslide", "source_type": "SeismicObject", "relation": "hasDescriptor", "target": "faulted", "target_type": "Descriptor", "salience": "occasional"}}
 ]
 
 IMPORTANT — DO NOT IGNORE NON-MTD SUBJECTS:
@@ -111,8 +112,7 @@
   {{"target": "chaotic seismic facies"}}  <- use "chaotic" not verbose form
   {{"source": "MTC_2"}}                   <- use class "mass transport deposit" not instance
   {{"target": "hummocky and irregular"}}  <- use "hummocky" not compound form
-  {{"source": "mass transport deposit", "target": "chaotic"}}  <- when text says debris flow is chaotic, use "debris flow" not MTD
-  {{"source": "mass transport deposit", "target": "blocky"}}   <- when text says slide is blocky, use "slide" not MTD
+  {{"source": "<parent class>", "target": "<descriptor>"}}  <- when the text attributes the descriptor to a sub-type (e.g. a debris flow), use that sub-type as source
 
 
 If no valid triples can be extracted, respond with: []
@@ -138,14 +138,15 @@
 4. Use ONLY these relations: {relations}
 5. Use ONLY these entity types: {types}
 
-GOLD EXAMPLES (from peer-reviewed MTD literature):
+EXAMPLES (format only):
 [
-  {{"source": "earthquake", "source_type": "Process", "relation": "triggers", "target": "slope failure", "target_type": "Process"}},
-  {{"source": "pore pressure", "source_type": "Process", "relation": "controls", "target": "slope failure", "target_type": "Process"}},
-  {{"source": "methane hydrate dissociation", "source_type": "Process", "relation": "causes", "target": "increase of volume and pressure buildup", "target_type": "Process"}},
-  {{"source": "large storm waves", "source_type": "Process", "relation": "triggers", "target": "shear stresses on the seafloor", "target_type": "Process"}},
-  {{"source": "gas hydrate dissociation", "source_type": "Process", "relation": "affects", "target": "retrogressive slide development", "target_type": "Process"}},
-  {{"source": "slope failure", "source_type": "Process", "relation": "causes", "target": "mass transport deposit", "target_type": "GeologicalObject"}}
+  {{"source": "halokinesis", "source_type": "Process", "relation": "triggers", "target": "mass transport deposit", "target_type": "GeologicalObject"}},
+  {{"source": "storm", "source_type": "Process", "relation": "triggers", "target": "submarine landslide", "target_type": "Process"}},
+  {{"source": "episodic fluid flow", "source_type": "Process", "relation": "triggers", "target": "slope instability", "target_type": "Process"}},
+  {{"source": "fault", "source_type": "GeologicalObject", "relation": "causes", "target": "chaotic rearrangement of sediments", "target_type": "Process"}},
+  {{"source": "sea level fluctuation", "source_type": "Process", "relation": "causes", "target": "mass transport deposit", "target_type": "GeologicalObject"}},
+  {{"source": "topography", "source_type": "Process", "relation": "controls", "target": "mass transport deposit distribution", "target_type": "Process"}},
+  {{"source": "landslide volume", "source_type": "Process", "relation": "controls", "target": "tsunami wave magnitude", "target_type": "Process"}}
 ]
 
 If no valid triples can be extracted, respond with: []
@@ -170,14 +171,15 @@
 3. Use ONLY these relations: {relations}
 4. Use ONLY these entity types: {types}
 
-GOLD EXAMPLES (from peer-reviewed MTD literature):
+EXAMPLES (format only):
 [
-  {{"source": "mass transport deposit", "source_type": "GeologicalObject", "relation": "occursIn", "target": "continental slope", "target_type": "GeologicalSetting"}},
-  {{"source": "mass transport deposit", "source_type": "GeologicalObject", "relation": "occursIn", "target": "abyssal plain", "target_type": "GeologicalSetting"}},
-  {{"source": "anahuac shale", "source_type": "GeologicalObject", "relation": "underlies", "target": "mass transport deposit", "target_type": "GeologicalObject"}},
-  {{"source": "slide", "source_type": "GeologicalObject", "relation": "overlies", "target": "hemipelagite", "target_type": "GeologicalObject"}},
-  {{"source": "debris flow", "source_type": "GeologicalObject", "relation": "occursIn", "target": "continental slope", "target_type": "GeologicalSetting"}},
-  {{"source": "turbidite", "source_type": "GeologicalObject", "relation": "occursIn", "target": "basin floor", "target_type": "GeologicalSetting"}}
+  {{"source": "submarine landslide", "source_type": "GeologicalObject", "relation": "occursIn", "target": "active margin", "target_type": "GeologicalSetting"}},
+  {{"source": "slump complex", "source_type": "GeologicalObject", "relation": "occursIn", "target": "continental margin", "target_type": "GeologicalSetting"}},
+  {{"source": "basal shear zone", "source_type": "GeologicalObject", "relation": "overlies", "target": "undeformed substrate", "target_type": "GeologicalObject"}},
+  {{"source": "post-MTD sediment", "source_type": "GeologicalObject", "relation": "overlies", "target": "mass transport deposit", "target_type": "GeologicalObject"}},
+  {{"source": "Cenozoic sedimentary unit", "source_type": "GeologicalObject", "relation": "overlies", "target": "Mesozoic igneous basement", "target_type": "GeologicalObject"}},
+  {{"source": "basal shear zone", "source_type": "GeologicalObject", "relation": "partOf", "target": "mass transport deposit", "target_type": "GeologicalObject"}},
+  {{"source": "translational domain", "source_type": "GeologicalObject", "relation": "partOf", "target": "mass transport deposit", "target_type": "GeologicalObject"}}
 ]
 
 If no valid triples can be extracted, respond with: []
--- pipeline/04_clean_validate.py
+++ pipeline/04_clean_validate_run18.py
@@ -1,4 +1,6 @@
 #!/usr/bin/env python3
+# GENERE par 176_generer_scripts_run18.py a partir de pipeline/04_clean_validate.py (sha 21383881fdf0).
+# Voir pipeline/RUN18_PATCHES.md. Ne pas editer a la main.
 """
 pipeline/04_clean_validate.py — Validation, Cleaning & Canonicalization (Stage 4, v5)
 ========================================================================================
@@ -42,41 +44,14 @@
     KNOWN_DESCRIPTORS,
 )
 from pipeline.rag.constants import (
-    LB2019_BENCHMARK_DESCRIPTORS as LB_DESCRIPTORS,
+    KNOWN_DESCRIPTORS as LB_DESCRIPTORS,  # run18 N1 : vocabulaire complet de l'ontologie
 )
 
 # ═══════════════════════════════════════════════════════════════════════
 # CONFIGURATION
 # ═══════════════════════════════════════════════════════════════════════
 
-LB_REFERENCE_EDGES = [
-    ("mass transport deposit", "hasDescriptor", "chaotic"),
-    ("mass transport deposit", "hasDescriptor", "transparent"),
-    ("mass transport deposit", "hasDescriptor", "hummocky"),
-    ("mass transport deposit", "hasDescriptor", "blocky"),
-    ("mass transport deposit", "hasDescriptor", "discontinuous"),
-    ("mass transport deposit", "hasDescriptor", "massive"),
-    ("turbidite", "hasDescriptor", "parallel"),
-    ("turbidite", "hasDescriptor", "continuous"),
-    ("turbidite", "hasDescriptor", "layered"),
-    ("turbidite", "hasDescriptor", "high-amplitude"),
-    ("debris flow", "hasDescriptor", "chaotic"),
-    ("debris flow", "hasDescriptor", "hummocky"),
-    ("slide", "hasDescriptor", "blocky"),
-    ("slide", "hasDescriptor", "undeformed"),
-    ("hemipelagite", "hasDescriptor", "parallel"),
-    ("hemipelagite", "hasDescriptor", "continuous"),
-    ("hemipelagite", "hasDescriptor", "low-amplitude"),
-    ("slope failure", "causes", "mass transport deposit"),
-    ("earthquake", "triggers", "slope failure"),
-    ("pore pressure", "controls", "slope failure"),
-    ("turbidity current", "formedBy", "debris flow"),
-    ("mass transport deposit", "occursIn", "continental slope"),
-    ("mass transport deposit", "occursIn", "abyssal plain"),
-    ("debris flow", "occursIn", "continental slope"),
-    ("turbidite", "occursIn", "basin floor"),
-    ("slide", "overlies", "hemipelagite"),
-]
+LB_REFERENCE_EDGES = []  # run18 N3 : aucune arete d'evaluation dans le pipeline
 
 VALID_RELATIONS = {
     "hasDescriptor",
@@ -485,14 +460,7 @@
         # Same protection for LB2019 setting terms — a setting merged into
         # a co-clustered non-setting entity would silently break occursIn
         # recall against the benchmark edges. (P9 = project ticket ref.)
-        lb_settings = {
-            "continental slope",
-            "abyssal plain",
-            "basin floor",
-            "hemipelagite",
-            "passive margins",
-            "continental shelf",
-        }
+        lb_settings = set(KNOWN_SETTINGS)  # run18 N2 : cadres de l'ontologie
         lb_setting_members = [m for m in members if m in lb_settings]
         if lb_setting_members:
             for m in members:

```
