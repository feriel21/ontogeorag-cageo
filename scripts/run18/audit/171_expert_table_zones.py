import csv, re, textwrap
from collections import Counter
from pathlib import Path
F = Path.home()/"ontogeorag/output/run16E/deliverable/ontogeorag_table_EXPERT.csv"
rows = list(csv.DictReader(open(F, encoding="utf-8", newline="")))
REF = [re.compile(r"^[a-z][\w\-'\u2019 ]{1,40},\s+(?:[a-z]\.\s*-?){1,4}", re.I),
       re.compile(r"\b(?:19|20)\d{2}[a-z]?\)?[\.,:]\s.{5,250}\b(?:v\.|vol\.|p\.|pp\.|bull\.|j\.|geol\.|res\.|sci\.)", re.I),
       re.compile(r"\b\d{1,4}\s*,\s*\d{1,5}\s*[\u2013\-]\s*\d{1,5}\.?\s*$")]
HEADER = [re.compile(r"^\d{1,4}\s+[a-z]", re.I),
          re.compile(r"copyright|special publication no|downloaded from|all rights reserved|\bissn\b", re.I),
          re.compile(r"\b[a-z][a-z ]{10,80}\s\d{1,4}\s*$", re.I)]
CAPTION = [re.compile(r"^\(?(?:fig\.?|figs\.?|figure|table)\s*\d", re.I), re.compile(r"^\(?[a-h]\)\s", re.I),
           re.compile(r"\(\s*[a-h]\s*\)\s*(?:poststack|seismic|profile|map|section|time|rms|amplitude)", re.I),
           re.compile(r"^note (?:the|that)\b|^the profile\b|\bfig\.\s*\d+\s+seismic", re.I)]
def ttype(p):
    s = " ".join(str(p or "").split()).strip(" \"'")
    for n, pats in (("reference", REF), ("header", HEADER), ("caption", CAPTION)):
        if any(x.search(s) for x in pats): return n
    return "body"
print(len(rows), "relations |", F.name)
print("text_zone :", dict(Counter(r["text_zone"] for r in rows)))
print("provenance:", dict(Counter(r["provenance"] for r in rows)))
c = Counter((r["text_zone"], ttype(r["source_passage"])) for r in rows)
T = ["body", "reference", "header", "caption"]
print(f"\n{'text_zone':22s}" + "".join(f"{t:>11s}" for t in T))
for z in sorted({r["text_zone"] for r in rows}):
    print(f"{z:22s}" + "".join(f"{c[(z, t)]:>11d}" for t in T))
out = Path.home()/"ontogeorag/output/audit/171_expert_table_zones.txt"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for r in sorted(rows, key=lambda r: (r["text_zone"], ttype(r["source_passage"]))):
        f.write(f"\n[{r['relation_id']}] zone={r['text_zone']} | type={ttype(r['source_passage'])} | section={r['source_section']} | tier={r['confidence_tier']}\n")
        f.write(f"    {r['subject']} --{r['relation_type']}--> {r['object']}\n")
        for line in textwrap.wrap(" ".join(str(r['source_passage'] or '').split()), 100):
            f.write("      " + line + "\n")
print("\ndetail:", out)
