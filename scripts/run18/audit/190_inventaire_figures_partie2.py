#!/usr/bin/env python3
"""
190_inventaire_figures_partie2.py  (lecture seule, CPU, quelques secondes)

Cherche, dans ~/sfm-repro, ~/cae-pythae et ~/cae-pythae-run, les figures et
les fichiers de résultats de la partie II (autoencodeur convolutif) :
  * reconstruction (patch d'entrée et patch reconstruit)
  * espace latent (UMAP, t-SNE, ACP, groupes)
  * contribution des dimensions (Chaos, Transparency)
  * sections et masques
  * métriques (silhouette, perte, erreur de reconstruction)

Produit, dans ~/ontogeorag/output/audit/190_inventaire_partie2/ :
  * inventaire.csv            : tous les fichiers trouvés, catégorie, taille, date
  * planche_<categorie>.png   : miniatures numérotées des images de chaque catégorie
  * resume.txt                : le résumé imprimé à l'écran
  * scripts_figures.csv       : les scripts .py qui produisent des figures

N'écrit rien ailleurs, ne modifie aucun fichier, refuse d'écraser.

Lancement :
  source ~/kg_test/venv/bin/activate
  python -u 190_inventaire_figures_partie2.py
Puis télécharger les planche_*.png et resume.txt, et les envoyer.
"""
import csv
import datetime
import os
import re
import sys
from pathlib import Path

ROOTS = ["~/sfm-repro", "~/cae-pythae", "~/cae-pythae-run"]
OUTDIR = "~/ontogeorag/output/audit/190_inventaire_partie2"
PRUNE = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints", "venv", ".venv",
         "site-packages", "wandb", ".cache"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg"}
DOC_EXT = {".pdf"}
DATA_EXT = {".json", ".csv", ".txt", ".log", ".npz", ".npy", ".pkl", ".yaml", ".yml"}
MAX_THUMBS = 40

CATEGORIES = [
    ("reconstruction", r"recon|reconstruct|decod|input.?vs|in_out|autoenc.*(out|pred)|original.?vs"),
    ("espace_latent", r"latent|umap|tsne|t-sne|pca|embedd|cluster|kmeans|k_means|hdbscan|z_space"),
    ("contribution", r"chaos|transpar|contrib|attribut|importance|saliency|ablation|channel|feature"),
    ("section_masque", r"mask|overlay|section|inline|xline|crossline|il\d{3,4}|xl\d{3,4}|geobod|facies"),
    ("metriques", r"silhouette|davies|calinski|loss|mse|metric|score|eval"),
]


def stop(msg):
    print("STOP: " + msg, file=sys.stderr)
    sys.exit(1)


def category(path_str):
    low = path_str.lower()
    hits = [name for name, rx in CATEGORIES if re.search(rx, low)]
    return hits[0] if hits else "autre", hits


def main():
    out = Path(OUTDIR).expanduser()
    if out.exists() and any(out.iterdir()):
        stop("dossier de sortie non vide, rien n'est écrasé : " + str(out))

    roots = [Path(r).expanduser() for r in ROOTS]
    present = [r for r in roots if r.is_dir()]
    if not present:
        stop("aucun des trois dépôts n'est trouvé : " + ", ".join(ROOTS))

    rows, scripts = [], []
    for root in present:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in PRUNE]
            for fn in filenames:
                p = Path(dirpath) / fn
                ext = p.suffix.lower()
                rel = str(p.relative_to(root.parent))
                if ext == ".py":
                    try:
                        src = p.read_text(errors="ignore")
                    except OSError:
                        continue
                    if "savefig" in src or "imsave" in src:
                        outs = sorted(set(re.findall(r"""["']([^"']+\.(?:png|pdf|jpg|svg))["']""", src)))
                        scripts.append([rel, len(outs), " | ".join(outs[:8])])
                    continue
                if ext not in IMG_EXT | DOC_EXT | DATA_EXT:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                cat, hits = category(rel)
                kind = "image" if ext in IMG_EXT else ("pdf" if ext in DOC_EXT else "donnees")
                rows.append({
                    "depot": root.name, "chemin": str(p), "relatif": rel, "type": kind,
                    "categorie": cat, "autres_categories": ",".join(hits[1:]),
                    "taille_ko": round(st.st_size / 1024, 1),
                    "modifie": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "mtime": st.st_mtime,
                })

    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["categorie"], -r["mtime"]))
    with open(out / "inventaire.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "depot", "type", "categorie", "autres_categories", "taille_ko", "modifie", "chemin"])
        for i, r in enumerate(rows, 1):
            r["id"] = i
            w.writerow([i, r["depot"], r["type"], r["categorie"], r["autres_categories"],
                        r["taille_ko"], r["modifie"], r["chemin"]])
    with open(out / "scripts_figures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["script", "n_figures_nommees", "figures_nommees"])
        w.writerows(sorted(scripts))

    lines = ["Inventaire partie II, " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
             "dépôts lus : " + ", ".join(str(r) for r in present),
             "dépôts absents : " + (", ".join(str(r) for r in roots if r not in present) or "aucun"), ""]
    for cat in [c for c, _ in CATEGORIES] + ["autre"]:
        sel = [r for r in rows if r["categorie"] == cat]
        imgs = [r for r in sel if r["type"] == "image"]
        pdfs = [r for r in sel if r["type"] == "pdf"]
        data = [r for r in sel if r["type"] == "donnees"]
        lines.append("== " + cat + " : " + str(len(imgs)) + " images, " + str(len(pdfs)) + " pdf, "
                     + str(len(data)) + " fichiers de données")
        if cat == "autre":
            lines.append("   (non détaillé : voir inventaire.csv)")
            continue
        for r in (imgs + pdfs)[:15]:
            lines.append("   [" + str(r["id"]) + "] " + r["modifie"] + "  " + r["relatif"])
        for r in data[:10]:
            lines.append("   [" + str(r["id"]) + "] données  " + r["modifie"] + "  " + r["relatif"])
        lines.append("")
    lines.append("scripts produisant des figures : " + str(len(scripts)) + " (voir scripts_figures.csv)")

    # planches de miniatures
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        lines.append("PIL absent : pas de planches, utiliser inventaire.csv")
        Image = None
    if Image is not None:
        for cat in [c for c, _ in CATEGORIES]:
            imgs = [r for r in rows if r["categorie"] == cat and r["type"] == "image"
                    and Path(r["chemin"]).suffix.lower() in {".png", ".jpg", ".jpeg"}][:MAX_THUMBS]
            if not imgs:
                continue
            tw, th, cols = 320, 240, 5
            nrow = (len(imgs) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * tw, nrow * (th + 40)), "white")
            draw = ImageDraw.Draw(sheet)
            for k, r in enumerate(imgs):
                try:
                    im = Image.open(r["chemin"]).convert("RGB")
                    im.thumbnail((tw - 10, th - 10))
                except Exception:
                    continue
                x, y = (k % cols) * tw, (k // cols) * (th + 40)
                sheet.paste(im, (x + 5, y + 5))
                label = "[" + str(r["id"]) + "] " + Path(r["chemin"]).name
                draw.text((x + 5, y + th + 2), label[:48], fill=(0, 0, 0))
                draw.text((x + 5, y + th + 18), r["depot"] + "  " + r["modifie"], fill=(90, 100, 114))
            sheet.save(out / ("planche_" + cat + ".png"))
            lines.append("planche écrite : planche_" + cat + ".png (" + str(len(imgs)) + " miniatures)")

    text = "\n".join(lines)
    (out / "resume.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print("\nécrit dans " + str(out))
    print("OK")


if __name__ == "__main__":
    main()