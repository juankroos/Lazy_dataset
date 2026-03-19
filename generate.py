"""
generate.py — Génère la population initiale de structures candidates

Prend une description libre du problème, extrait les 5 dimensions,
et produit 8-12 schémas JSON diversifiés prêts à être évalués.

Usage :
    python generate.py "description du problème"
    python generate.py "description" --output schemas/
    python generate.py "description" --model-desc "CNN, 224x224" --verbose
    python generate.py "description" --n 10 --output schemas/
"""

import json, argparse, re, sys
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE DU BESOIN — 5 DIMENSIONS
# ══════════════════════════════════════════════════════════════════════════════

class ProblemProfile:
    """
    Profil extrait de la description libre du problème.
    Différent de ModelProfile (evaluate.py) qui décrit l'architecture du modèle.
    Celui-ci décrit le PROBLÈME : quelles données, quelle tâche, quels objectifs.
    """

    def __init__(self):
        # Dimension 1 — tâche
        self.task            = "classification"   # classification · régression · génération · détection
        self.n_classes       = 4

        # Dimension 2 — modalités d'entrée détectées
        self.has_image       = False
        self.has_signal      = False              # capteur numérique, température, etc.
        self.has_text        = False
        self.has_audio       = False
        self.has_tabular     = False              # features structurées (âge, prix, etc.)
        self.image_size      = (32, 32)
        self.signal_size     = 10                 # taille du vecteur signal
        self.n_channels      = 1

        # Dimension 3 — sortie attendue
        self.output_type     = "label"            # label · score · text · vector · multi
        self.needs_confidence= False
        self.needs_explain   = False

        # Dimension 4 — contraintes
        self.embedded        = False
        self.latency_ms      = None
        self.has_gpu         = False
        self.dataset_size    = "medium"           # small (<1k) · medium · large (>100k)

        # Dimension 5 — priorité
        self.priority        = "performance"      # performance · simplicite · explicabilite · vitesse

        # hypothèses faites (pour le rapport)
        self.hypotheses      = []

    @classmethod
    def from_description(cls, desc: str) -> "ProblemProfile":
        p   = cls()
        txt = desc.lower()

        # ── Dimension 1 : tâche ──
        if any(k in txt for k in ("régression", "regression", "prédire une valeur",
                                   "valeur continue", "estimer", "quantifier")):
            p.task = "regression"
        elif any(k in txt for k in ("générer", "générat", "créer du texte", "summarize",
                                     "résumer", "traduire", "translation")):
            p.task = "generation"
        elif any(k in txt for k in ("détecter", "localiser", "bounding box",
                                     "segmenter", "segmentation")):
            p.task = "detection"
        else:
            p.task = "classification"

        m = re.search(r"(\d+)\s*(?:class|catégorie|état|label|classe)", txt)
        if m: p.n_classes = int(m.group(1))
        else:
            p.hypotheses.append("n_classes supposé = 4 (non spécifié)")

        # ── Dimension 2 : modalités ──
        if any(k in txt for k in ("image", "photo", "pixel", "caméra", "camera",
                                   "frame", "visuel", "visual", "thermique", "thermal")):
            p.has_image = True
            m = re.search(r"(\d{2,4})\s*[x×]\s*(\d{2,4})", txt)
            if m:
                p.image_size = (int(m.group(1)), int(m.group(2)))
            else:
                p.hypotheses.append("taille image supposée 32x32 (non spécifiée)")
            if any(k in txt for k in ("rgb", "couleur", "color", "3 canal")):
                p.n_channels = 3
            else:
                p.hypotheses.append("image supposée grayscale (non spécifié)")

        if any(k in txt for k in ("température", "temperature", "capteur", "sensor",
                                   "signal", "mesure", "relevé", "série temporelle",
                                   "time series", "vibration", "pression", "courant",
                                   "accéléromètre", "gyroscope", "météo", "weather")):
            p.has_signal = True
            m = re.search(r"(\d+)\s*(?:point|mesure|valeur|sample|capteur)", txt)
            if m:
                p.signal_size = int(m.group(1))
            else:
                p.hypotheses.append("taille vecteur signal supposée 10 (non spécifiée)")

        if any(k in txt for k in ("texte", "text", "phrase", "document", "tweet",
                                   "avis", "commentaire", "nlp", "langue")):
            p.has_text = True

        if any(k in txt for k in ("audio", "son", "parole", "speech", "microphone",
                                   "spectrogramme", "waveform")):
            p.has_audio = True

        if any(k in txt for k in ("tableau", "table", "colonne", "feature",
                                   "âge", "prix", "taille", "poids", "csv")):
            p.has_tabular = True

        # si aucune modalité détectée → supposer signal
        if not any([p.has_image, p.has_signal, p.has_text, p.has_audio, p.has_tabular]):
            p.has_signal = True
            p.hypotheses.append("modalité supposée : signal numérique (non spécifiée)")

        # ── Dimension 3 : sortie ──
        if any(k in txt for k in ("confiance", "confidence", "probabilité", "probability",
                                   "score de certitude")):
            p.needs_confidence = True
        if any(k in txt for k in ("expliquer", "explicable", "interprétable", "pourquoi",
                                   "comprendre", "raison", "feature importance")):
            p.needs_explain = True
            p.output_type   = "multi"
        if any(k in txt for k in ("plusieurs sorties", "multi-output", "multi-label")):
            p.output_type = "multi"

        # ── Dimension 4 : contraintes ──
        if any(k in txt for k in ("raspberry", "embarqué", "embedded", "edge",
                                   "mobile", "microcontroller", "iot", "nano")):
            p.embedded = True
        if any(k in txt for k in ("gpu", "cuda", "serveur", "cloud", "cluster")):
            p.has_gpu = True
        m = re.search(r"(\d+)\s*ms", txt)
        if m: p.latency_ms = int(m.group(1))

        if any(k in txt for k in ("peu de données", "petit dataset", "< 1000",
                                   "100 exemples", "quelques centaines")):
            p.dataset_size = "small"
        elif any(k in txt for k in ("grand dataset", "millions", "> 100k", "beaucoup de données")):
            p.dataset_size = "large"

        # ── Dimension 5 : priorité ──
        if any(k in txt for k in ("expliquer", "comprendre", "interpréter",
                                   "non technique", "équipe métier")):
            p.priority = "explicabilite"
        elif any(k in txt for k in ("rapide", "temps réel", "latence", "embarqué",
                                     "léger", "lightweight")):
            p.priority = "vitesse"
        elif any(k in txt for k in ("simple", "maintenir", "déployer facilement",
                                     "production", "robuste")):
            p.priority = "simplicite"
        else:
            p.priority = "performance"

        return p

    def summary(self) -> str:
        modalities = []
        if self.has_image:   modalities.append(f"image {self.image_size[0]}x{self.image_size[1]}")
        if self.has_signal:  modalities.append(f"signal(dim={self.signal_size})")
        if self.has_text:    modalities.append("texte")
        if self.has_audio:   modalities.append("audio")
        if self.has_tabular: modalities.append("tabulaire")

        lines = [
            f"  Tâche       : {self.task} ({self.n_classes} classes)",
            f"  Modalités   : {', '.join(modalities) or 'non détectées'}",
            f"  Sortie      : {self.output_type}"
                + (" + confiance" if self.needs_confidence else "")
                + (" + explication" if self.needs_explain else ""),
            f"  Contraintes : {'embarqué ' if self.embedded else ''}"
                + (f"latence<{self.latency_ms}ms " if self.latency_ms else "")
                + (f"{'GPU' if self.has_gpu else 'no-GPU'}"),
            f"  Priorité    : {self.priority}",
        ]
        if self.hypotheses:
            lines.append(f"  Hypothèses  :")
            for h in self.hypotheses:
                lines.append(f"    - {h}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# GÉNÉRATEUR DE STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

def make_output_field(prob: ProblemProfile) -> list:
    """Génère le(s) champ(s) de sortie selon le profil."""
    fields = []

    if prob.task == "regression":
        fields.append({"name":"output","type":"scalar","role":"output","derivation":"raw"})
    elif prob.task == "generation":
        fields.append({"name":"output","type":"text_label","role":"output","derivation":"raw"})
    else:
        ftype = f"int_{prob.n_classes}classes"
        fields.append({"name":"label","type":ftype,"role":"output","derivation":"raw"})

    if prob.needs_confidence:
        fields.append({"name":"confidence","type":"scalar","role":"output","derivation":"raw"})
    if prob.needs_explain:
        fields.append({"name":"explanation","type":"vector_8","role":"output","derivation":"raw"})

    return fields


def image_field(prob: ProblemProfile, derivation="raw") -> dict:
    h, w = prob.image_size
    return {"name":"image","type":f"matrix_{h}x{w}","role":"input","derivation":derivation}

def signal_field(prob: ProblemProfile, derivation="raw", name="signal") -> dict:
    return {"name":name,"type":f"vector_{prob.signal_size}","role":"input","derivation":derivation}


def generate_candidates(prob: ProblemProfile, model_desc: str = None) -> list:
    """
    Génère 8–12 structures candidates adaptées au profil du problème.
    Chaque candidat = dict {schema_id, description, rationale, fields, fusion_strategy, temporal_window}
    """
    out    = make_output_field(prob)
    schemas = []

    # ── S01 : Basique (toujours présent — baseline) ──
    fields = []
    if prob.has_image:  fields.append(image_field(prob))
    if prob.has_signal: fields.append(signal_field(prob))
    if prob.has_text:   fields.append({"name":"text","type":"vector_128","role":"input","derivation":"raw"})
    if prob.has_audio:  fields.append({"name":"audio","type":"vector_64","role":"input","derivation":"raw"})
    if prob.has_tabular:fields.append({"name":"features","type":"vector_16","role":"input","derivation":"raw"})
    schemas.append({
        "schema_id": "s01_basic",
        "description": "Données brutes — baseline",
        "rationale": "Point de référence minimal. Sert à mesurer le gain des autres structures.",
        "fields": fields + out,
        "fusion_strategy": "none",
        "temporal_window": None,
    })

    # ── S02 : Statistiques dérivées ──
    fields = []
    if prob.has_image:
        fields.append(image_field(prob, derivation="mean"))
    if prob.has_signal:
        fields.append({"name":"signal_stats","type":f"vector_{prob.signal_size}",
                       "role":"input","derivation":"stats"})
    if prob.has_tabular:
        fields.append({"name":"features","type":"vector_16","role":"input","derivation":"normalized"})
    if prob.has_text and not fields:
        fields.append({"name":"text_embed","type":"vector_128","role":"input","derivation":"normalized"})
    if not fields:
        fields = [signal_field(prob, "stats")]
    schemas.append({
        "schema_id": "s02_derived_stats",
        "description": "Statistiques dérivées (mean, std, min, max)",
        "rationale": "Réduit la dimensionnalité. Souvent meilleur qu'un vecteur brut pour les signaux.",
        "fields": fields + out,
        "fusion_strategy": "none",
        "temporal_window": None,
    })

    # ── S03 : Fusion précoce (si multimodal) ──
    if sum([prob.has_image, prob.has_signal, prob.has_text, prob.has_audio]) >= 2:
        fields = []
        if prob.has_image:  fields.append(image_field(prob, "concat"))
        if prob.has_signal: fields.append(signal_field(prob, "raw"))
        if prob.has_text:   fields.append({"name":"text","type":"vector_128","role":"input","derivation":"raw"})
        schemas.append({
            "schema_id": "s03_early_fusion",
            "description": "Fusion précoce — concat toutes modalités",
            "rationale": "Exploite les corrélations inter-modalités dès l'entrée du modèle.",
            "fields": fields + out,
            "fusion_strategy": "early_concat",
            "temporal_window": None,
        })

    # ── S04 : Historique temporel (si signal présent) ──
    if prob.has_signal:
        win = 5
        if prob.latency_ms and prob.latency_ms < 100:
            win = 3   # fenêtre courte si contrainte latence
        fields = []
        if prob.has_image: fields.append(image_field(prob))
        fields.append({"name":"signal_history","type":f"vector_{prob.signal_size}",
                       "role":"input","derivation":f"window(signal,{win})"})
        schemas.append({
            "schema_id": f"s04_temporal_w{win}",
            "description": f"Historique temporel (fenêtre={win})",
            "rationale": f"Capture les tendances temporelles sur {win} pas de temps.",
            "fields": fields + out,
            "fusion_strategy": "none",
            "temporal_window": win,
        })

        # fenêtre longue si pas de contrainte latence
        if not prob.embedded and not (prob.latency_ms and prob.latency_ms < 200):
            fields2 = []
            if prob.has_image: fields2.append(image_field(prob))
            fields2.append({"name":"signal_history_long","type":f"vector_{prob.signal_size}",
                            "role":"input","derivation":"window(signal,10)"})
            schemas.append({
                "schema_id": "s05_temporal_w10",
                "description": "Historique temporel long (fenêtre=10)",
                "rationale": "Contexte plus long — utile si les patterns ont une longue dépendance.",
                "fields": fields2 + out,
                "fusion_strategy": "none",
                "temporal_window": 10,
            })

    # ── S05/S06 : Corrélation croisée (si image + signal) ──
    if prob.has_image and prob.has_signal:
        fields = [
            image_field(prob),
            signal_field(prob),
            {"name":"cross_corr","type":"scalar","role":"input",
             "derivation":"corr(image_mean,signal_mean)"},
        ]
        schemas.append({
            "schema_id": "s06_cross_corr",
            "description": "Corrélation croisée image × signal",
            "rationale": "Capture la relation entre les deux modalités. Utile si elles sont physiquement liées.",
            "fields": fields + out,
            "fusion_strategy": "none",
            "temporal_window": None,
        })

    # ── S07 : Dérivées riches (fréquentielles si signal) ──
    if prob.has_signal and not prob.embedded:
        fields = []
        if prob.has_image: fields.append(image_field(prob, "mean"))
        fields.append({"name":"signal_fft","type":f"vector_{prob.signal_size}",
                       "role":"input","derivation":"fft"})
        fields.append({"name":"signal_stats","type":f"vector_{prob.signal_size}",
                       "role":"input","derivation":"stats"})
        schemas.append({
            "schema_id": "s07_freq_features",
            "description": "Features fréquentielles (FFT) + statistiques",
            "rationale": "Utile pour les signaux avec des patterns périodiques (vibrations, audio, météo).",
            "fields": fields + out,
            "fusion_strategy": "none",
            "temporal_window": None,
        })

    # ── S08 : Minimal explicable (priorité explicabilité) ──
    if prob.needs_explain or prob.priority == "explicabilite":
        fields = []
        if prob.has_signal:
            fields.append({"name":"signal_mean","type":f"vector_{prob.signal_size}",
                           "role":"input","derivation":"mean"})
            fields.append({"name":"signal_std","type":f"vector_{prob.signal_size}",
                           "role":"input","derivation":"stats"})
        if prob.has_image:
            fields.append(image_field(prob, "mean"))
        if prob.has_text and not fields:
            fields.append({"name":"text_features","type":"vector_16","role":"input","derivation":"normalized"})
        if not fields:
            fields = [signal_field(prob, "stats")]
        schemas.append({
            "schema_id": "s08_explainable",
            "description": "Structure minimale explicable",
            "rationale": "Peu de features interprétables. Idéal pour expliquer les prédictions à des non-techniciens.",
            "fields": fields + out,
            "fusion_strategy": "none",
            "temporal_window": None,
        })

    # ── S09 : Optimisé embarqué (si contrainte hardware) ──
    if prob.embedded or (prob.latency_ms and prob.latency_ms < 100):
        fields = []
        if prob.has_signal:
            fields.append({"name":"signal_stats","type":f"vector_{prob.signal_size}",
                           "role":"input","derivation":"stats"})
        elif prob.has_image:
            fields.append(image_field(prob, "mean"))
        if not fields:
            fields = [signal_field(prob, "stats")]
        schemas.append({
            "schema_id": "s09_embedded",
            "description": "Structure ultra-légère pour déploiement embarqué",
            "rationale": "Minimum de features pour respecter les contraintes mémoire et latence.",
            "fields": fields + out,
            "fusion_strategy": "none",
            "temporal_window": None,
        })

    # ── S10 : Sortie enrichie (confiance + explication) ──
    if not prob.needs_confidence and not prob.needs_explain and prob.task == "classification":
        base_inputs = []
        if prob.has_image:  base_inputs.append(image_field(prob))
        if prob.has_signal: base_inputs.append(signal_field(prob, "stats"))
        if not base_inputs: base_inputs = [signal_field(prob)]
        enriched_out = [
            {"name":"label","type":f"int_{prob.n_classes}classes","role":"output","derivation":"raw"},
            {"name":"confidence","type":"scalar","role":"output","derivation":"raw"},
        ]
        schemas.append({
            "schema_id": "s10_enriched_output",
            "description": "Sortie enrichie avec score de confiance",
            "rationale": "Ajoute un score de confiance à la prédiction — utile pour les décisions à risque.",
            "fields": base_inputs + enriched_out,
            "fusion_strategy": "none",
            "temporal_window": None,
        })

    # ── S11 : PCA / réduction dimensionnelle (si image haute résolution) ──
    if prob.has_image and prob.image_size[0] >= 64:
        fields = [
            image_field(prob, "pca"),
            signal_field(prob, "stats") if prob.has_signal else signal_field(prob),
        ]
        schemas.append({
            "schema_id": "s11_pca_reduced",
            "description": "Réduction dimensionnelle PCA sur l'image",
            "rationale": "Réduit la dimensionnalité des grandes images. Bon compromis vitesse/performance.",
            "fields": fields + out,
            "fusion_strategy": "none",
            "temporal_window": None,
        })

    # ── S12 : Normalisé (si tabulaire ou signal avec distributions variables) ──
    if prob.has_tabular or prob.has_signal:
        fields = []
        if prob.has_image:  fields.append(image_field(prob, "mean"))
        if prob.has_signal: fields.append(signal_field(prob, "normalized"))
        if prob.has_tabular:fields.append({"name":"features_norm","type":"vector_16",
                                           "role":"input","derivation":"normalized"})
        if not fields: fields = [signal_field(prob, "normalized")]
        schemas.append({
            "schema_id": "s12_normalized",
            "description": "Features normalisées (0-1)",
            "rationale": "Normalisation utile quand les signaux ont des échelles très différentes.",
            "fields": fields + out,
            "fusion_strategy": "none",
            "temporal_window": None,
        })

    return schemas[:12]  # max 12


# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT DE GÉNÉRATION
# ══════════════════════════════════════════════════════════════════════════════

def print_report(prob: ProblemProfile, schemas: list, verbose: bool):
    print("\n" + "="*60)
    print("ANALYSE DU BESOIN")
    print("="*60)
    print(prob.summary())
    print(f"\n{len(schemas)} structures générées :\n")
    for s in schemas:
        input_fields = [f["name"] for f in s["fields"] if f["role"] == "input"]
        output_fields = [f["name"] for f in s["fields"] if f["role"] == "output"]
        print(f"  {s['schema_id']:<22}  inputs={input_fields}")
        if verbose:
            print(f"    → {s['rationale']}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Génère une population de structures de datasets candidates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemples :
  python generate.py "Images thermiques de machines + température toutes les 10min, détecter les pannes"
  python generate.py "Tweets → 5 catégories émotionnelles, expliquer les prédictions" --verbose
  python generate.py "Capteurs IoT, Raspberry Pi, latence < 50ms" --output schemas/
        """
    )
    parser.add_argument("description",
        help="Description libre du problème")
    parser.add_argument("--output", default=None, metavar="DIR",
        help="Dossier où écrire les fichiers schema_*.json (défaut: stdout uniquement)")
    parser.add_argument("--n", type=int, default=12,
        help="Nombre max de structures à générer (défaut: 12)")
    parser.add_argument("--model-desc", default=None,
        help="Description de l'architecture du modèle (enrichit le profil)")
    parser.add_argument("--verbose", action="store_true",
        help="Afficher le rationale de chaque structure")
    args = parser.parse_args()

    # ── analyser le besoin ──
    prob = ProblemProfile.from_description(args.description)

    # enrichir avec la description du modèle si fournie
    if args.model_desc:
        mdesc = args.model_desc.lower()
        if any(k in mdesc for k in ("gpu","cuda")) and not prob.has_gpu:
            prob.has_gpu = True
        if any(k in mdesc for k in ("raspberry","embedded","embarqué")) and not prob.embedded:
            prob.embedded = True
        m = re.search(r"(\d+)\s*ms", mdesc)
        if m and not prob.latency_ms:
            prob.latency_ms = int(m.group(1))

    # ── générer les candidats ──
    schemas = generate_candidates(prob, args.model_desc)[:args.n]

    # ── afficher rapport ──
    print_report(prob, schemas, args.verbose)

    # ── écrire les fichiers ou stdout ──
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in schemas:
            path = out_dir / f"{s['schema_id']}.json"
            with open(path, "w") as f:
                json.dump(s, f, indent=2, ensure_ascii=False)
        print(f"Fichiers écrits dans : {out_dir}/")
        print(f"Prochaine étape :")
        print(f"  python evaluate.py {out_dir}/s01_basic.json --verbose")
        print(f"  # ou lancer la boucle complète :")
        print(f"  python run.py --schemas {out_dir}/ --verbose")
    else:
        # stdout : JSON array compact
        print(json.dumps(schemas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # fix pour le bug dans summary() qui référence p au lieu de self
    p = None
    main()
