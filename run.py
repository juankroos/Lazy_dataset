"""
run.py — Boucle principale d'exploration de structures de datasets

Orchestre generate.py + evaluate.py en une seule commande.
Mute la meilleure structure, évalue, keep/revert, checkpoint tous les 10 essais.

Usage :
    # Démarrer depuis une description
    python run.py --desc "Images thermiques + température, détecter pannes, Raspberry Pi"

    # Démarrer depuis des schemas déjà générés
    python run.py --schemas schemas/

    # Avec modèle
    python run.py --desc "..." --model-desc "CNN, 64x64" --model-path model.py

    # Options
    python run.py --desc "..." --max-iter 30 --checkpoint 5 --seeds 3 --verbose
"""

import json, argparse, copy, subprocess, sys, time, random
from pathlib import Path
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# MUTATIONS
# ══════════════════════════════════════════════════════════════════════════════

MUTATIONS = [
    "ADD_STATS",
    "ADD_TEMPORAL",
    "ADD_CROSS",
    "ADD_DERIVED",
    "REPLACE_RAW_WITH_NORM",
    "REMOVE_FIELD",
    "CHANGE_FUSION",
    "ADD_CONFIDENCE",
]

def apply_mutation(schema: dict, mutation: str, rng: random.Random) -> tuple[dict, str]:
    """
    Applique une mutation au schéma. Retourne (nouveau_schema, justification).
    Retourne (None, raison) si la mutation n'est pas applicable.
    """
    s    = copy.deepcopy(schema)
    inputs = [f for f in s["fields"] if f["role"] == "input"]
    raw    = [f for f in inputs if f.get("derivation", "raw") == "raw"]

    if mutation == "ADD_STATS":
        # ajouter stats sur un champ raw signal/vector
        candidates = [f for f in raw if f["type"].startswith("vector_")]
        if not candidates:
            return None, "Aucun champ vector_* brut à dériver"
        target = rng.choice(candidates)
        new_field = {
            "name":       f"{target['name']}_stats",
            "type":       target["type"],
            "role":       "input",
            "derivation": "stats",
        }
        if any(f["name"] == new_field["name"] for f in s["fields"]):
            return None, "Champ stats déjà présent"
        s["fields"].append(new_field)
        return s, f"Ajout stats ({new_field['name']}) sur {target['name']}"

    if mutation == "ADD_TEMPORAL":
        candidates = [f for f in raw if f["type"].startswith("vector_")]
        if not candidates:
            return None, "Aucun champ vector_* brut pour fenêtre temporelle"
        target = rng.choice(candidates)
        current_win = s.get("temporal_window") or 0
        win = rng.choice([w for w in [3, 5, 7, 10] if w != current_win]) or 5
        new_field = {
            "name":       f"{target['name']}_hist_w{win}",
            "type":       target["type"],
            "role":       "input",
            "derivation": f"window({target['name']},{win})",
        }
        if any(f["name"] == new_field["name"] for f in s["fields"]):
            return None, "Fenêtre temporelle similaire déjà présente"
        s["fields"].append(new_field)
        s["temporal_window"] = win
        return s, f"Ajout historique temporel w={win} sur {target['name']}"

    if mutation == "ADD_CROSS":
        matrices = [f for f in inputs if f["type"].startswith("matrix_")]
        vectors  = [f for f in inputs if f["type"].startswith("vector_")]
        if not matrices or not vectors:
            return None, "Besoin d'au moins une image et un signal pour corrélation croisée"
        img = rng.choice(matrices)
        sig = rng.choice(vectors)
        new_field = {
            "name":       f"corr_{img['name']}_{sig['name']}",
            "type":       "scalar",
            "role":       "input",
            "derivation": f"corr({img['name']}_mean,{sig['name']}_mean)",
        }
        if any(f["name"] == new_field["name"] for f in s["fields"]):
            return None, "Corrélation croisée déjà présente"
        s["fields"].append(new_field)
        return s, f"Ajout corrélation croisée {img['name']} × {sig['name']}"

    if mutation == "ADD_DERIVED":
        candidates = [f for f in raw if f["type"].startswith("matrix_")]
        if not candidates:
            candidates = [f for f in raw if f["type"].startswith("vector_")]
        if not candidates:
            return None, "Aucun champ brut à dériver"
        target  = rng.choice(candidates)
        deriv   = rng.choice(["mean", "pca", "fft", "normalized"])
        new_field = {
            "name":       f"{target['name']}_{deriv}",
            "type":       target["type"],
            "role":       "input",
            "derivation": deriv,
        }
        if any(f["name"] == new_field["name"] for f in s["fields"]):
            return None, f"Dérivation {deriv} déjà présente sur {target['name']}"
        s["fields"].append(new_field)
        return s, f"Ajout dérivation {deriv} sur {target['name']}"

    if mutation == "REPLACE_RAW_WITH_NORM":
        if not raw:
            return None, "Aucun champ brut à normaliser"
        target = rng.choice(raw)
        for f in s["fields"]:
            if f["name"] == target["name"]:
                f["derivation"] = "normalized"
                f["name"]       = f"{target['name']}_norm"
        return s, f"Normalisation de {target['name']}"

    if mutation == "REMOVE_FIELD":
        # ne pas supprimer si moins de 2 champs input
        if len(inputs) <= 1:
            return None, "Un seul champ input — suppression impossible"
        # préférer supprimer les champs les plus complexes / redondants
        candidates = [f for f in inputs if f.get("derivation", "raw") != "raw"]
        if not candidates:
            candidates = inputs
        target = rng.choice(candidates)
        s["fields"] = [f for f in s["fields"] if f["name"] != target["name"]]
        return s, f"Suppression de {target['name']} ({target.get('derivation','raw')})"

    if mutation == "CHANGE_FUSION":
        current = s.get("fusion_strategy", "none")
        options = [x for x in ["none", "early_concat", "late_fusion"] if x != current]
        s["fusion_strategy"] = rng.choice(options)
        return s, f"Changement fusion : {current} → {s['fusion_strategy']}"

    if mutation == "ADD_CONFIDENCE":
        outputs = [f for f in s["fields"] if f["role"] == "output"]
        if any(f["name"] == "confidence" for f in outputs):
            return None, "Score de confiance déjà présent"
        s["fields"].append({
            "name": "confidence", "type": "scalar",
            "role": "output", "derivation": "raw"
        })
        return s, "Ajout score de confiance en sortie"

    return None, f"Mutation inconnue : {mutation}"


def mutate(schema: dict, rng: random.Random, tried: list) -> tuple[dict, str, str]:
    """
    Essaie les mutations dans un ordre aléatoire jusqu'à en trouver une applicable.
    Retourne (nouveau_schema, mutation_name, justification).
    """
    mutations_order = MUTATIONS[:]
    rng.shuffle(mutations_order)

    for mutation in mutations_order:
        new_schema, justification = apply_mutation(schema, mutation, rng)
        if new_schema is not None:
            # éviter de retomber sur une structure déjà essayée
            sig = schema_signature(new_schema)
            if sig not in tried:
                return new_schema, mutation, justification

    # fallback : forcer ADD_STATS même si déjà vu
    new_schema, justification = apply_mutation(schema, "ADD_STATS", rng)
    if new_schema:
        return new_schema, "ADD_STATS", justification + " (fallback)"

    return schema, "NO_OP", "Aucune mutation applicable"


def schema_signature(schema: dict) -> str:
    parts = sorted(
        f"{f['name']}:{f.get('derivation','raw')}"
        for f in schema["fields"]
    )
    return "|".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# APPEL À evaluate.py
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluate(schema: dict, all_schemas: list, args, tmp_dir: Path) -> dict:
    """Écrit le schéma dans un fichier tmp et appelle evaluate.py."""
    schema_path = tmp_dir / f"_current_{schema['schema_id']}.json"
    history_path = tmp_dir / "_history.json"

    with open(schema_path, "w") as f:
        json.dump(schema, f)
    with open(history_path, "w") as f:
        json.dump(all_schemas, f)

    cmd = [
        sys.executable, str(Path(__file__).parent / "evaluate.py"),
        str(schema_path),
        "--seeds",       str(args.seeds),
        "--all-schemas", str(history_path),
    ]
    if args.model_desc:
        cmd += ["--model-desc", args.model_desc]
    if args.model_path:
        cmd += ["--model-path", args.model_path, "--epochs", str(args.epochs)]
    if args.verbose:
        cmd += ["--verbose"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # reconstruire le JSON même si du texte verbose est mêlé à la sortie
        # stratégie : trouver le dernier bloc { ... } complet dans stdout
        stdout = result.stdout
        depth, start, best_json = 0, -1, None
        for i, ch in enumerate(stdout):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = stdout[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if "fitness" in parsed:
                            best_json = parsed
                    except json.JSONDecodeError:
                        pass
        if best_json:
            return best_json
        return {"error": result.stderr or result.stdout or "Pas de sortie JSON", "fitness": 0.0}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout (>120s)", "fitness": 0.0}
    except Exception as e:
        return {"error": str(e), "fitness": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT DE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def print_checkpoint(iteration: int, best: dict, best_result: dict,
                     history: list, log: list, args):
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"CHECKPOINT — itération {iteration}")
    print(sep)

    print(f"\nMeilleure structure trouvée")
    print(f"  ID          : {best['schema_id']}")
    print(f"  Fitness     : {best_result['fitness']:.4f}")
    print(f"  Accuracy    : {best_result['accuracy']:.4f}  "
          f"Simplicité  : {best_result['simplicity']:.4f}")
    print(f"  Novelty     : {best_result['novelty']:.4f}  "
          f"Vitesse     : {best_result['speed']:.4f}")
    print(f"  Robustesse  : ±{best_result.get('robustesse', best_result.get('robustness',0)):.4f}  "
          f"Mémoire     : {best_result['memory_mb']:.1f} MB")
    print(f"  Champs input: {[f['name'] for f in best['fields'] if f['role']=='input']}")

    # tableau comparatif top-5
    sorted_log = sorted(log, key=lambda x: x["fitness"], reverse=True)[:5]
    if len(sorted_log) > 1:
        print(f"\nTop-5 structures explorées")
        print(f"  {'ID':<24} {'Fitness':>7} {'Acc':>6} {'Simp':>6} {'Mode':>10} {'Décision':>8}")
        print(f"  {'-'*68}")
        for entry in sorted_log:
            marker = " ← BEST" if entry["schema_id"] == best["schema_id"] else ""
            print(f"  {entry['schema_id']:<24} {entry['fitness']:>7.4f} "
                  f"{entry['accuracy']:>6.4f} {entry['simplicity']:>6.4f} "
                  f"{entry.get('eval_mode','proxy_rf'):>10} "
                  f"{entry['decision']:>8}{marker}")

    # autres structures notables
    notable = [e for e in sorted_log[1:] if e["fitness"] > best_result["fitness"] * 0.85]
    if notable:
        print(f"\nAutres structures intéressantes")
        for entry in notable:
            gap = best_result["fitness"] - entry["fitness"]
            print(f"  {entry['schema_id']:<24} fitness={entry['fitness']:.4f} "
                  f"(−{gap:.4f} vs best)")
            ctx = _context_hint(entry, best_result)
            if ctx:
                print(f"    → meilleure si : {ctx}")

    # mutations explorées
    mutations_tried = [e.get("mutation","?") for e in log]
    from collections import Counter
    mut_counts = Counter(mutations_tried)
    keeps = sum(1 for e in log if e["decision"] == "KEEP")
    reverts = sum(1 for e in log if e["decision"] == "REVERT")
    print(f"\nExploration : {len(log)} itérations  |  {keeps} KEEP  |  {reverts} REVERT")
    print(f"Mutations   : {dict(mut_counts.most_common(4))}")

    print(f"\n{sep}")
    print("Continuer l'exploration ? [O/n] ", end="", flush=True)
    try:
        answer = input().strip().lower()
        return answer not in ("n", "non", "no", "q", "quit")
    except (EOFError, KeyboardInterrupt):
        return False


def _context_hint(entry: dict, best: dict) -> str:
    """Génère une hint contextuelle pour les structures alternatives."""
    hints = []
    if entry["simplicity"] > best["simplicity"] + 0.1:
        hints.append("ressources limitées ou déploiement embarqué")
    if entry.get("explicable", 0) > best.get("explicable", 0):
        hints.append("explicabilité requise")
    if entry["speed"] > best["speed"] + 0.1:
        hints.append("contrainte de latence stricte")
    if entry["novelty"] > 0.7 and entry["accuracy"] > best["accuracy"] * 0.9:
        hints.append("exploration de nouvelles relations entre features")
    return " / ".join(hints) if hints else ""


# ══════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    tmp_dir = Path("/tmp/run_tmp")
    tmp_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)

    # ── 1. Charger ou générer la population initiale ──
    if args.schemas:
        schema_dir = Path(args.schemas)
        initial_schemas = []
        for p in sorted(schema_dir.glob("*.json")):
            with open(p) as f:
                initial_schemas.append(json.load(f))
        print(f"Chargement de {len(initial_schemas)} schémas depuis {schema_dir}/")
    elif args.desc:
        print(f"Génération de la population initiale...")
        cmd = [
            sys.executable, str(Path(__file__).parent / "generate.py"),
            args.desc, "--output", str(tmp_dir / "initial"),
        ]
        if args.model_desc:
            cmd += ["--model-desc", args.model_desc]
        if args.verbose:
            cmd += ["--verbose"]
        subprocess.run(cmd, check=True)
        initial_schemas = []
        for p in sorted((tmp_dir / "initial").glob("*.json")):
            with open(p) as f:
                initial_schemas.append(json.load(f))
        print(f"Population initiale : {len(initial_schemas)} structures\n")
    else:
        print("Erreur : fournir --desc ou --schemas"); sys.exit(1)

    # ── 2. Évaluer la population initiale ──
    print("Évaluation de la population initiale...")
    print(f"{'ID':<26} {'Fitness':>7} {'Acc':>6} {'Simp':>6} {'Nov':>6} {'t(s)':>6}")
    print("-" * 60)

    all_schemas_history = []
    scored_initial = []

    for schema in initial_schemas:
        result = run_evaluate(schema, all_schemas_history, args, tmp_dir)
        result["schema_id"]  = schema["schema_id"]
        result["decision"]   = "INIT"
        result["mutation"]   = "INIT"
        result["iteration"]  = 0
        all_schemas_history.append(schema)
        scored_initial.append((schema, result))
        print(f"  {schema['schema_id']:<24} {result['fitness']:>7.4f} "
              f"{result['accuracy']:>6.4f} {result['simplicity']:>6.4f} "
              f"{result.get('novelty',0):>6.4f} {result.get('train_time_s',0):>6.2f}s")

    # meilleur initial
    best_schema, best_result = max(scored_initial, key=lambda x: x[1]["fitness"])
    print(f"\nMeilleur initial : {best_schema['schema_id']} "
          f"(fitness={best_result['fitness']:.4f})\n")

    # ── 3. Journal ──
    log = []
    for schema, result in scored_initial:
        entry = dict(result)
        entry["schema_id"] = schema["schema_id"]
        entry["mutation"]  = "INIT"
        entry["decision"]  = "INIT"
        log.append(entry)

    tried_signatures = {schema_signature(s) for s in initial_schemas}

    # journal fichier
    log_path = Path(f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    def write_log(entry):
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    for entry in log:
        write_log(entry)

    # ── 4. Boucle d'exploration ──
    iteration    = 0
    no_improve   = 0
    keep_running = True

    while keep_running and iteration < args.max_iter:
        iteration += 1

        # mutation
        new_schema, mutation_name, justification = mutate(best_schema, rng, tried_signatures)
        new_schema["schema_id"] = f"s_iter{iteration:03d}_{mutation_name.lower()[:8]}"
        tried_signatures.add(schema_signature(new_schema))

        if args.verbose:
            print(f"\n[iter {iteration:03d}] {mutation_name} — {justification}")

        # évaluation
        result = run_evaluate(new_schema, all_schemas_history, args, tmp_dir)
        result["schema_id"] = new_schema["schema_id"]
        result["mutation"]  = mutation_name
        result["iteration"] = iteration

        fitness_new  = result.get("fitness", 0.0)
        fitness_prev = best_result.get("fitness", 0.0)

        # keep or revert
        if fitness_new > fitness_prev:
            decision   = "KEEP"
            best_schema = new_schema
            best_result = result
            no_improve  = 0
            all_schemas_history.append(new_schema)
            symbol = "↑"
        else:
            decision  = "REVERT"
            no_improve += 1
            # si très novel, garder quand même en historique pour novelty
            if result.get("novelty", 0) > 0.7:
                all_schemas_history.append(new_schema)
            symbol = "·"

        result["decision"] = decision
        log.append(result)
        write_log(result)

        print(f"  [{iteration:03d}] {mutation_name:<22} "
              f"fitness={fitness_new:.4f} {symbol} "
              f"({decision}) — {justification[:50]}")

        # arrêt anticipé fitness exceptionnel
        if fitness_new > 0.90:
            print(f"\n  ★ FITNESS EXCEPTIONNEL ({fitness_new:.4f}) — arrêt anticipé")
            break

        # arrêt convergence
        if no_improve >= 5:
            print(f"\n  Convergence détectée (5 itérations sans amélioration)")
            break

        # checkpoint
        if iteration % args.checkpoint == 0:
            keep_running = print_checkpoint(
                iteration, best_schema, best_result,
                all_schemas_history, log, args
            )

    # ── 5. Rapport final ──
    print(f"\n{'='*64}")
    print(f"RÉSULTAT FINAL")
    print(f"{'='*64}")
    print(f"\nStructure gagnante : {best_schema['schema_id']}")
    print(f"Fitness            : {best_result['fitness']:.4f}")
    print(f"Accuracy           : {best_result['accuracy']:.4f}")
    print(f"Champs input       : {[f['name'] for f in best_schema['fields'] if f['role']=='input']}")
    print(f"\nSchéma JSON :")
    print(json.dumps(best_schema, indent=2, ensure_ascii=False))

    # sauvegarder le meilleur
    best_path = Path(f"best_schema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(best_path, "w") as f:
        json.dump({"schema": best_schema, "result": best_result}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nSauvegardé dans   : {best_path}")
    print(f"Journal complet   : {log_path}")
    print(f"{'='*64}\n")

    return best_schema, best_result


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Boucle principale de recherche de structure de dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemples :
  python run.py --desc "Images thermiques + température, détecter pannes, Raspberry Pi"
  python run.py --schemas schemas/ --model-desc "CNN, 64x64, 4 classes"
  python run.py --desc "..." --model-path model.py --max-iter 50
        """
    )

    # source de schemas
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--desc",    metavar="DESCRIPTION",
        help="Description libre du problème (génère la population via generate.py)")
    src.add_argument("--schemas", metavar="DIR",
        help="Dossier contenant des fichiers schema_*.json déjà générés")

    # modèle
    parser.add_argument("--model-desc", default=None,
        help="Description de l'architecture du modèle (Mode A)")
    parser.add_argument("--model-path", default=None,
        help="Fichier model.py PyTorch (Mode B1)")
    parser.add_argument("--epochs", type=int, default=5,
        help="Époques d'entraînement PyTorch (défaut: 5)")

    # exploration
    parser.add_argument("--max-iter",   type=int, default=50,
        help="Nombre max d'itérations (défaut: 50)")
    parser.add_argument("--checkpoint", type=int, default=10,
        help="Fréquence des checkpoints humains (défaut: 10)")
    parser.add_argument("--seeds",      type=int, default=3,
        help="Nombre de seeds par évaluation (défaut: 3)")
    parser.add_argument("--seed",       type=int, default=42,
        help="Seed aléatoire pour les mutations (défaut: 42)")

    parser.add_argument("--verbose", action="store_true",
        help="Mode verbeux")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
