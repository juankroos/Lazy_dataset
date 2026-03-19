# program.md — Recherche de structure de dataset optimale

> L'agent cherche LA meilleure structure via exploration et évaluation proxy.
> Une fois trouvée, il présente aussi les autres avec leurs stats et contextes.
> L'humain valide tous les 10 essais.

---

## Entrée : description du besoin

L'utilisateur décrit son problème librement. Avant de générer quoi que ce
soit, l'agent extrait ces 5 dimensions et les note explicitement :

```
Tâche            : classification / régression / génération / ...
Modalités        : image · signal · texte · audio · capteurs · ...
Sortie attendue  : label · score · texte · vecteur · ...
Contraintes      : latence · mémoire · compute · déploiement · ...
Priorité         : performance · simplicité · explicabilité · vitesse · ...
```

Si une dimension est absente de la description, faire une hypothèse explicite
et la noter. Ne pas commencer l'exploration sans avoir complété les 5.

---

## Ce que l'agent modifie

Un seul fichier : `schema.json` — la définition des champs du dataset candidat.

```json
{
  "schema_id": "s01_basic",
  "description": "Données brutes image + température",
  "fields": [
    { "name": "image",  "type": "matrix_32x32", "role": "input",  "derivation": "raw" },
    { "name": "temp",   "type": "vector_10",    "role": "input",  "derivation": "raw" },
    { "name": "output", "type": "text_label",   "role": "output", "derivation": "raw" }
  ],
  "fusion_strategy": "none",
  "temporal_window": null
}
```

---

## Mutations autorisées (une seule par itération)

| Mutation          | Description                                    |
|-------------------|------------------------------------------------|
| `ADD_DERIVED`     | Ajouter un champ calculé depuis un existant    |
| `ADD_STATS`       | Ajouter des statistiques sur un signal         |
| `ADD_TEMPORAL`    | Ajouter une fenêtre temporelle                 |
| `ADD_CROSS`       | Ajouter une corrélation entre deux signaux     |
| `REPLACE_RAW`     | Remplacer un champ brut par sa version dérivée |
| `REMOVE_FIELD`    | Supprimer un champ redondant ou bruité         |
| `CHANGE_FUSION`   | Changer la stratégie de fusion                 |
| `ADD_OUTPUT_META` | Enrichir la sortie (confiance, explication)    |

---

## Évaluation proxy

Modèle : RandomForest (ou petit NN 1–2 couches). Cible : < 10s par candidat.

```python
fitness = (0.50 * accuracy) + (0.25 * simplicity) + (0.15 * novelty) + (0.10 * speed)

accuracy   = cross_val_score(RandomForest(), X, y, cv=5).mean()
simplicity = 1 / (1 + nb_champs + 0.5 * nb_derivations)
novelty    = min(mean([edit_distance(schema, s) for s in top5]) / 10, 1.0)
speed      = 1 / (1 + temps_entrainement_secondes)
```

Métriques secondaires (notées mais pas dans fitness) :
- `robustesse` : écart-type du score sur 5 seeds
- `memoire_mb` : taille estimée du dataset
- `explicable`  : 0 = non · 1 = partiel · 2 = oui

---

## Règle keep-or-revert

```
fitness_nouveau > fitness_précédent  →  KEEP   (devient le nouveau candidat de base)
fitness_nouveau ≤ fitness_précédent  →  REVERT (retour au candidat précédent)
```

Exception : si novelty > 0.7, logger dans `experiment_log.md` même en cas
de revert — la structure peut être utile comme point de départ futur.

---

## Population initiale (itération 0)

Ne pas modifier ces 5 structures — elles servent de baseline comparative.

| ID            | Description              | Champs principaux                        |
|---------------|--------------------------|------------------------------------------|
| `s01_basic`   | Brut                     | image, temp, output                      |
| `s02_derived` | Stats dérivées           | image_mean, temp_stats, output           |
| `s03_fusion`  | Concaténation            | concat(image_flat, temp), output         |
| `s04_temporal`| Historique temporel      | image, temp_history(w=5), output         |
| `s05_cross`   | Corrélation croisée      | image, temp, corr(img_mean, temp), output|

---

## Protocole d'une itération

```
1. Lire schema.json courant
2. Choisir UNE mutation — justifier en une phrase (hypothèse)
3. Écrire le nouveau schema.json
4. Appeler evaluate.py → obtenir fitness + métriques secondaires
5. Appliquer keep-or-revert
6. Logger dans experiment_log.md :
   schema_id | mutation | fitness | accuracy | simplicity | novelty | décision
7. Si checkpoint atteint → produire le rapport (voir ci-dessous)
```

---

## Checkpoints humains (tous les 10 essais)

L'agent s'arrête et produit un rapport. Format :

```
=== CHECKPOINT itération N ===

Meilleure structure trouvée
  ID       : [schema_id]
  Fitness  : 0.84  |  Accuracy : 0.89  |  Simplicité : 0.71  |  Vitesse : 0.78
  Schéma   : { ... json complet ... }

  POURQUOI C'EST LA MEILLEURE
  [2–3 phrases basées sur les stats : quelle corrélation elle exploite,
  quel pattern elle capture, pourquoi les autres sont moins bons ici]

  IDÉALE QUAND : [conditions opérationnelles précises]
  ÉVITER SI    : [contre-indications précises]

---

Autres structures intéressantes (déjà calculées, valeur gratuite)

  [schema_id]  fitness=0.79  →  meilleure si [contexte différent]
               accuracy=0.78, simplicité=0.91, vitesse=0.95
               Idéale si : ressources limitées / déploiement embarqué

  [schema_id]  fitness=0.73  →  meilleure si [autre contexte]
               accuracy=0.70, simplicité=0.95, explicable=2
               Idéale si : besoin d'expliquer les prédictions

---

Tableau comparatif complet

| ID            | Fitness | Accuracy | Simplicité | Novelty | Mémoire | Contexte optimal       |
|---------------|---------|----------|------------|---------|---------|------------------------|
| [gagnante]    |  0.84   |   0.89   |    0.71    |  0.68   |  12 MB  | haute performance      |
| [structure 2] |  0.79   |   0.78   |    0.91    |  0.55   |   3 MB  | prod / ressources lim. |
| [structure 3] |  0.73   |   0.70   |    0.95    |  0.62   |   1 MB  | explicabilité requise  |
| s01_basic     |  0.61   |   0.63   |    0.97    |  0.00   |   1 MB  | baseline               |

---

Recommandation
  Étant donné ta priorité ([priorité]) et tes contraintes ([contraintes]),
  commence avec [gagnante].
  Si [condition change] → [structure 2] donne résultats similaires
  avec [avantage concret] au prix de [trade-off].

Continuer l'exploration ? (O/N)
=== FIN CHECKPOINT ===
```

---

## Arrêt automatique

L'agent s'arrête sans attendre le checkpoint si :
- `fitness > 0.90` → signaler immédiatement (résultat exceptionnel)
- Aucune amélioration sur 5 itérations consécutives → convergence atteinte

---

## Fichiers du projet

```
/
├── program.md           ← ce fichier (ne pas modifier)
├── schema.json          ← seul fichier modifié par l'agent
├── evaluate.py          ← script d'évaluation (ne pas modifier)
└── experiment_log.md    ← journal des itérations (alimenté par l'agent)
```
