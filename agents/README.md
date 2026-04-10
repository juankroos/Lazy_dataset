# 🤖 AGENTS INTELLIGENTS - Intégration Lazy_dataset

## 🎯 Objectif

Intégrer les agents intelligents dans le projet **Lazy_dataset** pour créer un système hybride capable de :

1. **Optimizer les structures de datasets** (fonctionnalité existante)
2. **Utiliser des agents intelligents** pour l'exploration et l'évaluation
3. **Combiner les deux approches** pour une optimisation plus puissante

---

## 🏗️ Structure

```
Lazy_dataset/
├── agents/                          # 🆕 Agents intelligents
│   ├── brains/                      # 🧠 Cerveaux d'agents
│   │   └── brain_final.py          # Brain ReAct + WebSearch
│   ├── tools/                       # 🔧 Outils d'agents
│   │   └── web_search_tool.py      # Recherche web
│   ├── plans/                       # 📋 Plans d'agents
│   ├── planificateur.py            # Planificateur intelligent
│   └── executeur.py                # Exécuteur intelligent
│
├── program.md                       # Spécification du système
├── generate.py                      # Génération de schemas
├── evaluate.py                      # Évaluation de schemas
├── run.py                           # Boucle d'évolution
│
└── README.md                        # Documentation principale
```

---

## 🔄 Intégration Agents + Lazy_dataset

### Concept

Le système **Lazy_dataset** optimise des structures de datasets via mutations et évolutions.

Les **agents intelligents** peuvent :

1. **Explorer l'espace des mutations** de manière plus intelligente
2. **Évaluer les schemas** avec des scénarios réels
3. **Générer des rapports** plus détaillés
4. **S'adapter automatiquement** aux résultats

### Architecture Hybride

```
┌─────────────────────────────────────────────────────────────┐
│                    LAZY_DATASET + AGENTS                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. GENERATEUR                                              │
│     ├── Analyse du besoin (humain ou agent)                 │
│     ├── Génération de population initiale                    │
│     └── Brain ReAct pour comprendre les exigences           │
│                                                              │
│  2. MUTATEUR                                                │
│     ├── Mutations aléatoires (existant)                     │
│     └── Mutations guidées par agent (NOUVEAU)               │
│                                                              │
│  3. ÉVALUATEUR                                              │
│     ├── Évaluation proxy Random Forest (existant)           │
│     └── Évaluation par agent intelligent (NOUVEAU)           │
│         ├── Scénarios de test réels                         │
│         ├── Mesures de performance                         │
│         └── Analyse qualitative                             │
│                                                              │
│  4. ORCHESTRATEUR                                           │
│     ├── Keep-or-revert                                      │
│     ├── Checkpoints humains                                 │
│     └── Recommandations (NOUVEAU - par agent)              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Utilisation

### 1. Mode Classique (Sans Agents)

```bash
# Utilisation normale de Lazy_dataset
python run.py --desc "Images thermiques + température, détecter pannes"
```

### 2. Mode Agent (Nouveau)

```python
# Utilisation avec agent intelligent
from agents.executeur import ExecuteurIntelligent
from agents.brains.brain_final import build_optimized_graph

# Initialiser l'agent
executeur = ExecuteurIntelligent()

# Laisser l'agent analyser et optimiser
resultat = executeur.comprendre_et_planifier(
    "Trouver la meilleure structure de dataset pour détecter des pannes"
)

# L'agent génère et teste des schemas automatiquement
resultats = executeur.executer_plan_intelligent(resultat)
```

### 3. Mode Hybride (Combiné)

```python
# Lazy_dataset génère des schemas
python generate.py "Description du problème" --output schemas/

# L'agent évalue et optimise
python agents/evaluate_with_agent.py --schemas schemas/
```

---

## 🔧 Composants Copiés depuis agent_smart

### 1. Brain ReAct (`brains/brain_final.py`)
- ✅ Routing dynamique (4 chemins)
- ✅ WebSearch intégré
- ✅ Structure ReAct complète
- ✅ Gestion d'erreurs robuste

### 2. WebSearch Tool (`tools/web_search_tool.py`)
- ✅ Recherche DuckDuckGo (gratuit)
- ✅ Support Tavily API (optionnel)
- ✅ Résultats formatés

### 3. Planificateur Intelligent (`planificateur.py`)
- ✅ Analyse d'objectifs
- ✅ Décomposition en tâches
- ✅ Gestion de dépendances

### 4. Exécuteur Intelligent (`executeur.py`)
- ✅ Exécution séquentielle
- ✅ Intégration Brain
- ✅ Gestion d'erreurs

---

## 📊 Modes d'Évaluation

### Mode 1: Proxy (Existant)
```python
# Évaluation rapide avec Random Forest
fitness = evaluate_proxy(schema)
```

### Mode 2: Agent (Nouveau)
```python
# Évaluation avec agent intelligent
fitness = evaluate_with_agent(schema, scenarios)
```

### Mode 3: Hybride (Nouveau)
```python
# Combiner les deux approches
fitness_proxy = evaluate_proxy(schema)
fitness_agent = evaluate_with_agent(schema, scenarios)
fitness_final = 0.6 * fitness_proxy + 0.4 * fitness_agent
```

---

## 🎯 Scénarios d'Utilisation

### Scénario 1: Recherche Automatisée

```python
# L'agent explore automatiquement l'espace des schemas
agent = AgentOptimiseur()
resultat = agent.explorer(
    description="Dataset pour détection de pannes industrielles",
    iterations=50,
    checkpoints=True
)

# L'agent :
# - Génère des schemas
# - Les évalue intelligemment
# - Propose des optimisations
# - Donne des recommandations
```

### Scénario 2: Analyse et Recommandations

```python
# L'agent analyse un schema existant
agent = AgentAnalyseur()
analyse = agent.analyser_schema("schema.json")

# L'agent fournit :
# - Forces du schema
# - Faiblesses
# - Suggestions d'amélioration
# - Alternatives à considérer
```

### Scénario 3: Optimisation Guidée

```python
# L'agent guide l'exploration
agent = AgentGuide()

# L'utilisateur pose une question
reponse =.agent.conseiller(
    "Quel type de dérivées seraient utiles pour des données temporelles ?"
)

# L'agent répond avec :
# - Recommandations spécifiques
# - Justifications
# - Exemples
# - Références
```

---

## 🔧 Configuration

### Variables d'Environnement

```bash
# Pour Groq (LLM des agents)
GROQ_API_KEY=votre_key

# Pour WebSearch (optionnel)
TAVILY_API_KEY=votre_key
```

### Dépendances

```bash
pip install langchain-groq langgraph dotenv
```

---

## 📈 Avantages de l'Intégration

1. **Exploration plus intelligente** : Les agents comprennent le contexte
2. **Évaluation plus riche** : Scénarios réels au lieu de données synthétiques
3. **Recommandations expertes** : Analyse qualitative des résultats
4. **Adaptabilité** : S'adapte automatiquement aux problèmes
5. **Explicabilité** : Peut expliquer pourquoi une structure est meilleure

---

## 🚀 Prochaines Étapes

1. **Adapter les imports** des fichiers copiés
2. **Créer l'évaluateur avec agent** (`agents/evaluate_with_agent.py`)
3. **Créer l'orchestrateur hybride** (`agents/orchestrateur_hybride.py`)
4. **Tester l'intégration** complète
5. **Documenter les résultats**

---

**L'intégration Agents + Lazy_dataset combine le meilleur des deux mondes ! 🚀**

---

*Version: 1.0 - Intégration initiale*
*Date: 2026-04-10*
*Auteur: Adapté depuis agent_smart*
