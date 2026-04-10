"""
🧠 PLANIFICATEUR INTELLIGENT - Utilise le Brain pour comprendre et décomposer
Comme Claude : comprend → décompose → exécute → adapte
"""
import json
import os
import sys
from typing import List, Dict, Literal, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Imports adaptés pour Lazy_dataset
sys.path.insert(0, str(Path(__file__).parent))
from brains.brain_final import build_optimized_graph
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Tache:
    """Représente une tâche individuelle"""
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed", "blocked", "cancelled"]
    priority: Literal["high", "medium", "low"]
    created_at: str
    updated_at: str
    depends_on: List[str]
    result: Optional[str] = None
    error: Optional[str] = None
    reflection: Optional[str] = None  # Réflexion après exécution
    estimated_difficulty: Optional[str] = None  # easy/medium/hard


@dataclass
class Plan:
    """Représente un plan intelligent"""
    id: str
    objective: str
    understanding: str  # Ce que le brain a compris
    approach: str  # Approche proposée
    tasks: List[Tache]
    created_at: str
    updated_at: str
    status: Literal["planning", "active", "completed", "failed", "cancelled"]
    lessons_learned: List[str] = None  # Leçons apprises

    def __post_init__(self):
        if self.lessons_learned is None:
            self.lessons_learned = []


class PlanificateurIntelligent:
    """
    Planificateur qui utilise le Brain ReAct pour :
    1. Comprendre l'objectif
    2. Décomposer en petites tâches
    3. Exécuter séquentiellement
    4. Réfléchir et s'adapter
    """

    def __init__(self, workspace_path: str = None):
        if workspace_path is None:
            workspace_path = os.path.join(os.getcwd(), "agent_smart", "plans")

        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        self.plans: Dict[str, Plan] = {}
        self._load_plans()

        # Brain pour la planification (lazy loading)
        self.llm = None
        self.graph = None

        # Historique pour l'apprentissage
        self.historique = []

    def _init_brain(self):
        """Initialise le brain (lazy loading)"""
        if self.llm is None:
            try:
                self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
                self.graph = build_optimized_graph(self.llm)
            except Exception as e:
                print(f"⚠️ Erreur initialisation brain: {e}")
                return False
        return True

    def _load_plans(self):
        """Charge tous les plans existants"""
        try:
            for plan_file in self.workspace_path.glob("*.json"):
                with open(plan_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    plan = self._dict_to_plan(data)
                    self.plans[plan.id] = plan
        except Exception as e:
            print(f"⚠️ Erreur chargement plans: {e}")

    def _save_plan(self, plan: Plan):
        """Sauvegarde un plan"""
        try:
            plan_file = self.workspace_path / f"{plan.id}.json"
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(self._plan_to_dict(plan), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde plan: {e}")

    def _plan_to_dict(self, plan: Plan) -> dict:
        """Convertit un Plan en dictionnaire"""
        return {
            "id": plan.id,
            "objective": plan.objective,
            "understanding": plan.understanding,
            "approach": plan.approach,
            "tasks": [asdict(task) for task in plan.tasks],
            "created_at": plan.created_at,
            "updated_at": plan.updated_at,
            "status": plan.status,
            "lessons_learned": plan.lessons_learned
        }

    def _dict_to_plan(self, data: dict) -> Plan:
        """Convertit un dictionnaire en Plan"""
        tasks = []
        for task_data in data.get("tasks", []):
            # Nettoyer les champs qui ne sont pas dans le dataclass
            task_data_clean = {k: v for k, v in task_data.items()
                              if k in ["id", "content", "status", "priority",
                                       "created_at", "updated_at", "depends_on",
                                       "result", "error", "reflection", "estimated_difficulty"]}
            tasks.append(Tache(**task_data_clean))

        return Plan(
            id=data["id"],
            objective=data["objective"],
            understanding=data.get("understanding", ""),
            approach=data.get("approach", ""),
            tasks=tasks,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=data["status"],
            lessons_learned=data.get("lessons_learned", [])
        )

    def analyser_et_planifier(self, objective: str, contexte: str = "") -> Plan:
        """
        Utilise le Brain pour analyser et créer un plan intelligent

        Args:
            objective: L'objectif à accomplir
            contexte: Contexte supplémentaire

        Returns:
            Le Plan créé avec compréhension et décomposition
        """
        if not self._init_brain():
            return None

        print(f"\n🧠 ANALYSE INTELLIGENTE")
        print("="*70)
        print(f"Objectif: {objective}")
        print("-"*70)

        # Étape 1: Comprendre l'objectif
        prompt_comprehension = f"""Analyse cet objectif et comprends ce qui est demandé.

OBJECTIF: "{objective}"
CONTEXTE: {contexte}

Réponds en 3 sections:
1. COMPRÉHENSION: Que faut-il vraiment faire ? Quel est le but final ?
2. COMPLEXITÉ: Est-ce simple, moyen ou complexe ? Pourquoi ?
3. APPROCHE: Quelle est la meilleure stratégie pour accomplir cela ?

Sois précis et analytique."""

        try:
            response = self.graph.invoke({
                "question": prompt_comprehension,
                "history": [],
                "trace": [],
                "thoughts": [],
                "actions": [],
                "observations": [],
                "analyzed_input": "",
                "identified_intent": "",
                "constraints": "",
                "generated_options": "",
                "self_correction": "",
                "formatted_thinking": "",
                "final_answer": "",
                "confidence_score": 0.5,
                "uncertainty_factors": [],
                "needs_clarification": False,
                "clarification_questions": [],
                "iteration_count": 0,
                "max_iterations": 2,
                "should_retry": False,
                "retry_reason": "",
                "next_action": "deep_think",
                "complexity_level": "medium",
                "search_query": "",
                "search_results": [],
                "search_performed": False,
                "needs_web_search": False
            })

            comprehension = response.get("final_answer", "")

        except Exception as e:
            comprehension = f"Erreur analyse: {str(e)}"
            return None

        # Étape 2: Décomposer en tâches
        prompt_decomposition = f"""À partir de l'analyse suivante, décompose l'objectif en petites tâches exécutables.

OBJECTIF: "{objective}"

ANALYSE:
{comprehension}

Crée 3-8 tâches spécifiques et exécutables.
Chaque tâche doit être:
- Indépendante (sauf si dépendance nécessaire)
- Mesurable (on sait quand c'est fini)
- De taille raisonnable (pas trop grosse)

Format JSON:
{{
  "tasks": [
    {{"content": "Première tâche", "priority": "high", "difficulty": "easy"}},
    {{"content": "Deuxième tâche", "priority": "medium", "difficulty": "medium"}},
    ...
  ]
}}

IMPORTANT: Réponds SEULEMENT en JSON, sans texte avant ou après."""

        try:
            response = self.graph.invoke({
                "question": prompt_decomposition,
                "history": [],
                "trace": [],
                "thoughts": [],
                "actions": [],
                "observations": [],
                "analyzed_input": "",
                "identified_intent": "",
                "constraints": "",
                "generated_options": "",
                "self_correction": "",
                "formatted_thinking": "",
                "final_answer": "",
                "confidence_score": 0.5,
                "uncertainty_factors": [],
                "needs_clarification": False,
                "clarification_questions": [],
                "iteration_count": 0,
                "max_iterations": 2,
                "should_retry": False,
                "retry_reason": "",
                "next_action": "deep_think",
                "complexity_level": "medium",
                "search_query": "",
                "search_results": [],
                "search_performed": False,
                "needs_web_search": False
            })

            reponse = response.get("final_answer", "")

            # Extraire le JSON
            import re
            json_match = re.search(r'\{.*\}', reponse, re.DOTALL)
            if json_match:
                tasks_data = json.loads(json_match.group())
                tasks_list = tasks_data.get("tasks", [])
            else:
                # Fallback: parser manuellement
                tasks_list = self._parser_manuel(reponse)

        except Exception as e:
            print(f"⚠️ Erreur décomposition, utilisation méthode fallback: {e}")
            tasks_list = self._decomposition_fallback(objective)

        # Créer le plan
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        task_objects = []
        for i, task_data in enumerate(tasks_list):
            task = Tache(
                id=f"{plan_id}_task_{i+1}",
                content=task_data.get("content", f"Tâche {i+1}"),
                status="pending",
                priority=task_data.get("priority", "medium"),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                depends_on=[],
                estimated_difficulty=task_data.get("difficulty", "medium")
            )
            task_objects.append(task)

        plan = Plan(
            id=plan_id,
            objective=objective,
            understanding=comprehension[:500],
            approach="Approche séquentielle avec réflexion",
            tasks=task_objects,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status="active"
        )

        self.plans[plan_id] = plan
        self._save_plan(plan)

        # Afficher le résumé
        print(f"\n✅ Plan créé: {plan_id}")
        print(f"📊 {len(task_objects)} tâches identifiées")
        print(f"🧠 Compréhension: {comprehension[:200]}...\n")

        return plan

    def _parser_manuel(self, reponse: str) -> List[Dict]:
        """Parse manuel si JSON échoue"""
        tasks = []
        lignes = reponse.split('\n')

        current_task = {}
        for ligne in lignes:
            ligne = ligne.strip()
            if not ligne:
                continue

            if 'tâche' in ligne.lower() or 'task' in ligne.lower():
                if current_task:
                    tasks.append(current_task)
                current_task = {"content": ligne, "priority": "medium", "difficulty": "medium"}
            elif 'priorité' in ligne.lower() or 'priority' in ligne.lower():
                if 'high' in ligne.lower():
                    current_task["priority"] = "high"
            elif 'difficult' in ligne.lower():
                if 'easy' in ligne.lower():
                    current_task["difficulty"] = "easy"
                elif 'hard' in ligne.lower():
                    current_task["difficulty"] = "hard"

        if current_task:
            tasks.append(current_task)

        return tasks[:8]  # Maximum 8 tâches

    def _decomposition_fallback(self, objective: str) -> List[Dict]:
        """Décomposition fallback si tout échoue"""
        return [
            {"content": f"Analyser: {objective}", "priority": "high", "difficulty": "easy"},
            {"content": "Planifier l'approche", "priority": "high", "difficulty": "medium"},
            {"content": "Exécuter la solution", "priority": "high", "difficulty": "medium"},
            {"content": "Vérifier et tester", "priority": "medium", "difficulty": "easy"}
        ]

    def reflechir_apres_tache(self, plan_id: str, task_id: str,
                             succes: bool, resultat: str) -> Optional[str]:
        """
        Réfléchit après l'exécution d'une tâche (comme Claude !)

        Args:
            plan_id: ID du plan
            task_id: ID de la tâche
            succes: Si la tâche a réussi
            resultat: Résultat de la tâche

        Returns:
            Réflexion ou leçon apprise
        """
        plan = self.get_plan(plan_id)
        if not plan:
            return None

        task = next((t for t in plan.tasks if t.id == task_id), None)
        if not task:
            return None

        if not self._init_brain():
            return None

        # Réfléchir sur ce qui s'est passé
        prompt_reflection = f"""Réfléchis sur l'exécution de cette tâche.

TÂCHE: {task.content}
SUCCÈS: {succes}
RÉSULTAT: {resultat[:500]}

Questions:
1. Qu'est-ce qui a bien fonctionné ?
2. Qu'est-ce qui pourrait être amélioré ?
3. Y a-t-il des leçons à retenir pour les tâches futures ?

Réflexion courte (2-3 phrases):"""

        try:
            response = self.graph.invoke({
                "question": prompt_reflection,
                "history": [],
                "trace": [],
                "thoughts": [],
                "actions": [],
                "observations": [],
                "analyzed_input": "",
                "identified_intent": "",
                "constraints": "",
                "generated_options": "",
                "self_correction": "",
                "formatted_thinking": "",
                "final_answer": "",
                "confidence_score": 0.5,
                "uncertainty_factors": [],
                "needs_clarification": False,
                "clarification_questions": [],
                "iteration_count": 0,
                "max_iterations": 1,
                "should_retry": False,
                "retry_reason": "",
                "next_action": "quick_response",
                "complexity_level": "simple",
                "search_query": "",
                "search_results": [],
                "search_performed": False,
                "needs_web_search": False
            })

            reflection = response.get("final_answer", "")[:300]

            # Sauvegarder la réflexion
            task.reflection = reflection
            if succes and "leçon" in reflection.lower():
                plan.lessons_learned.append(reflection)

            self._save_plan(plan)

            return reflection

        except Exception as e:
            return None

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Retourne un plan"""
        return self.plans.get(plan_id)

    def lister_plans(self) -> List[Plan]:
        """Liste tous les plans"""
        return sorted(self.plans.values(), key=lambda p: p.updated_at, reverse=True)

    def mettre_a_jour_tache(self, plan_id: str, task_id: str,
                           status: str = None, result: str = None,
                           error: str = None) -> bool:
        """Met à jour une tâche"""
        plan = self.get_plan(plan_id)
        if not plan:
            return False

        task = next((t for t in plan.tasks if t.id == task_id), None)
        if not task:
            return False

        if status:
            task.status = status
        if result is not None:
            task.result = result[:500]
        if error is not None:
            task.error = error[:300]

        task.updated_at = datetime.now().isoformat()
        plan.updated_at = datetime.now().isoformat()

        self._save_plan(plan)
        return True

    def get_next_task(self, plan_id: str) -> Optional[Tache]:
        """Retourne la prochaine tâche à exécuter"""
        plan = self.get_plan(plan_id)
        if not plan or plan.status != "active":
            return None

        pending_tasks = [t for t in plan.tasks if t.status == "pending"]

        if not pending_tasks:
            return None

        # Prioriser par priorité
        priority_order = {"high": 0, "medium": 1, "low": 2}
        pending_tasks.sort(key=lambda t: priority_order.get(t.priority, 1))

        return pending_tasks[0]

    def afficher_plan(self, plan_id: str):
        """Affiche un plan de manière détaillée"""
        plan = self.get_plan(plan_id)
        if not plan:
            print(f"❌ Plan {plan_id} non trouvé")
            return

        print(f"\n{'='*70}")
        print(f"📋 PLAN: {plan.objective}")
        print(f"{'='*70}")

        if plan.understanding:
            print(f"\n🧠 Compréhension:")
            print(f"   {plan.understanding[:300]}...")

        if plan.approach:
            print(f"\n💡 Approche:")
            print(f"   {plan.approach}")

        progress = self.get_progress(plan_id)
        if progress:
            print(f"\n📊 Progression: {progress['completed']}/{progress['total']} ({progress['progress_percent']:.1f}%)")

        print(f"\n📝 Tâches:")
        for i, task in enumerate(plan.tasks, 1):
            status_emoji = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "blocked": "🚫",
                "cancelled": "❌"
            }.get(task.status, "❓")

            priority_emoji = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(task.priority, "⚪")

            difficulty_emoji = {
                "easy": "📗",
                "medium": "📙",
                "hard": "📕"
            }.get(task.estimated_difficulty or "medium", "📙")

            print(f"\n  {status_emoji} Tâche {i}: {task.content} {priority_emoji} {difficulty_emoji}")
            print(f"     Statut: {task.status}")

            if task.reflection:
                print(f"     💭 Réflexion: {task.reflection[:100]}...")

        if plan.lessons_learned:
            print(f"\n📚 Leçons apprises:")
            for leçon in plan.lessons_learned[-3:]:
                print(f"  • {leçon[:100]}...")

        print(f"\n{'='*70}\n")

    def get_progress(self, plan_id: str) -> Dict[str, int]:
        """Retourne la progression d'un plan"""
        plan = self.get_plan(plan_id)
        if not plan:
            return {}

        total = len(plan.tasks)
        completed = sum(1 for t in plan.tasks if t.status == "completed")
        in_progress = sum(1 for t in plan.tasks if t.status == "in_progress")
        pending = sum(1 for t in plan.tasks if t.status == "pending")
        blocked = sum(1 for t in plan.tasks if t.status == "blocked")

        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "blocked": blocked,
            "progress_percent": (completed / total * 100) if total > 0 else 0
        }


if __name__ == "__main__":
    # Test du planificateur intelligent
    print("🧠 Test du Planificateur Intelligent\n")

    planificateur = PlanificateurIntelligent()

    # Créer un plan avec décomposition automatique
    plan = planificateur.analyser_et_planifier(
        "Créer une calculatrice en Python",
        contexte="Doit être simple et bien documentée"
    )

    if plan:
        planificateur.afficher_plan(plan.id)

        # Simuler une réflexion après tâche
        print("📝 Simulation de réflexion après tâche...\n")
        reflection = planificateur.reflechir_apres_tache(
            plan.id,
            plan.tasks[0].id,
            succes=True,
            resultat="Calculatrice créée avec succès"
        )

        if reflection:
            print(f"💭 Réflexion: {reflection}\n")

        print("✅ Test terminé !")
