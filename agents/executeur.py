"""
🤖 EXÉCUTEUR INTELLIGENT - Exécute comme Claude
Comprend → Planifie → Exécute → Réfléchit → Adapte
"""
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

# Imports adaptés pour Lazy_dataset
sys.path.insert(0, str(Path(__file__).parent))

from brains.brain_final import build_optimized_graph
from tools.web_search_tool import WebSearchTool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from planificateur import PlanificateurIntelligent, Tache, Plan

load_dotenv()


class ExecuteurIntelligent:
    """
    Exécuteur intelligent qui travaille comme Claude :
    1. Comprend l'objectif
    2. Décompose en tâches
    3. Exécute séquentiellement
    4. Réfléchit après chaque tâche
    5. Adapte si nécessaire
    """

    def __init__(self, workspace_path: str = None):
        if workspace_path is None:
            workspace_path = os.path.join(os.getcwd(), "agent_smart")

        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Planificateur intelligent
        self.planificateur = PlanificateurIntelligent(str(self.workspace_path / "plans"))

        # Brain et WebSearch (lazy loading)
        self.llm = None
        self.graph = None
        self.web_search = None

        # Historique
        self.history = []

        # Statistiques
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "reflections_done": 0,
            "adaptations_made": 0,
            "plans_created": 0
        }

    def _init_brain(self):
        """Initialise le brain"""
        if self.llm is None:
            try:
                self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
                self.web_search = WebSearchTool()
                self.graph = build_optimized_graph(self.llm, self.web_search)
            except Exception as e:
                print(f"⚠️ Erreur initialisation brain: {e}")
                return False
        return True

    def comprendre_et_planifier(self, objectif: str, contexte: str = "") -> Optional[str]:
        """
        Étape 1: Comprendre et planifier (comme Claude !)

        Args:
            objectif: Ce qu'on veut accomplir
            contexte: Contexte supplémentaire

        Returns:
            ID du plan créé
        """
        if not self._init_brain():
            return None

        print(f"\n{'='*70}")
        print("🧠 ÉTAPE 1: COMPRENDRE ET PLANIFIER")
        print(f"{'='*70}")
        print(f"Objectif: {objectif}")
        print(f"Contexte: {contexte or 'Aucun'}")
        print("-"*70)

        # Utiliser le planificateur intelligent
        plan = self.planificateur.analyser_et_planifier(objectif, contexte)

        if plan:
            self.stats["plans_created"] += 1
            return plan.id
        else:
            print("❌ Erreur lors de la planification")
            return None

    def executer_tache_avec_reflexion(self, plan_id: str, task_id: str) -> Tuple[bool, str]:
        """
        Étape 2: Exécuter une tâche avec réflexion

        Args:
            plan_id: ID du plan
            task_id: ID de la tâche

        Returns:
            (succès, résultat)
        """
        plan = self.planificateur.get_plan(plan_id)
        if not plan:
            return False, "Plan non trouvé"

        task = next((t for t in plan.tasks if t.id == task_id), None)
        if not task:
            return False, "Tâche non trouvée"

        # Marquer en cours
        self.planificateur.mettre_a_jour_tache(plan_id, task_id, "in_progress")

        print(f"\n🔄 Exécution: {task.content}")
        print("-"*70)

        try:
            # Utiliser le brain pour accomplir la tâche
            prompt = self._construire_prompt_tache(plan, task)

            resultat = self._utiliser_brain(prompt)

            # Analyser si c'est un succès
            succes = self._evaluer_succes(task, resultat)

            # Marquer comme terminé
            status = "completed" if succes else "blocked"
            self.planificateur.mettre_a_jour_tache(
                plan_id, task_id,
                status=status,
                result=resultat[:500],
                error=None if succes else "Tâche échouée"
            )

            # Réfléchir après l'exécution (comme Claude !)
            print(f"\n💭 Réflexion...")
            reflexion = self.planificateur.reflechir_apres_tache(
                plan_id, task_id,
                succes=succes,
                resultat=resultat
            )

            if reflexion:
                print(f"   {reflexion}")
                self.stats["reflections_done"] += 1

            if succes:
                self.stats["tasks_completed"] += 1
                print(f"\n✅ Tâche complétée: {task.content}")
            else:
                self.stats["tasks_failed"] += 1
                print(f"\n❌ Tâche échouée: {task.content}")

            return succes, resultat

        except Exception as e:
            error_msg = f"Erreur: {str(e)}"
            self.planificateur.mettre_a_jour_tache(plan_id, task_id, "blocked", error=error_msg)
            self.stats["tasks_failed"] += 1
            print(f"\n❌ {error_msg}")
            return False, error_msg

    def _construire_prompt_tache(self, plan: Plan, task: Tache) -> str:
        """Construit le prompt pour le brain"""
        return f"""OBJECTIF GLOBAL: {plan.objective}

CONTEXTE: {plan.understanding or 'Pas de contexte'}

TAÂCHE À EXÉCUTER: {task.content}

Instructions:
1. Comprends bien ce qui est demandé
2. Fais les recherches nécessaires si besoin
3. Exécute la tâche concrètement
4. Donne un rapport clair de ce qui a été fait

Commence maintenant:"""

    def _utiliser_brain(self, prompt: str) -> str:
        """Utilise le brain pour générer une réponse"""
        try:
            result = self.graph.invoke({
                "question": prompt,
                "history": self.history,
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

            answer = result.get("final_answer", "")

            # Ajouter à l'historique
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": answer})

            return answer

        except Exception as e:
            return f"Erreur brain: {str(e)}"

    def _evaluer_succes(self, task: Tache, resultat: str) -> bool:
        """Évalue si une tâche a réussi"""
        if not resultat:
            return False

        # Si le résultat contient des mots d'erreur
        error_words = ["erreur", "error", "échoué", "failed", "impossible"]
        resultat_lower = resultat.lower()

        for word in error_words:
            if word in resultat_lower:
                return False

        # Sinon, considérer comme succès
        return len(resultat) > 10

    def executer_plan_intelligemment(self, plan_id: str,
                                     pause_entre_taches: float = 2.0) -> Dict:
        """
        Exécute un plan de manière intelligente avec réflexion

        Args:
            plan_id: ID du plan
            pause_entre_taches: Pause en secondes entre les tâches

        Returns:
            Statistiques d'exécution
        """
        plan = self.planificateur.get_plan(plan_id)
        if not plan:
            return {"error": "Plan non trouvé"}

        print(f"\n{'='*70}")
        print(f"🚀 EXÉCUTION INTELLIGENTE DU PLAN")
        print(f"{'='*70}")

        self.planificateur.afficher_plan(plan_id)

        results = {
            "plan_id": plan_id,
            "total_tasks": len(plan.tasks),
            "completed": 0,
            "failed": 0,
            "reflections": 0,
            "start_time": None,
            "end_time": None
        }

        results["start_time"] = time.time()

        try:
            task_number = 0
            while True:
                next_task = self.planificateur.get_next_task(plan_id)
                if not next_task:
                    break

                task_number += 1
                print(f"\n{'='*70}")
                print(f"📋 TÂCHE {task_number}/{len(plan.tasks)}")
                print(f"{'='*70}")

                succes, _ = self.executer_tache_avec_reflexion(plan_id, next_task.id)

                if succes:
                    results["completed"] += 1
                else:
                    results["failed"] += 1

                # Pause entre les tâches
                if task_number < len(plan.tasks):
                    print(f"\n⏸️ Pause ({pause_entre_taches}s) avant la prochaine tâche...")
                    time.sleep(pause_entre_taches)

        except KeyboardInterrupt:
            print("\n\n⚠️ Exécution interrompue")

        except Exception as e:
            print(f"\n\n❌ Erreur: {e}")

        finally:
            results["end_time"] = time.time()
            results["duration"] = results["end_time"] - results["start_time"]

        # Afficher le résultat final
        print(f"\n{'='*70}")
        print(f"📊 RÉSULTATS DE L'EXÉCUTION")
        print(f"{'='*70}")
        print(f"Tâches complétées: {results['completed']}/{results['total_tasks']}")
        print(f"Tâches échouées: {results['failed']}")
        print(f"Réflexions faites: {self.stats['reflections_done']}")
        print(f"Durée totale: {results['duration']:.1f} secondes")
        print(f"{'='*70}\n")

        # Afficher le plan final avec réflexions
        self.planificateur.afficher_plan(plan_id)

        return results

    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        return self.stats.copy()


if __name__ == "__main__":
    # Test de l'exécuteur intelligent
    print("🤖 Test de l'Exécuteur Intelligent\n")

    executeur = ExecuteurIntelligent()

    # Créer un plan avec décomposition automatique
    plan_id = executeur.comprendre_et_planifier(
        "Créer une calculatrice en Python",
        contexte="Application simple avec additions et soustractions"
    )

    if plan_id:
        # Exécuter avec réflexion
        resultats = executeur.executer_plan_intelligemment(plan_id, pause_entre_taches=1.0)

        print("\n🎉 Test terminé !")
        print(f"📊 Stats: {executeur.get_stats()}")
