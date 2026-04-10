"""
🧠 BRAIN REACT avec WebSearch - VERSION FINALE OPTIMISÉE
- Prompts courts pour éviter rate limits
- Gestion d'erreurs robuste
- Tous les scénarios testés
"""
import sys
import time
from typing import TypedDict, List, Dict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
from datetime import datetime

# UTF-8 pour Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

load_dotenv()

# Import adapté pour Lazy_dataset
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.web_search_tool import WebSearchTool, SearchResult


class ThinkingState(TypedDict):
    question: str
    history: List[Dict]
    trace: List[Dict]
    analyzed_input: str
    identified_intent: str
    constraints: str
    generated_options: str
    self_correction: str
    formatted_thinking: str
    final_answer: str
    thoughts: List[str]
    actions: List[str]
    observations: List[str]
    confidence_score: float
    uncertainty_factors: List[str]
    needs_clarification: bool
    clarification_questions: List[str]
    iteration_count: int
    max_iterations: int
    should_retry: bool
    retry_reason: str
    next_action: Literal["deep_think", "quick_response", "use_web_search", "ask_clarification", "finalize"]
    complexity_level: Literal["simple", "medium", "complex"]
    search_query: str
    search_results: List[SearchResult]
    search_performed: bool
    needs_web_search: bool


class MemorySystem:
    def __init__(self, memory_path: str = "brain_memory.json"):
        self.memory_path = memory_path
        self.data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "patterns": [],
                "stats": {"total_interactions": 0, "avg_confidence": 0.0, "web_searches_used": 0}
            }

    def save(self):
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def add_pattern(self, question_type: str, intent: str, success: bool, confidence: float, used_search: bool = False):
        pattern = {
            "timestamp": datetime.now().isoformat(),
            "question_type": question_type,
            "intent": intent[:200],
            "success": success,
            "confidence": confidence,
            "used_web_search": used_search
        }
        self.data["patterns"].append(pattern)
        self.data["stats"]["total_interactions"] += 1
        if used_search:
            self.data["stats"]["web_searches_used"] = self.data["stats"].get("web_searches_used", 0) + 1

        successful = [p for p in self.data["patterns"] if p["success"]]
        if successful:
            self.data["stats"]["avg_confidence"] = sum(p["confidence"] for p in successful[-20:]) / len(successful[-20:])

        self.save()


# ====================== NODES OPTIMISÉS ======================

def build_initial_thought_node(llm):
    """Analyse initiale - VERSION COURTE"""
    def node(state: ThinkingState):
        prompt = f"""Analyse: "{state['question'][:100]}"

Détermine:
1. Complexité (simple/medium/complex)
2. Besoin WebSearch? (questions actualités/prix/stats)
3. Confiance (0-100)
4. Stratégie (quick_response/deep_think/use_web_search)

Réponds en JSON uniquement:
{{"complexity":"simple","needs_web_search":false,"confidence":90,"strategy":"quick_response"}}"""

        try:
            response = llm.invoke(prompt)
            data = json.loads(response.content)

            return {
                "thoughts": [f"Strategy: {data.get('strategy', 'deep_think')}"],
                "confidence_score": data.get('confidence', 70) / 100,
                "complexity_level": data.get('complexity', 'medium'),
                "needs_web_search": data.get('needs_web_search', False),
                "next_action": data.get('strategy', 'deep_think'),
                "iteration_count": 0,
                "max_iterations": 2,  # Réduit pour économiser des tokens
                "should_retry": False,
                "search_performed": False,
                "search_results": []
            }
        except:
            return {
                "thoughts": ["Strategy: default"],
                "confidence_score": 0.6,
                "complexity_level": "medium",
                "needs_web_search": False,
                "next_action": "deep_think",
                "iteration_count": 0,
                "max_iterations": 2,
                "should_retry": False,
                "search_performed": False,
                "search_results": []
            }
    return node


def build_web_search_node(web_search_tool: WebSearchTool):
    """WebSearch optimisé"""
    def node(state: ThinkingState):
        search_query = state['question'][:100]

        try:
            results = web_search_tool.search(search_query, max_results=3)  # Réduit à 3

            return {
                "search_query": search_query,
                "search_results": results,
                "search_performed": True,
                "thoughts": state.get("thoughts", []) + [f"WebSearch: {len(results)} résultats"],
                "search_results": results
            }
        except Exception as e:
            return {
                "search_query": search_query,
                "search_results": [],
                "search_performed": False,
                "thoughts": state.get("thoughts", []) + [f"WebSearch ERROR"],
                "search_results": []
            }
    return node


def build_analysis_node(llm):
    """Node d'analyse unique - fusion de deep_analysis + reasoning"""
    def node(state: ThinkingState):
        question = state['question'][:200]

        web_info = ""
        if state.get("search_performed") and state.get("search_results"):
            web_info = "\nWeb: " + str(len(state.get("search_results", []))) + " résultats"

        prompt = f"""Analyse: "{question}"{web_info}

En 3 phrases:
1. Intent et contexte
2. Comment répondre
3. Sources web (si dispo)"""

        try:
            response = llm.invoke(prompt)
            analysis = response.content.strip()

            return {
                "analyzed_input": analysis,
                "identified_intent": analysis,
                "thoughts": state.get("thoughts", []) + [f"Analyzed: {analysis[:50]}..."]
            }
        except:
            return {
                "analyzed_input": "Analyse basique",
                "identified_intent": "Intent identifié",
                "thoughts": state.get("thoughts", []) + ["Analyzed (fallback)"]
            }
    return node


def build_simple_response_node(llm):
    """Génère la réponse finale - VERSION SIMPLE"""
    def node(state: ThinkingState):
        question = state['question']
        analysis = state.get('analyzed_input', '')

        web_context = ""
        if state.get("search_performed") and state.get("search_results"):
            results = state.get("search_results", [])[:2]
            web_context = "\n\nSources web:\n" + "\n".join([f"- {r.title}: {r.url}" for r in results])

        prompt = f"""Q: "{question}"
Contexte: {analysis[:200]}{web_context}

Réponds en 1-5 phrases, naturel et utile. Si sources web, cite-les."""

        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()

            # Ajouter sources web si présentes
            if state.get("search_performed") and state.get("search_results"):
                sources = "\n\n📚 Sources:\n"
                for i, r in enumerate(state.get("search_results", [])[:3], 1):
                    sources += f"{i}. {r.title}\n   {r.url}\n"
                answer = answer + sources

            return {
                "final_answer": answer,
                "confidence_score": 0.75
            }
        except Exception as e:
            return {
                "final_answer": f"Désolé, erreur: {str(e)[:50]}",
                "confidence_score": 0.3
            }
    return node


def build_quick_response_node(llm):
    """Réponse ultra-rapide pour questions simples"""
    def node(state: ThinkingState):
        prompt = f"Réponds simplement: {state['question'][:100]}"

        try:
            response = llm.invoke(prompt)
            answer = response.content.strip()

            return {
                "final_answer": answer,
                "formatted_thinking": f"Thinking: Question simple → Réponse directe\n\n{answer}",
                "confidence_score": 0.9
            }
        except:
            return {
                "final_answer": "Je suis là pour vous aider !",
                "formatted_thinking": "Thinking: Mode rapide",
                "confidence_score": 0.7
            }
    return node


# ====================== ROUTING ======================

def route_after_initial(state: ThinkingState) -> str:
    action = state.get("next_action", "deep_think")
    if action == "quick_response":
        return "quick"
    elif action == "use_web_search":
        return "web_search"
    else:
        return "analyze"


# ====================== GRAPH ======================

def build_optimized_graph(llm: ChatGroq, web_search_tool: WebSearchTool = None):
    """Graph optimisé - minimaliste pour éviter rate limits"""
    if web_search_tool is None:
        web_search_tool = WebSearchTool()

    workflow = StateGraph(ThinkingState)

    # Nodes essentiels seulement
    workflow.add_node("initial", build_initial_thought_node(llm))
    workflow.add_node("web_search", build_web_search_node(web_search_tool))
    workflow.add_node("analyze", build_analysis_node(llm))
    workflow.add_node("respond", build_simple_response_node(llm))
    workflow.add_node("quick", build_quick_response_node(llm))

    # Entrée
    workflow.set_entry_point("initial")

    # Routing intelligent
    workflow.add_conditional_edges(
        "initial",
        route_after_initial,
        {
            "quick": "quick",
            "web_search": "web_search",
            "analyze": "analyze"
        }
    )

    # WebSearch → Analyze
    workflow.add_edge("web_search", "analyze")

    # Analyze → Respond
    workflow.add_edge("analyze", "respond")

    # Fins
    workflow.add_edge("respond", END)
    workflow.add_edge("quick", END)

    return workflow.compile()


# ====================== CLI ======================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  BRAIN REACT + WebSearch - VERSION FINALE")
    print("="*70)
    print("\nCommandes: question | 'stats' | 'quit'\n")

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
        web_search = WebSearchTool()
        memory = MemorySystem()
        graph = build_optimized_graph(llm, web_search)
        history = []

        print("✅ Brain prêt!\n")

        while True:
            try:
                user_input = input("Toi: ").strip()

                if user_input.lower() in ["quit", "q", "exit"]:
                    print(f"\n📊 Stats: {memory.data['stats']['total_interactions']} msgs, "
                          f"{memory.data['stats']['avg_confidence']:.1%} conf, "
                          f"{memory.data['stats'].get('web_searches_used', 0)} web\n")
                    break

                if user_input.lower() == "stats":
                    print(f"\n📊 {memory.data['stats']}\n")
                    continue

                if not user_input:
                    continue

                history.append({"role": "user", "content": user_input})

                result = graph.invoke({
                    "question": user_input,
                    "history": history,
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

                confidence = result.get("confidence_score", 0.5)
                emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.5 else "🔴"
                web = result.get("search_performed", False)

                print(f"\n📊 {confidence:.0%}{emoji}{' +🔍Web' if web else ''}")
                print(f"\n🤖 {result.get('final_answer', 'Erreur')}\n")

                history.append({"role": "assistant", "content": result.get('final_answer', '')})

                if confidence >= 0.5:
                    memory.add_pattern(
                        result.get('complexity_level', 'medium'),
                        result.get('identified_intent', ''),
                        True,
                        confidence,
                        web
                    )

            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!\n")
                break
            except Exception as e:
                print(f"\n❌ Erreur: {str(e)[:100]}\n")
                continue

    except Exception as e:
        print(f"\n❌ Init error: {str(e)}")
